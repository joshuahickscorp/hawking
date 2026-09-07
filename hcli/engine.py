from __future__ import annotations

from .wall_profile import phase as _wall_phase

import ast
import hashlib
import json
import functools
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .backends import (
    CompletionResult,
    SchemaViolation,
    extract_json_object,
    StructuredOutputContract,
    StructuredOutputExhausted,
    backend_supports_response_format,
    completion_from_openai,
    make_structured_output_contract,
    schema_instruction,
    structured_output_attempts,
    structured_output_record,
)
from .config import Config
from .resources import _atomic_write_text
from .context_budget import (
    ContextBudget,
    PreflightResult,
    preflight,
    resolve as resolve_context_budget,
)
from .events import Event, EventBus
from .goal import GoalCompiler
from .mutation import compile_python_file
from .runtime import store_observed_overlap
from .workspace import Workspace
from .hawking_native import _boundary_trace


_REASONING_KEYS = {
    "reasoning",
    "reasoning_content",
    "hidden_reasoning",
    "chain_of_thought",
    "thinking",
}

# Matches tools/bench/structured_output_probe.py::RESULT_SCHEMA. Do not
# diverge: constrained decoding is only a guarantee if the schema is the same
# contract the system prompt already describes.
HCLI_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["answer", "mutation", "tool_use"]},
        "content": {"type": "string"},
        "operations": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "replace",
                            "create",
                            "replace_file",
                            "insert_before",
                            "insert_after",
                            "append",
                        ],
                    },
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    # LINES, as an alternative to the escaped string above.
                    # Measured: the resident emitted a test body whose bytes
                    # literal carried \\n inside a JSON string and produced
                    # "unexpected character after line continuation character".
                    # A list of plain lines needs no newline escaping at all,
                    # which removes that whole failure mode rather than
                    # returning a better error about it.
                    "old_lines": {"type": "array", "items": {"type": "string"}},
                    "new_lines": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["op", "path"],
                "additionalProperties": False,
            },
        },
        "tests": {"type": "array", "items": {"type": "string"}},
        # The field a tool call can occupy. Before this existed the schema was
        # edit-only: `operations` are file edits, so a model that wanted to LOOK
        # at something had nowhere to say so and answered from whatever the
        # retriever happened to pick. 61 registered tools were unreachable from
        # this path by schema construction -- not unimplemented, impossible.
        #
        # Arguments are flat name/value STRING pairs -- not a nested object and
        # not JSON-encoded-in-a-string. Strict mode cannot express a free-form
        # object, and the JSON-in-a-string version was tried and measurably
        # failed: the local model could not escape quotes inside quotes and blew
        # all three structured-output attempts on "Expecting ',' delimiter".
        # Pairs have nothing to escape. Values are typed back against each
        # tool's own input schema at invoke time, which is where the real
        # contract lives anyway.
        "tool_calls": {
            "type": "array",
            # Matches MAX_TOOL_CALLS_PER_ROUND. A schema that allows fewer calls
            # than the executor will run is a cap the model cannot see past.
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["name", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    # Only the discriminant and the short human-readable result are universal.
    # The three payload arrays are mode-specific: requiring all of them made a
    # read/tool reply spend tokens spelling empty arrays, and made a truncated
    # answer look like a schema failure instead of a small answer. The executor
    # still defaults omitted arrays to [] and applies the mutation evidence gate
    # to `tests` before accepting any edit.
    "required": ["kind", "content"],
    "additionalProperties": False,
}

# Degraded agentic cognition does not need to serialize empty arrays on every
# turn.  HCLI supplies omitted operations/tests/tool_calls as empty lists after
# validation; mutation support remains available when those fields are used.
HCLI_COMPACT_RESULT_SCHEMA: Dict[str, Any] = {
    **HCLI_RESULT_SCHEMA,
    "required": ["kind", "content"],
}

# Worker turns already receive the compiled WorkUnit packet. Repeating the
# full root-oriented policy block and pretty-printed schema on every decision
# consumed ~6K characters in the measured O003 prompt. Keep the validator and
# executor contract unchanged, but send the resident the smaller operational
# envelope it needs for this one decision.
# The compaction above removed the pretty-printed schema but the validator kept
# enforcing it, so the resident was asked for a shape nobody described and its
# replies were rejected for not having it. Measured against a live 4B resident on
# the real captured prompt, same seeds and temperature: WITHOUT a shape line 0/8
# conformant (every reply stringified the action into `content`); WITH it 8/8.
# One line restores the contract at ~200 characters instead of the ~6K block.
_AGENTIC_SYSTEM_PROMPT = """HCLI worker for one bounded WorkUnit. Disk state and deterministic evidence are authority.
Return one JSON object with kind=answer|mutation|tool_use and concise content/action.
Mutation: {"kind":"mutation","content":"what changed","operations":[{"op":"create","path":"dir/file.txt","new_lines":["line one","line two"]}],"tests":["cmd"]}
operations is an ARRAY of 1..20 objects; op is one of replace|create|replace_file|insert_before|insert_after|append; op and path are REQUIRED; path is relative to the workspace ROOT and must NOT begin with "workspace/"; use old_lines/new_lines (arrays of plain lines, no escaping); no other keys are allowed.
tool_use: {"kind":"tool_use","content":"why","tool_calls":[{"tool":"fs.read","arguments":[{"name":"path","value":"dir/file.json"}]}]}
tool_calls is an ARRAY of at most 16 objects; each REQUIRES "tool" (a name from the catalog) and "arguments" (an array of {"name","value"} pairs, values as STRINGS).
No reasoning, markdown, or essay. Paths are workspace-relative; never modify .git. HCLI supplies receipts."""

_AGENTIC_SCHEMA_INSTRUCTION = """
JSON only: kind=answer|mutation|tool_use; content is short.
Mutation: operations is an ARRAY of 1..20 objects, each requiring "op" (replace|create|replace_file|insert_before|insert_after|append) and "path", with old_lines/new_lines arrays; no other keys. tests is an array of commands.
tool_use: tool_calls is an ARRAY of <=16 objects, each requiring "tool" (catalog name) and "arguments" ([{"name","value"}], values as strings). Omit empty arrays and rationale."""

_AGENTIC_TOOL_CATALOG = (
    "fs.read(path,start_line,end_line); fs.search(pattern,root,glob,max_results); "
    "fs.list(path,recursive,glob); filesystem.write(path,content,overwrite); "
    "tests.run(paths); receipt.read(path); git.diff(path); git.status(path); "
    "tools.catalog(focus). Use tools.catalog for omitted signatures."
)

_SYSTEM_PROMPT = """You are the HCLI engineering worker.

You may be backed by a local resident, a local server, or a remote provider;
the provider identity is recorded outside this prompt.  Treat deterministic
evidence and verifier results as authority regardless of which model answers.

MODELS THINK.
TOOLS KNOW.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.

Return exactly one JSON object and nothing else. Keep `content` under 200
characters. Omit mode-specific empty arrays.

For a read-only answer, emit only:
{"kind":"answer","content":"concise final answer"}

For a requested code/file change, emit:
{"kind":"mutation","content":"what changed",
 "operations":[{"op":"replace","path":"workspace/relative/path",
 "old_lines":["exact existing line"],"new_lines":["replacement line"]}],
 "tests":["hcli/test_engine_tool_loop.py"]}

For a read-only inspection, emit:
{"kind":"tool_use","content":"why these calls",
 "tool_calls":[{"tool":"fs.search","arguments":[{"name":"pattern",
 "value":"fn qwen38_batched_prefill_allowed"},{"name":"path",
 "value":"crates/hawking-core/src/model/qwen38_hybrid_decode.rs"}]},
 {"tool":"fs.read","arguments":[{"name":"path",
 "value":"crates/hawking-core/src/model/qwen38_hybrid_decode.rs"},
 {"name":"start_line","value":"280"},{"name":"end_line","value":"300"}]}]}
Results come back as OBSERVATIONS and you are asked again.

An fs.read with no window is truncated at ~4,000 chars ("truncated": true) and
real files here are 100x that. SEARCH FIRST, then read that region. Never edit a
region you have not read.

USE old_lines/new_lines, NOT old_text/new_text, for anything with more than one
line: one entry per line, no trailing "\n" of your own, nothing to escape.
Escaping newlines by hand has cost whole rounds. old_text/new_text remain valid
for a short single-line anchor.

A MUTATION WITH AN EMPTY or omitted "tests" LIST CANNOT BE ACCEPTED. The
verifier records it UNVERIFIED -- reason NO_EVIDENCE -- which is terminal, so
the work is thrown away no matter how good the change was. Name a test that
fails before your change and passes after it. If none exists, write one as part
of the same mutation. Deterministic evidence is the only thing that can accept
work here.

ADMISSIBLE TEST FORMS -- anything else is refused as NOT_ADMITTED and your
mutation is ROLLED BACK, however correct it was:
  path/to/test_x.py                  a bare path to a Python file
  pytest path/to/test_x.py           pytest, optionally with -q -v -x -s --tb=
  python -m pytest path/to/test_x.py
  python path/to/script.py           exactly two tokens
NOT admissible: python -c "...", shell pipelines, &&, redirection, cd, env
assignments, or any command that is not one of the four forms above. If your
check needs arbitrary code, WRITE IT INTO A .py FILE in the same mutation and
name that file as the test.

For an implementation mutation, emit ONE operation against the source file and
keep the combined old/new replacement text under 900 UTF-8 bytes. Make the
smallest executable change justified by the supplied evidence; do not serialize
a whole file, a test rewrite, or a plan.

A NEW TEST FILE IS A SECOND OPERATION AND IS EXPECTED. The rule above bounds
the size of the source edit; it does not forbid the test this contract
requires. Measured: three consecutive rounds read the one-operation rule as
permission to write ONLY the test file, so the function under test was never
added, the test passed before the change as well as after, and
red-before-green refused all three. Adding a function and the test that proves
it is TWO operations -- emit both.


ASK FOR EVERYTHING YOU NEED IN ONE REPLY. ONE REPLY may contain at most 16
tool calls. A tool call costs about a
millisecond, while another model round costs minutes. Request only real paths
needed for this goal, in one bundle, and keep the bundle small. If a tool says
a path is missing, change the path or answer from the observation; never repeat
the identical failed call. Do not invent `hcli/tests/test_engine.py` or
`hcli/tests/__init__.py`: the production module is `hcli/engine.py`, and the
tool error's directory listing/suggestion is authoritative.

PROOF MUST NAME AN EXISTING FILE. For this self-repair loop,
`hcli/test_engine_tool_loop.py` is an admitted proving test; use that exact
path unless the observations name a more specific existing test. Never invent
anything under `hcli/tests/`. A mutation must change executable behavior or
add/repair a test; comment-only, marker-only, whitespace-only, and no-op edits
are invalid and will be rolled back.

`hcli/engine.py` already exists. To change it, use `replace` with an exact
existing anchor; never use `create` for that path. `create` is only for a file
the observations prove does not exist.

Never list the SAME call twice. Duplicates are discarded, not executed, and a
reply that repeats one call has asked for one thing. Ask for fewer calls rather
than filling the bundle with guesses.

Prefer looking over guessing: an answer that says evidence is missing when a
tool could have fetched it is a wrong answer.

Rules:
- kind and content are required; operations/tests/tool_calls are required only
  for the mode that uses them and otherwise may be omitted
- op is exactly one of: replace, create, replace_file, insert_before,
  insert_after, append
- tool_calls is [] unless kind is "tool_use"
- every argument value is a plain string; never nest JSON inside a string
- maximum 20 operations
- paths are workspace-relative
- never modify .git/**
- never return shell commands as mutation authority
- use exact old_text anchors
- use create only for nonexistent files
- if asked only a question, do not mutate
- operations and tests are needed only for a mutation; tool_calls is needed
  only for a tool_use; omitted arrays are treated as empty by HCLI
- do not include reasoning_content, hidden reasoning, chain-of-thought, or <think>
"""

_OBSERVATION_HEADER = "OBSERVATIONS (tool results, this goal):"
_OBSERVATION_SEP = "\n\n----- "


def _observation_blocks(trailing: str) -> List[str]:
    """Split the observations tail into one entry per tool result.

    Splitting on the rendered separator rather than re-deriving from the
    observation list: this operates on the block that will actually be posted,
    which is the only thing whose size the budget cares about.
    """
    if not trailing or _OBSERVATION_HEADER not in trailing:
        return []
    body = trailing.split(_OBSERVATION_HEADER, 1)[1]
    parts = body.split(_OBSERVATION_SEP)
    return [p for p in (part.strip() for part in parts) if p]


def _join_observations(blocks: List[str]) -> str:
    if not blocks:
        return ""
    rendered = "\n\n----- ".join(blocks)
    return (
        f"{_OBSERVATION_HEADER}\n"
        f"[earlier tool results dropped to fit the context window]\n\n"
        f"----- {rendered}"
    )


#: Tokenizer estimates and the real tokenizer disagree by a little. Leave room,
#: because being over by one token costs the entire call.
_CTX_ESTIMATE_MARGIN = 96
#: Observed disagreement between `_estimate_prompt_tokens` and the resident's
#: real tokenizer, with headroom. 5.8% on prose; 25% on a payload carrying
#: Python source -- estimated ~5300 against a real 6605, which overflowed the
#: window and killed the call outright. 30% reserved.
#:
#: A learned ratio cannot cover this on its own: density is a property of the
#: payload, not of history. The first call of a goal is prose and calibrates
#: near 3.0; the second carries a source file and is far denser. So the
#: reserve has to be sized for the worst payload, not the last one.
#:
#: Being over by one token costs the whole call. Reserving too much only
#: shortens a reply.
_CTX_ESTIMATE_ERROR = 0.30
_MAX_TOKENS_FLOOR = 512
# A valid mutation reply is usually 800 to 1500 tokens, but a create operation
# can legitimately carry a small new module and its verifier in one structured
# object. The previous 2048 ceiling exhausted those replies before JSON closed
# (the live repair receipt recorded 2048, then 1709, then 1378-token attempts).
# 4096 still fits the resident's 9728-token position window while giving the
# contract retries room to shorten the reply instead of truncating every try.
_MAX_TOKENS_CEILING = 4096
_CHARS_PER_TOKEN = 3
#: The estimator divides characters by a single constant, but the constant is
#: not one number. Measured on the live resident: markdown prose runs near 3,
#: Python source near 2.4 -- indentation and punctuation are their own tokens.
#: A goal that read one 40 KB source file overflowed the window because the
#: estimate said ~5300 tokens and the tokenizer said 6605, a 25% error against
#: a 12% reserve, and the whole call was refused. So the ratio is LEARNED from
#: the real counts the runtime reports, clamped to a sane band, and the
#: MINIMUM ever seen is used -- the most pessimistic ratio is the safe one,
#: since being over by a token costs the entire call.
_CHARS_PER_TOKEN_FLOOR = 2.0
_CHARS_PER_TOKEN_CEILING = 4.0
#: Reserve when the count is EXACT. Only the chat template's role markers are
#: unaccounted for, since the tokenizer is the resident's own.
_CTX_EXACT_MARGIN = 0.04
#: The share of the usable input ONE tool observation may occupy.
#:
#: It was the whole of it. A single fs.read grew the prompt from 1,925 tokens
#: to 5,348 against a per-request window of 6,745 -- 3,423 tokens for one
#: observation -- and the reply was left 1,107 tokens and truncated mid-object.
#: The scaffolding (system prompt 763, worker packet 110, tool catalog, schema
#: instruction) is ~1,925 tokens before any evidence arrives, so a cap equal to
#: the whole usable input can never be satisfied alongside it.
_MAX_OBSERVATION_SHARE = 0.35


@functools.lru_cache(maxsize=1)
def _exact_tokenizer() -> Any:
    """The resident's own tokenizer, or None.

    Estimating characters-per-token cost this campaign two opposite failures on
    the same 8192-token window: a 25% under-count overflowed it and killed the
    call, and the 30% reserve that fixed the overflow left 1107 completion
    tokens after a 5422-token prompt, so the reply was truncated mid-object.
    There is no ratio that satisfies both. The tokenizer is on disk and
    `tokenizers` is installed, so count instead of guessing.
    """
    try:
        from tokenizers import Tokenizer

        from .hawking_native import _sealed_profile

        profile = _sealed_profile()
        path = getattr(profile, "tokenizer", "") if profile else ""
        if not path or not os.path.exists(path):
            return None
        return Tokenizer.from_file(path)
    except Exception:
        return None
_THINK_OPEN_RE = re.compile(r"<think\b", re.I)

_PATH_TOKEN_RE = re.compile(
    r"""(?:
        (?P<quoted>["'`])(?P<qpath>[^"'`\n]+?)(?P=quoted)
        |
        (?P<path>
            (?:\./|\../)?
            [A-Za-z0-9_.@+~/-]+
            \.
            (?:jsonl|json|py|md|txt|toml|yaml|yml|js|ts|tsx|jsx|rs|c|cc|cpp|h|hpp|sh)
            # A boundary, or a longer extension is silently truncated to a
            # shorter known one: `COMPILE_ECONOMICS.jsonl` was extracted as
            # `COMPILE_ECONOMICS.js`, which does not exist, so `_safe_path`
            # refused it and the packet inlined NO evidence at all. The model
            # was told which file mattered and then had to spend a whole tool
            # round -- about 150 s -- asking for the file we had already named.
            (?![A-Za-z0-9_])
        )
    )""",
    re.VERBOSE,
)

_DIRECTORY_LIST_INTENT_RE = re.compile(
    r"\b(?:"
    r"what(?:['’]s| is)\s+(?:in|inside)\s+(?:this|the\s+current|the)\s+"
    r"(?:directory|folder)|"
    r"directory\s+listing|"
    r"(?:list|show|display)\s+(?:the\s+)?(?:files|contents|entries)"
    r"(?:\s+(?:in|under|inside|at)\s+[^\n]+)?"
    r")\b",
    re.IGNORECASE,
)
_DIRECTORY_PATH_RE = re.compile(
    r"\b(?:in|under|inside|at|of)\s+(?:the\s+)?"
    r"([A-Za-z0-9_.@+~/-]+)",
    re.IGNORECASE,
)
_DIRECTORY_STOPWORDS = frozenset({"this", "current", "the", "a", "an", "directory", "folder"})
_DIRECTORY_MUTATION_RE = re.compile(
    r"\b(?:create|write|edit|modify|delete|remove|change|fix|implement|move|rename)\b",
    re.IGNORECASE,
)



def _truncation_message(
    budget: Any, completion_tokens: Any, prompt_tokens: Any
) -> str:
    """Say what the model ACTUALLY produced, not only what it was offered.

    Reporting the requested budget alone made a runtime that clamped below it
    look identical to a model that genuinely exhausted it. The native adapter
    was capping an explicit 6310-token request at its 2048 default, and every
    receipt said "hit the 6310-token completion budget" -- so the real ceiling
    was invisible in the only artifact anyone reads.
    """
    base = (
        f"model produced {completion_tokens} tokens against a "
        f"{budget}-token completion budget after an {prompt_tokens}-token "
        f"prompt and never closed the JSON object"
    )
    try:
        if completion_tokens is not None and budget is not None and (
            int(completion_tokens) < int(budget)
        ):
            return (
                base + f"; the runtime stopped {int(budget) - int(completion_tokens)}"
                " tokens SHORT of the budget, so the real ceiling is the"
                " runtime's, not this budget"
            )
    except (TypeError, ValueError):
        pass
    return base


_REJECTED_EXCERPT_CHARS = 800


def _degraded_structured_record(
    contract: "StructuredOutputContract",
    *,
    last_text: Optional[str] = None,
    **fields: Any,
) -> Dict[str, Any]:
    """Degraded receipt, plus WHY constrained decoding was not used.

    `features` in this record is the list of things the backend does NOT
    have, which reads like a capability list to anyone holding the receipt:
    mode=degraded with features=[response_format, grammar] looked like the
    engine declined two features it had. It did not have them. The contract
    only exists because supports() answered False for every name in that
    list, so the field was withheld -- a request key the runtime ignores is
    not enforcement, and pretending otherwise is the whole failure mode this
    path exists to avoid.
    """
    record = structured_output_record(
        mode="degraded",
        max_attempts=int(contract.max_attempts),
        features=list(contract.degraded_features),
        **fields,
    )
    record["degraded_features"] = list(contract.degraded_features)
    if contract.grammar_supported:
        record["constrained_decoding"] = "grammar_syntax"
        record["grammar_requested"] = "json"
        record["constrained_decoding_reason"] = (
            "native grammar='json' was sent and enforces JSON syntax only; "
            "schema shape remains prompt+validate+bounded retry"
        )
    else:
        record["constrained_decoding"] = "unavailable"
        record["constrained_decoding_reason"] = (
            "backend.supports() reports "
            + ", ".join(f"{name}=False" for name in contract.degraded_features)
            + "; the field is withheld rather than sent-and-ignored, so an "
            "unclosed JSON object is prevented by prompt+validate+bounded retry, "
            "not structurally"
        )
    if contract.attempt_budgets:
        record["structured_attempt_budgets"] = list(contract.attempt_budgets)
    if contract.repairs:
        record["structured_repairs"] = list(contract.repairs)
    if contract.value_repairs:
        record["structured_value_repairs"] = list(contract.value_repairs)
    if contract.truncation_repairs:
        record["structured_truncation_repairs"] = list(contract.truncation_repairs)
    if last_text is not None:
        # A rejection that does not carry the offending reply is not
        # diagnosable. Every structured-output receipt on disk said "response
        # is not a JSON object" and none of them said WHAT the reply was, so a
        # whole unattended run's failure had to be inferred from a token count.
        # Bounded, and never the prompt -- only what the model wrote back.
        text = str(last_text)
        record["rejected_reply_chars"] = len(text)
        record["rejected_reply_excerpt"] = _rejected_excerpt(text)
        record["rejected_reply_truncated_in_receipt"] = (
            len(text) > _REJECTED_EXCERPT_CHARS
        )
    return record


def is_accepted_work(validation: Any, result: Any = None) -> bool:
    """True only for a mutation that applied AND passed a real check.

    The ok flag alone is not that predicate: an ANSWER also sets ok=True, and
    five consecutive fabricated completions were recorded that way. Counting
    accepted work by ok would have scored all five as successes.
    """
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        return False
    if validation.get("accepted_work") is False:
        return False
    if validation.get("kind") == "read_only":
        return False
    checks = validation.get("checks")
    if not isinstance(checks, list):
        return False
    return any(
        isinstance(c, dict) and c.get("kind") == "test" and c.get("exit_code") == 0
        for c in checks
    )


def validation_failure_message(validation: Any) -> str:
    """Say WHY deterministic validation failed.

    The reason was previously extracted and discarded, so every failure reached
    the mission log as the bare string below. That opacity blocked three
    obligations at once: a failed unit could not be diagnosed from its own
    receipt without re-running it.

    Pure, so it can be tested by calling it rather than by reading the source
    around it -- the first version of this test asserted substrings in a window
    of engine.py and passed under every mutation.
    """
    head = "Deterministic validation failed"
    if not isinstance(validation, dict):
        return f"{head}: validation={str(validation)[:300]}"
    bits = []
    for key in ("reason", "failed", "failures", "returncode",
                "command", "stderr", "tests", "files"):
        value = validation.get(key)
        if value in (None, "", [], {}):
            continue
        bits.append(f"{key}={str(value)[:300]}")
    # The per-test reasons live in `checks`, not at the top level. Omitting it
    # produced a message that named the file the model wrote and nothing about
    # why the run was rejected -- true, useless, and it cost a full diagnostic
    # cycle to notice.
    checks = validation.get("checks")
    if isinstance(checks, list):
        bad = []
        for c in checks:
            if not isinstance(c, dict):
                continue
            if c.get("reason") or c.get("fatal") or c.get("exit_code") not in (None, 0):
                # `requested` is the single most useful field on a rejected
                # test -- it is the command the model actually asked for, and
                # without it NOT_ADMITTED says a form was refused but not which.
                keep = {k: c[k] for k in ("kind", "path", "reason", "exit_code",
                                          "cmd", "requested")
                        if c.get(k) not in (None, "")}
                stderr = str(c.get("stderr") or "")[-200:]
                if stderr:
                    keep["stderr_tail"] = stderr
                bad.append(keep)
        if bad:
            bits.append(f"failing_checks={str(bad)[:600]}")
    if not bits:
        return f"{head}: validation={str(validation)[:300]}"
    return f"{head}: " + "; ".join(bits)


class EngineError(RuntimeError):
    pass


class ContextPreflightError(EngineError):
    """Budget violation detected before the HTTP call to llama-server."""

    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        self.kind = result.kind
        self.shortfall = result.shortfall
        self.demand = result.demand
        self.usable = result.usable
        self.per_request_ctx = result.per_request_ctx
        self.remedy = result.remedy
        super().__init__(
            f"context preflight failed ({result.kind}): demand {result.demand} "
            f"exceeds per-request ctx {result.per_request_ctx} "
            f"(shortfall {result.shortfall}). {result.remedy}"
        )


class NoOpMutation(EngineError):
    """Mutation wrote identical bytes, or an individual op could not change disk."""

    def __init__(
        self,
        detail: str = "",
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.reason = "NO_OP_MUTATION"
        self.files = files or []
        message = "NO_OP_MUTATION"
        if detail:
            message = f"NO_OP_MUTATION: {detail}"
        super().__init__(message)


_COLLECTED_RE = re.compile(r"collected (\d+) items?")
_SUMMARY_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)"
)
_PYTHON_INVOKER_RE = re.compile(r"^python3(\.\d+)?$")
_PROTECTED_PATH_PREFIXES = frozenset({".git", ".hcli"})
_TEST_ENV_KEYS = ("PATH", "HOME", "LANG", "TMPDIR")

def _sha256_bytes(data: Optional[bytes]) -> Optional[str]:
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()


def _is_python_invoker(token: str) -> bool:
    if token in {sys.executable, "python", "python3"}:
        return True
    base = os.path.basename(token)
    if base in {"python", "python3"}:
        return True
    return bool(_PYTHON_INVOKER_RE.fullmatch(base))


def _if_is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]

    def is_name(n: ast.AST) -> bool:
        return isinstance(n, ast.Name) and n.id == "__name__"

    def is_main(n: ast.AST) -> bool:
        return isinstance(n, ast.Constant) and n.value == "__main__"

    return (is_name(left) and is_main(right)) or (
        is_main(left) and is_name(right)
    )


def _looks_like_a_test_filename(path: Path) -> bool:
    """`test_*.py` / `*_test.py`. A file that calls itself a test is one."""
    name = path.name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _ast_is_pytest_idiom(tree: ast.AST) -> bool:
    has_tests = False
    has_main = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                has_tests = True
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                has_tests = True
        elif isinstance(node, ast.If) and _if_is_main_guard(node):
            has_main = True
    return has_tests and not has_main


_TRACEBACK_FRAMES = 40
_TRACEBACK_CHARS = 65536


def _error_message(exc: BaseException) -> str:
    msg = str(exc).strip()
    if msg:
        return msg
    name = type(exc).__name__.strip()
    if name:
        return name
    return "unknown error"


def _error_traceback(exc: BaseException) -> str:
    tb = exc.__traceback__
    if tb is None:
        text = "".join(
            traceback.format_exception_only(type(exc), exc)
        )
    else:
        extracted = traceback.extract_tb(tb)
        if len(extracted) > _TRACEBACK_FRAMES:
            extracted = extracted[-_TRACEBACK_FRAMES:]
        text = "".join(traceback.format_list(extracted))
        text += "".join(
            traceback.format_exception_only(type(exc), exc)
        )
    text = text.strip()
    if not text:
        text = type(exc).__name__.strip() or "traceback unavailable"
    if len(text) > _TRACEBACK_CHARS:
        text = text[-_TRACEBACK_CHARS:]
    return text


def _error_fields(exc: BaseException) -> Dict[str, str]:
    return {
        "error": _error_message(exc),
        "error_type": type(exc).__name__.strip() or "Error",
        "error_traceback": _error_traceback(exc),
    }


_HEARTBEAT_S = 1.0


class _PhaseHeartbeat:
    """Daemon ticks while a blocking call (the model) is in flight.

    The resident transport is request/response JSONL with no token stream,
    so elapsed-time heartbeats are the honest progress signal. Payloads
    never include model text or chain-of-thought.
    """

    def __init__(
        self,
        emit: Callable[..., None],
        *,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
        interval: float = _HEARTBEAT_S,
    ) -> None:
        self._emit = emit
        self._phase = phase
        self._extra = dict(extra or {})
        self._interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def __enter__(self) -> "_PhaseHeartbeat":
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name="hcli-model-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            payload = dict(self._extra)
            payload["phase"] = self._phase
            payload["elapsed_s"] = round(time.perf_counter() - self._t0, 1)
            try:
                self._emit("heartbeat", payload)
            except Exception:
                return

    def __exit__(self, *exc: Any) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        return False


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_SYMBOL_CACHE: Dict[str, Dict[str, int]] = {}


def _python_symbol_lines(source: str) -> Dict[str, int]:
    """Every def/class/assignment in a Python file, mapped to its line.

    Deterministic. `ast` knows exactly where `pid_is_alive` is defined, so
    spending a 150 s resident round -- or a lexical guess that picked the block
    where fs.read is REGISTERED over the function implementing it -- to locate
    it is cognition wasted on a solved problem.
    """
    key = _sha256_bytes(source.encode("utf-8", errors="replace"))
    cached = _SYMBOL_CACHE.get(key)
    if cached is not None:
        return cached
    found: Dict[str, int] = {}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        _SYMBOL_CACHE[key] = found
        return found
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, target.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.setdefault(node.target.id, node.target.lineno)
    if len(_SYMBOL_CACHE) > 64:
        _SYMBOL_CACHE.clear()
    _SYMBOL_CACHE[key] = found
    return found
_DEF_RE = re.compile(r"\s*(?:def|class|fn|pub fn)\s")


_TOP_LEVEL_DEF_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s")


def _focused_excerpt(content: str, prompt: str, limit: int, path: str) -> str:
    """The part of the file the request is ABOUT, not the first `limit` bytes.

    Truncating from the top is the same defect as an fs.read with no offset: a
    goal about `_read_file` at line 509 of a 100 KB file received the first
    5,913 characters -- about line 150 -- and had to spend a tool round asking
    for the rest of a file it had already been handed.

    The window is centred on the densest run of identifiers the prompt and the
    file share, so the excerpt is the neighbourhood of the thing being changed.
    Falls back to the head when nothing matches, which is no worse than before.
    """
    if len(content) <= limit:
        return content

    # Symbols from the OBJECTIVE, not the whole packet. The compiled packet
    # carries INVARIANTS, ACCEPTANCE, NEIGHBORHOOD and any failure context, and
    # those name code the goal never mentioned. Measured: every one of 25
    # consecutive calls anchored on lines 1838-1964, where
    # `default_tool_registry` is defined -- a 21-character name that outranks
    # the `directories_seen` the goal actually names, and which reached `wanted`
    # only through the packet's own scaffolding. The model was shown the tool
    # registration block on every attempt and duly edited it.
    # The OBJECTIVE LINE, when there is one. Cutting at a list of known section
    # headers is a guess about the packet's shape and missed whichever section
    # actually carried the noise: 25 consecutive calls still anchored on
    # `default_tool_registry`. The objective line is where the goal lives, and
    # it is the only part guaranteed to be about the task.
    # The GOAL SENTENCE inside the objective line. A repair objective reads
    #   OBJECTIVE: repair of implement.repair.1: repair of implement:
    #   obligations=G001 <the goal> Relevant files: ...
    # and carries the prior failure's text with it, which is how
    # `default_tool_registry` -- a symbol from the attempt being repaired --
    # reached the symbol set and pulled the window to lines 1838-1964 on every
    # call of every attempt. The goal sits between the obligations marker and
    # the file list.
    focus = prompt or ""
    for line in focus.splitlines():
        if line.lstrip().startswith("OBJECTIVE:"):
            focus = line
            marker = re.search(r"obligations?=[A-Za-z0-9_.,]+\s*", focus)
            if marker:
                focus = focus[marker.end():]
            cut = focus.find("Relevant files:")
            if cut > 0:
                focus = focus[:cut]
            break
    wanted = {w for w in _IDENT_RE.findall(focus) if not w.isupper()}
    lines = content.splitlines(keepends=True)
    if not wanted or not lines:
        return content[:limit]

    # A DEFINITION the parser found beats every lexical guess. Exact symbol
    # first; density is the fallback, not the authority.
    anchor_line: Optional[int] = None
    if path.endswith(".py"):
        symbols = _python_symbol_lines(content)
        named = [(symbols[w], w) for w in wanted if w in symbols]
        if named:
            # The MOST SPECIFIC name, not the earliest-defined one. min() on
            # (line, name) picks whichever matching symbol appears first in the
            # file, so a goal naming `_list_files` and `directories_seen` was
            # anchored on `path` or `glob` -- short names that occur early and
            # everywhere. Measured: the window came out as lines 251-403, which
            # contains TOOL_SCHEMA and neither the function the goal names nor
            # the dict it asks to change, and the model duly edited the return
            # dict it had actually been shown.
            #
            # A longer identifier is a more particular one. Ties go to the
            # earliest definition, as before.
            anchor_line = max(named, key=lambda pair: (len(pair[1]), -pair[0]))[0] - 1

    if os.environ.get("HCLI_DUMP_EVIDENCE"):
        try:
            with open(os.environ["HCLI_DUMP_EVIDENCE"] + ".focus", "a") as h:
                h.write(f"--- path={path} limit={limit}\n")
                h.write(f"    focus[:160]={focus[:160]!r}\n")
                h.write(f"    wanted_sample={sorted(wanted)[:12]}\n")
        except Exception:
            pass
    hits = [i for i, line in enumerate(lines) if any(w in line for w in wanted)]
    if anchor_line is None and not hits:
        return content[:limit]

    # A DEFINITION outweighs a mention. Density alone chose the block where
    # fs.read is registered over the `_read_file` that implements it -- both
    # mention the name, only one is the thing being changed.
    def score(i: int) -> tuple:
        line = lines[i]
        defines = bool(_DEF_RE.match(line)) and any(w in line for w in wanted)
        near = sum(1 for j in hits if abs(j - i) <= 60)
        return (1 if defines else 0, near)

    best = anchor_line if anchor_line is not None else max(hits, key=score)

    # Grow outward from the chosen line, and NEVER trim with a head slice
    # afterwards: expansion is symmetric, so `[:limit]` cuts the tail and can
    # drop the very line the window was centred on. Stop growing at the limit
    # instead, and keep the anchor line even if it alone exceeds it.
    # When the anchor is a DEFINITION, its body comes before its neighbours.
    # Symmetric growth spent half the budget on code ABOVE the function and
    # covered only part of it: a goal naming _read_file received the `def` line
    # and the first half of the body, with the whole-file return dict it was
    # asked to change past the end of the window. The model could not copy an
    # anchor it had never been shown, so it invented one.
    # The next TOP-LEVEL definition, not the next symbol. _python_symbol_lines
    # also reports locals, so the "next symbol" after `def _read_file` was the
    # `path` assigned on the line below it -- the body ended before it began,
    # forward growth did nothing, and the window stopped ten lines short of the
    # return dict the goal names.
    symbol_end = len(lines) - 1
    if anchor_line is not None and path.endswith(".py"):
        for i in range(best + 1, len(lines)):
            if _TOP_LEVEL_DEF_RE.match(lines[i]):
                symbol_end = i - 1
                break

    lo = hi = best
    size = len(lines[best])
    # Forward to the end of the definition first, then outward as usual.
    while hi < symbol_end and hi < len(lines) - 1:
        if size + len(lines[hi + 1]) > limit:
            break
        hi += 1
        size += len(lines[hi])
    while lo > 0 or hi < len(lines) - 1:
        grew = False
        if hi < len(lines) - 1 and size + len(lines[hi + 1]) <= limit:
            hi += 1
            size += len(lines[hi])
            grew = True
        if lo > 0 and size + len(lines[lo - 1]) <= limit:
            lo -= 1
            size += len(lines[lo])
            grew = True
        if not grew:
            break
    body = "".join(lines[lo:hi + 1])
    return (
        f"# {path} lines {lo + 1}-{hi + 1} of {len(lines)}, "
        f"selected around the code this request names\n" + body
    )


def _anchor_violation(path: str, anchor: str, current: str, hits: int) -> str:
    """A retry instruction for an anchor that did not match exactly once."""
    probe = ""
    at = -1
    for line in (l.strip() for l in anchor.strip().splitlines()):
        if len(line) > 5:
            at = current.find(line)
            if at >= 0:
                probe = line
                break
    if hits > 1:
        return (
            f"your old_text for {path} matches {hits} places; it must match "
            f"exactly one. Include more surrounding lines so it is unique."
        )
    if at < 0:
        return (
            f"your old_text for {path} matches nothing in the file -- not one "
            f"line of it. Re-copy the exact bytes from the supplied evidence; "
            f"do not add trailing spaces or a literal backslash-n."
        )
    # The source snapshot is already in the evidence packet. Returning a full
    # actual/sent excerpt here made the retry spend its completion reserve
    # repeating bytes the model could already see; one such repair fell from
    # 1536 to 867 tokens and exhausted all three structured attempts. Keep the
    # diagnosis actionable and bounded, especially for one trailing-space byte.
    return (
        f"your old_text for {path} does not appear in the file. Re-copy the "
        f"exact bytes from the supplied evidence, including newlines but no "
        f"trailing spaces or literal backslash-n escapes. Keep the operation "
        f"short."
    )


_PATCH_BLOCK_RE = re.compile(
    r"PATH:[ \t]*(?P<path>\S+)\s*\n"
    r"FIND:[ \t]*\n(?P<find>.*?)\n?REPLACE:[ \t]*\n(?P<replace>.*?)\n?END\b",
    re.S,
)


def _patch_block_to_operations(text: str) -> Optional[Dict[str, Any]]:
    """A patch the model cannot syntactically break.

    JSON asks for quotes, escapes, brackets and a closing brace before the edit
    is expressible. Measured under every precondition finally satisfied at once
    -- correct objective, evidence containing the target, zero tools, the unique
    anchor supplied, a 2048-token budget -- the resident produced 1,338 to 1,446
    tokens on three consecutive attempts and never closed the object. That is
    the model's own frontier, and no better error message moves it.

    This form has nothing to close:

        PATH: hcli/tool_registry.py
        FIND:
            clipped = raw[:limit]
        REPLACE:
            clipped = raw[:limit]
            total = len(text.splitlines())
        END

    Leading and trailing blank lines are stripped; the interior is taken
    verbatim, so indentation survives. FIND must still match exactly once --
    the applier enforces that, and it is the whole safety property.
    """
    match = _PATCH_BLOCK_RE.search(str(text or ""))
    if match is None:
        return None
    find = match.group("find").strip("\n")
    replace = match.group("replace").strip("\n")
    if not find.strip():
        return None
    return {
        "kind": "mutation",
        "content": "patch block",
        "operations": [{
            "op": "replace",
            "path": match.group("path").strip().rstrip(","),
            "old_text": find,
            "new_text": replace,
        }],
        "tests": [],
        "tool_calls": [],
    }


def _normalize_micro_mutation(parsed: Any) -> Any:
    """Accept the smallest reply that can express one edit.

    The full envelope asks for kind, content, operations, tests and tool_calls
    before a single character of the edit. Measured across a day of attempts,
    the resident reliably decides the right change and unreliably serializes
    that envelope: replies that stop inside an array, anchors wrong in one
    character, bodies whose escaped newlines break the file they describe.

    So let it send the edit and nothing else:

        {"path": "...", "find": "...", "replace": "..."}

    `find` must occur exactly once -- the applier already enforces that, and a
    unique anchor is the whole safety property. Everything downstream keeps
    seeing the full envelope, because this expands into one before it gets
    there. One authority, one shape, no second parser.

    Lists are accepted for find/replace as well, so a multi-line edit still
    needs no newline escaping.
    """
    if not isinstance(parsed, dict):
        return parsed
    if "operations" in parsed or "kind" in parsed:
        return parsed
    path = parsed.get("path")
    if not isinstance(path, str) or "find" not in parsed:
        return parsed

    def _text(value: Any) -> Optional[str]:
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return "\n".join(value) + "\n" if value else ""
        return value if isinstance(value, str) else None

    find = _text(parsed.get("find"))
    replace = _text(parsed.get("replace"))
    if find is None or replace is None:
        return parsed
    return {
        "kind": "mutation",
        "content": str(parsed.get("content") or "micro mutation"),
        "operations": [{
            "op": "replace",
            "path": path,
            "old_text": find,
            "new_text": replace,
        }],
        "tests": parsed.get("tests") or [],
        "tool_calls": [],
    }


def _operation_text(operation: Dict[str, Any], field: str) -> Optional[str]:
    """The text of an operation field, however the model chose to send it.

    `new_text` is one JSON string and therefore carries every newline as an
    escape. The resident got that wrong in a way no better error message fixes:
    a test body containing a bytes literal came back with a stray line
    continuation, three attempts running.

    `new_lines` is the same content as a list of plain lines. There is nothing
    to escape, so there is nothing to get wrong. Lines carry no trailing
    newline of their own; the block ends with one, which is what a Python file
    or a spliced block always does.

    ONE resolver for both the applier and the preflight. Two readers disagreeing
    about what an operation says is exactly the defect that let a bad anchor
    reach _apply_operations with the contract reporting no complaint.
    """
    # Delegates. The resolver lives in hcli/mutation.py so the applier there and
    # the preflight here cannot drift apart -- which they had, and a line-form
    # operation through mutation.py resolved to an EMPTY anchor.
    from .mutation import operation_text

    return operation_text(operation, field)


def blast_radius(before: str, after: str) -> Dict[str, Any]:
    """What a whole-file write did to the file, as a number the receipt keeps.

    Receipt ecf6d616 was ACCEPTED -- mutation, completed, py_compile exit 0,
    4 of 4 tests green, red_before_green True -- for a `replace_file` that cut a
    202-line dashboard to 72 lines and left two names read but never bound. The
    recorded operation carried `op`, `path` and `new_text`, so the only account
    of what the change DID was the model's own prose, which said "add two
    helpers". A narrow test then licensed the deletion and nothing anywhere
    contradicted it.

    This does NOT refuse the write. Legitimate rewrites exist, and a guard that
    blocked them would be a worse defect than the one it closes. It makes the
    size of the change VISIBLE, which is what was missing.
    """
    before_lines = before.count("\n") + (1 if before and not before.endswith("\n") else 0)
    after_lines = after.count("\n") + (1 if after and not after.endswith("\n") else 0)
    removed = max(0, before_lines - after_lines)
    fraction = (removed / before_lines) if before_lines else 0.0
    return {
        "lines_before": before_lines,
        "lines_after": after_lines,
        "lines_removed": removed,
        "fraction_removed": round(fraction, 4),
        "bytes_before": len(before.encode("utf-8")),
        "bytes_after": len(after.encode("utf-8")),
        # A third of a file is a lot to lose to an operation whose stated purpose
        # was to add something. The threshold is a REPORTING trigger, not a veto.
        "mostly_deleted": fraction >= 0.34,
    }


def _rejected_excerpt(text: str) -> str:
    """Keep both ENDS of a rejected reply, not just its head.

    A head-only excerpt spent its whole budget on the `content` prose and cut
    off at the word "operations", which is the one part a rejection about an
    operation needs. Measured: a 2,221-character reply rejected three times for
    a bracket error inside new_text, and all the receipt preserved was 800
    characters of description ending at '"op": "replace",'.

    Half from each end, so the shape of the reply and the thing that broke both
    survive.
    """
    if len(text) <= _REJECTED_EXCERPT_CHARS:
        return text
    half = _REJECTED_EXCERPT_CHARS // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n[... {dropped} characters elided ...]\n{text[-half:]}"


def _python_syntax_violation(content: str) -> Optional[str]:
    """The reply's Python operations must compile, or say why they do not.

    Returns a retry instruction naming the file, line and error, or None when
    every Python operation parses. Non-Python paths are not checked here: the
    verifier owns them and this is only about giving the model back the one
    error it can act on.
    """
    # Parse the reply the way the ENGINE parses it. A bare json.loads returned
    # None for any reply the model wrapped in a markdown fence or prefaced with
    # a sentence -- both of which the engine's own extractor tolerates and then
    # acts on. So the preflight silently did nothing on exactly those replies,
    # and every correction built on it -- the anchor retry, the syntax retry,
    # the quoted line -- was skipped without a trace.
    #
    # Measured: a 343-character anchor correct but for ONE character,
    # 'len(raw}' where 'len(raw)}' belongs, reached _apply_operations and
    # killed the unit with attempts=2 and errors=[] -- the receipt recording
    # that the contract had found nothing to complain about.
    #
    # Two parsers disagreeing about what a reply says is one parser too many.
    if isinstance(content, dict):
        parsed = content
    else:
        try:
            parsed = extract_json_object(content)
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    # ONLY a reply that is actually applying its operations may be judged on
    # them. A tool_use reply is asking to read the file precisely BECAUSE it
    # does not yet know the exact bytes, and it fills operations with a
    # placeholder to satisfy the shape. Judging that placeholder rejected the
    # tool request three times over -- measured: content "need exact unique
    # anchor for _read_file total_lines", old_text "x", refused with "matches
    # 497 places" on every attempt, so the model could never obtain the bytes
    # the refusal was demanding. The check that was added to make anchors
    # correctable had made the correction itself unreachable.
    if str(parsed.get("kind") or "") != "mutation":
        return None
    for op in parsed.get("operations") or []:
        if not isinstance(op, dict):
            continue
        path = str(op.get("path") or "")
        if not path.endswith(".py"):
            continue
        body = _operation_text(op, "new_text")
        if not isinstance(body, str) or not body.strip():
            continue

        # Compile the RESULTING FILE, not the fragment. A replace operation's
        # new_text is spliced into an existing file, so an indented block is
        # correct as a replacement for an indented block -- compiling it
        # standalone reported "unexpected indent at line 1" and rejected
        # patches that were fine. Measured: three consecutive goals died on
        # that false rejection, having been told to fix code that was not
        # broken. A check that refuses correct work is worse than no check.
        candidate = body
        if str(op.get("op") or "") == "create" and Path(path).exists():
            # CORRECTABLE, and it was terminal: _apply_operations runs after
            # the contract accepts, so a unit that offered to create a file
            # already on disk died holding whatever else it had proposed.
            # Measured twice on the same goal, whose spec file the model kept
            # trying to author even though the goal said it already existed.
            return (
                f"{path} already exists, so it cannot be created. If you meant "
                f"to change it, use a replace operation with an exact anchor. "
                f"If it already says what you need, leave it out of your "
                f"operations entirely."
            )
        if str(op.get("op") or "") == "replace":
            anchor = _operation_text(op, "old_text")
            try:
                current = (Path(path)).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not isinstance(anchor, str):
                continue
            hits = current.count(anchor)
            if hits != 1:
                # An anchor that does not match is CORRECTABLE, and until now it
                # was terminal: _apply_operations runs after the contract has
                # accepted the reply, so the unit died holding a patch that was
                # right except for its anchor. Measured: one literal backslash-n
                # where a newline belonged, every other line correct.
                #
                # Route it through the contract's retry instead, and hand back
                # the file's real bytes so the next attempt has something to
                # copy rather than something to guess.
                return _anchor_violation(path, anchor, current, hits)
            candidate = current.replace(anchor, body, 1)

        try:
            compile(candidate, path or "<operation>", "exec")
        except SyntaxError as exc:
            # QUOTE the offending line. A line number in the RESULTING file is
            # a coordinate the model cannot resolve: it never sees that file,
            # only its own new_text. Measured: three attempts on one goal, each
            # told "closing parenthesis '}' does not match opening parenthesis
            # '(' at line 592 of the resulting file", each repeating the same
            # bracket mistake, because nothing in the message showed the line
            # it was talking about.
            where = f"line {exc.lineno}" if exc.lineno else "an unknown line"
            quoted = ""
            if exc.lineno:
                lines = candidate.splitlines()
                lo = max(0, exc.lineno - 2)
                window = lines[lo:exc.lineno + 1]
                if window:
                    numbered = "\n".join(
                        f"{lo + i + 1}: {line}" for i, line in enumerate(window)
                    )
                    quoted = f"\nthe resulting file reads there:\n{numbered}"
            return (
                f"applying your operation to {path} would not compile: "
                f"{exc.msg} at {where} of the resulting file{quoted}"
                f"\nfix that operation and keep it short"
            )
        except ValueError as exc:
            return f"operation on {path} could not be compiled: {exc}"
    return None


_UNCHECKABLE_SOURCE_SUFFIXES = frozenset(
    {".metal", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".swift", ".go"}
)


def _owning_cargo_package(path: Path, root: Path) -> Optional[str]:
    """The Cargo package that owns `path`, from the nearest Cargo.toml ancestor."""
    here = path.parent if path.is_file() or path.suffix else path
    while True:
        manifest = here / "Cargo.toml"
        if manifest.is_file():
            try:
                text = manifest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            in_package = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    in_package = stripped == "[package]"
                    continue
                if in_package and stripped.startswith("name"):
                    _, _, value = stripped.partition("=")
                    return value.strip().strip('"').strip("'") or None
            return None
        if here == root or here.parent == here:
            return None
        here = here.parent


def check_rust_file(path: Path, root: Path) -> Dict[str, Any]:
    """`cargo check` the package owning `path`. Absent cargo is NOT a pass."""
    package = _owning_cargo_package(path, root)
    argv = ["cargo", "check", "--offline"]
    if package:
        argv += ["-p", package]
    try:
        proc = subprocess.run(
            argv, cwd=str(root), capture_output=True, text=True, timeout=900
        )
        return {
            "package": package,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError:
        return {"package": package, "exit_code": 127, "stdout": "",
                "stderr": "cargo not found; a Rust mutation cannot be verified here"}
    except subprocess.TimeoutExpired:
        return {"package": package, "exit_code": 124, "stdout": "",
                "stderr": "cargo check timed out"}


class Engine:
    """Native HCLI execution boundary.

    This is intentionally standalone.  It does not import HAIDER or the
    bootstrap tool bridge.

    The current bridge supports:
      * deterministic named-file evidence
      * GoalCompiler metadata
      * local llama-server structured responses
      * read-only answers
      * structured multi-file mutation
      * complete transaction rollback
      * deterministic Python validation
      * bounded admitted test execution
      * durable receipts
      * cancellation checkpoints
      * reasoning-field suppression

    More advanced WorkUnit scheduling is layered on top of this boundary.
    """

    MAX_EVIDENCE_FILES = 16
    MAX_EVIDENCE_CHARS_PER_FILE = 24000
    MAX_TOTAL_EVIDENCE_CHARS = 120000
    MAX_OPERATIONS = 20
    # A closed tool turn must leave room for the structured mutation itself.
    # The resident's usable window is larger, but allowing a 6K observation
    # transcript here made a second retrieval call consume the whole turn and
    # left no dependable path to an HCLI-authored operation.
    CLOSED_TURN_TARGET_TOKENS = 3900
    # A closed turn also reserves the structured-output instruction. Keep one
    # newest observation small enough that evidence can be abandoned only when
    # necessary and the mutation still has room to decode.
    CLOSED_OBSERVATION_CHARS = 500

    def __init__(
        self,
        workspace: Workspace,
        model_client: Any = None,
        event_bus: Optional[EventBus] = None,
        runtime_provider: Optional[Callable[[], Any]] = None,
        runtime_state_provider: Optional[Callable[[], Any]] = None,
        runtime_count: int = 1,
        model_name: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.workspace = workspace
        self.root = Path(workspace.root).expanduser().resolve()

        self.model_client = model_client
        self.event_bus = event_bus or EventBus()
        self.runtime_provider = runtime_provider
        self.runtime_state_provider = runtime_state_provider
        self.runtime_count = max(1, int(runtime_count))
        self.model_name = model_name or "local"
        self.config = config or Config(str(self.root))

        self.goal_compiler = GoalCompiler()

        self._cancelled = False
        self._active_goal_id: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self._model_calls: List[Dict[str, Any]] = []
        self._last_call_plan: Dict[str, Any] = {}
        self._model_inflight_lock = threading.Lock()
        self._model_inflight = 0
        self.max_model_in_flight = 0
        self._reset_evidence_efficiency()
        from .prefix_probe import PrefixProbe

        # One rendered prompt per live goal, so a turn can be compared with
        # the turn before it. Never a transcript.
        self._prefix_probe = PrefixProbe()
        self._last_rendered_prompt: str = ""
        # (tool, args) already executed in the CURRENT goal. Cleared per goal:
        # a repeat across goals is a different question with the same shape.
        self._tool_calls_seen: Dict[tuple, Dict[str, Any]] = {}

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active_goal_id is not None

    def cancel(self) -> None:
        self._cancelled = True

    def _emit(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.event_bus.emit(
            Event(event_type, data or {})
        )

    @contextmanager
    def _model_call_scope(
        self,
        prompt_tokens: Optional[int] = None,
    ) -> Iterator[None]:
        """Emit model_call started/finished and 1 Hz heartbeats around a block.

        The blocked body is the HTTP/pool/client call. Rendering stays on
        the EventBus; this must not wait on the TUI.
        """
        t0 = time.perf_counter()
        data: Dict[str, Any] = {}
        if self._active_goal_id is not None:
            data["goal_id"] = self._active_goal_id
        if prompt_tokens is not None:
            try:
                data["prompt_tokens"] = int(prompt_tokens)
            except (TypeError, ValueError):
                pass
        self._emit("model_call_started", dict(data))
        ok = False
        try:
            with _PhaseHeartbeat(
                self._emit,
                phase="thinking",
                extra=data,
            ):
                yield
            ok = True
        finally:
            payload: Dict[str, Any] = {
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "ok": ok,
            }
            if self._active_goal_id is not None:
                payload["goal_id"] = self._active_goal_id
            plan = self._last_call_plan or {}
            tokens = plan.get("prompt_tokens_est", prompt_tokens)
            if tokens is not None:
                try:
                    payload["prompt_tokens"] = int(tokens)
                except (TypeError, ValueError):
                    pass
            self._emit("model_call_finished", payload)

    def _cancel_result(
        self,
        goal_id: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        result = {
            "kind": "error",
            "content": "",
            "operations": [],
            "tests": [],
            "cancelled": True,
            "error": "Goal cancelled",
            "error_type": "Cancelled",
            "error_traceback": "Goal cancelled",
            "goal_id": goal_id,
            "status": "cancelled",
        }

        recorded = list(evidence) if evidence is not None else list()
        self._write_receipt(
            goal_id=goal_id,
            goal="",
            result=result,
            evidence=recorded,
            validation=None,
            rolled_back=False,
        )

        self._emit(
            "goal_completed",
            {
                "goal_id": goal_id,
                "status": "cancelled",
            },
        )

        self._emit(
            "final_response",
            {
                "content": "Goal cancelled.",
                "status": "cancelled",
            },
        )

        return result

    # -----------------------------------------------------------------
    # Primary execution
    # -----------------------------------------------------------------

    def complete_text(
        self,
        prompt: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        compiled: Any = None,
    ) -> str:
        """Run provider-neutral plain-text cognition.

        This is a diagnostic and interoperability surface, not a replacement
        for the structured mutation/receipt path.  It deliberately disables
        the HCLI result schema so a provider's raw text can be qualified
        independently before structured mission execution relies on it.
        Empty output is an explicit provider failure, never ``text=""``.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return ""
        self._model_calls = []
        self._last_call_plan = {}
        self._reset_evidence_efficiency()
        evidence = list(evidence or ())
        value = self._call_model(
            prompt,
            evidence=list(evidence),
            compiled=compiled,
            enable_thinking=False,
            response_schema=False,
            plain_text=True,
        )
        if not isinstance(value, str):
            raise EngineError("plain-text cognition returned a non-text result")
        return value

    # One constant used to mean two different things: how many ROUNDS the
    # agentic loop may take, and how many tool calls one round may execute.
    # They are not the same budget and their costs differ by five orders of
    # magnitude -- a round is a model call at 92-385 s, a tool call is 1-3 ms.
    # Capping calls-per-round at the round budget priced milliseconds like
    # minutes.
    # Overridable because a goal that carries its own evidence needs ZERO tool
    # rounds, and each unused round is a full model call. Measured: a 249-token
    # goal that said "do not read any file" made 6 tool calls and arrived at
    # the mutation with a 4,943-token prompt, most of it observations of a file
    # whose bytes were already in the goal.
    MAX_TOOL_ROUNDS = int(os.environ.get("HCLI_MAX_TOOL_ROUNDS", "6"))
    MAX_TOOL_CALLS_PER_ROUND = 16
    # Kept as an alias: external callers and tests referenced the old name for
    # the per-round cap, and silently changing what it means is worse than
    # carrying it.
    # Bounded: an event stream is a trail, not a transcript.
    TOOL_ERROR_EVENT_CHARS = 400

    def _tool_registry(self):
        """The one already built for the executor. Not a second registry."""
        registry = getattr(self, "_tools_cached", None)
        if registry is None:
            from .tool_registry import default_tool_registry

            workspace = getattr(self, "workspace", None)
            # self.workspace is a Workspace object here, not a path; the git
            # root is the right repo_root when the workspace sits inside one.
            root = Path(getattr(workspace, "root", None) or workspace or ".")
            repo_root = Path(getattr(workspace, "git_root", None) or root)
            registry = default_tool_registry(root, repo_root=repo_root)
            self._tools_cached = registry
        return registry

    @classmethod
    def _directory_listing_path(cls, prompt: str) -> Optional[str]:
        """Recognize only unambiguous, read-only directory questions.

        This is deliberately a tiny fast path, not a second natural-language
        planner.  A simple request to see a folder should not require a cold
        27B resident just to discover that ``fs.list`` exists; anything that
        also asks for a mutation stays on the normal model path.
        """
        text = str(prompt or "").strip()
        if not text or not _DIRECTORY_LIST_INTENT_RE.search(text):
            return None
        if _DIRECTORY_MUTATION_RE.search(text):
            return None

        # Quoted paths are the least ambiguous form.  Keep the accepted
        # spelling broad here; ToolContext is the authority that enforces
        # workspace/repository containment when the call is invoked.
        quoted = re.search(r"(?:['\"`])([^'\"`\n]+)(?:['\"`])", text)
        if quoted:
            candidate = quoted.group(1).strip()
            if candidate:
                return candidate

        match = _DIRECTORY_PATH_RE.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in _DIRECTORY_STOPWORDS:
                return candidate
        return "."

    @staticmethod
    def _render_directory_listing(value: Dict[str, Any]) -> str:
        """Turn the bounded ``fs.list`` observation into concise answer text."""
        root = str(value.get("root") or ".")
        files = [item for item in value.get("files", []) if isinstance(item, dict)]
        directories = [
            item for item in value.get("directories", []) if isinstance(item, dict)
        ]
        total = len(files) + len(directories)
        if total == 0:
            return f"Observed directory {root}: empty."

        lines = [
            f"Observed directory {root}: {len(directories)} director"
            f"{'y' if len(directories) == 1 else 'ies'}, {len(files)} file"
            f"{'s' if len(files) != 1 else ''}."
        ]
        for item in directories:
            name = str(item.get("path") or "")
            if name:
                lines.append(f"[directory] {name}/")
        for item in files:
            name = str(item.get("path") or "")
            if not name:
                continue
            size = item.get("bytes")
            suffix = f" ({size} bytes)" if isinstance(size, int) else ""
            lines.append(f"[file] {name}{suffix}")
        if value.get("truncated"):
            lines.append("[listing truncated by the bounded fs.list limit]")
        return "\n".join(lines)

    def _try_deterministic_directory_answer(
        self,
        prompt: str,
        goal_id: str,
        evidence: List[Dict[str, Any]],
        started: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Answer an obvious directory read without starting a model."""
        path = self._directory_listing_path(prompt)
        if path is None:
            return None

        observations = self._run_tool_calls(
            [
                {
                    "tool": "fs.list",
                    "arguments": [
                        {"name": "path", "value": path},
                        {"name": "recursive", "value": "false"},
                        {"name": "max_results", "value": "200"},
                    ],
                }
            ],
            goal_id,
        )
        observation = observations[0] if observations else None
        if not isinstance(observation, dict):
            return None

        if not observation.get("ok"):
            result: Dict[str, Any] = {
                "kind": "error",
                "content": "",
                "operations": [],
                "tests": [],
                "tool_calls": [],
                "error": str(observation.get("text") or "fs.list failed"),
                "status": "failed",
                "goal_id": goal_id,
            }
            validation: Dict[str, Any] = {
                "ok": False,
                "kind": "read_only",
                "tool": "fs.list",
                "reason": "tool_failed",
            }
        else:
            try:
                value = json.loads(str(observation.get("text") or "{}"))
            except (TypeError, ValueError):
                return None
            if not isinstance(value, dict):
                return None
            result = {
                "kind": "answer",
                "content": self._render_directory_listing(value),
                "operations": [],
                "tests": [],
                "tool_calls": [],
                "status": "completed",
                "goal_id": goal_id,
            }
            validation = {
                "ok": True,
                "kind": "read_only",
                # An answer carries NO deterministic evidence. Measured: five
                # autonomous rounds returned answers claiming a function was
                # "already present ... the test passes as expected"; the name
                # existed nowhere, and every one was recorded ok=True. On the
                # ok field alone those receipts were indistinguishable from a
                # verified mutation. This field is what tells them apart.
                "evidence": "none",
                "accepted_work": False,
                "tool": "fs.list",
                "observed_files": len(value.get("files", [])),
                "observed_directories": len(value.get("directories", [])),
                "truncated": bool(value.get("truncated")),
            }

        receipt = self._write_receipt(
            goal_id=goal_id,
            goal=prompt,
            result=result,
            evidence=evidence,
            validation=validation,
            rolled_back=False,
            started=started,
        )
        result["receipt"] = receipt
        self._emit("goal_completed", {"goal_id": goal_id, "status": result["status"]})
        self._emit(
            "final_response",
            {
                "goal_id": goal_id,
                "content": result.get("content") or result.get("error") or "",
                "status": result["status"],
            },
        )
        self.last_result = result
        return result

    @_wall_phase("tools")
    def _run_tool_calls(
        self,
        calls: List[Dict[str, Any]],
        goal_id: str,
    ) -> List[Dict[str, Any]]:
        """Execute what the model asked to look at. A failure is an observation.

        A tool that errors must come back as text the model can read and react
        to, never as an exception that ends the goal -- "that path does not
        exist" is information, and a daemon that dies on a bad argument is not
        unattended.
        """
        # An empty tool bundle means the unit HAS no tools, not that it was
        # asked nicely not to use them. Suppressing the catalog was not enough:
        # the reply schema still offers tool_calls, so the model asked anyway
        # and the executor ran them -- measured, seven tool calls on a goal that
        # carried its own evidence and had no catalog. Refuse here, and say so
        # in terms the model can act on.
        if os.environ.get("HCLI_NO_TOOLS") == "1" and calls:
            return [{
                "tool": str((call or {}).get("tool") or "?"),
                "ok": False,
                "text": (
                    "no tools are available for this work unit. Everything you "
                    "need is already in the goal above. Answer from it."
                ),
            } for call in calls[: self.MAX_TOOL_CALLS_PER_ROUND]]
        registry = self._tool_registry()
        out: List[Dict[str, Any]] = []
        for call in calls[: self.MAX_TOOL_CALLS_PER_ROUND]:
            name = str((call or {}).get("tool") or "").strip()
            args = self._typed_arguments(registry, name, (call or {}).get("arguments"))

            # A repeated call is the loop signature, not a request. Measured:
            # one goal spent five of eleven invocations re-issuing fs.read with
            # the same rejected argument and fs.search with no `pattern`, each
            # costing a whole round. Re-running it would produce the identical
            # observation, so answer from the first one and SAY it is a repeat
            # -- the model cannot break a loop it cannot see.
            key = (name, json.dumps(args, sort_keys=True, default=str))
            # Lazy: the cache is an optimization, not an invariant, and callers
            # that build an Engine without __init__ must still be able to run
            # tools rather than die on a missing attribute.
            seen = getattr(self, "_tool_calls_seen", None)
            if seen is None:
                seen = {}
                self._tool_calls_seen = seen
            prior = seen.get(key)
            if prior is not None:
                repeated = dict(prior)
                note = (
                    f"REPEAT: you already called {name} with these exact "
                    f"arguments in this goal and got the result below. Calling "
                    f"it again cannot change it."
                )
                if not prior.get("ok"):
                    note += (
                        " It FAILED then and fails now for the same reason. "
                        "Change the arguments or answer from what you have."
                    )
                repeated["text"] = f"{note}\n\n{prior.get('text', '')}"
                repeated["repeat"] = True
                out.append(repeated)
                self._emit("tool_call_repeated", {
                    "goal_id": goal_id, "tool": name, "ok": prior.get("ok"),
                })
                continue

            self._emit("tool_call_started", {
                "goal_id": goal_id, "tool": name,
            })
            started = time.perf_counter()
            try:
                result = registry.invoke(name, args)
                ok = bool(getattr(result, "ok", False))
                payload = getattr(result, "value", None) if ok else getattr(result, "error", None)
                text = payload if isinstance(payload, str) else json.dumps(
                    payload, default=str, ensure_ascii=False
                )
            except Exception as exc:  # a tool must never end the goal
                ok, text = False, f"{type(exc).__name__}: {exc}"
            elapsed = round(time.perf_counter() - started, 3)
            if not ok:
                # The signature travels WITH the error. An error that says
                # "missing required property 'pattern'" three rounds away from
                # the catalog is an error the model answers by guessing.
                spec = registry.get(name)
                schema = getattr(spec, "input_schema", None) or {}
                if schema:
                    text = (
                        f"{text}\n"
                        f"signature: {name}("
                        + ", ".join(
                            f"{key}{'*' if key in set(schema.get('required') or []) else ''}"
                            f":{(schema.get('properties', {}).get(key) or {}).get('type', 'string')}"
                            for key in sorted(schema.get("properties") or {})
                        )
                        + ")  (* = required)"
                    )
            observation = {
                "tool": name,
                "ok": ok,
                "text": self._clamp_observation(text),
            }
            seen[key] = observation
            out.append(observation)
            # `ok: false` with no reason is not observability. Five fs.read
            # failures in one goal said only that they failed; the cause (a
            # directory passed where a file was wanted) had to be reproduced by
            # hand afterwards. The reason travels with the event now, bounded.
            failure = None if ok else str(text)[: self.TOOL_ERROR_EVENT_CHARS]
            self._emit("tool_call_finished", {
                "goal_id": goal_id, "tool": name, "ok": ok,
                "elapsed_s": elapsed, "error": failure,
            })
            self._emit("tool_invoked", {
                "goal_id": goal_id, "tool": name, "ok": ok,
                "elapsed_s": elapsed, "error": failure,
            })
        return out

    @staticmethod
    def _tool_catalog(registry) -> str:
        """Names AND arguments. A bare name list makes the model guess.

        Measured: with names only, the model spent its whole tool budget calling
        fs.search without `pattern`, reading the failure, and guessing again. It
        never got to the answer. Required arguments are marked with *.
        """
        try:
            specs = registry.discover()
        except Exception:
            return ""
        lines = []
        for spec in sorted(specs, key=lambda s: s.get("name", "")):
            schema = spec.get("input_schema") or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            args = ", ".join(
                f"{key}{'*' if key in required else ''}:"
                f"{(props[key] or {}).get('type', 'string')}"
                for key in sorted(props)
            )
            lines.append(f"{spec.get('name')}({args})")
        return "\n".join(lines)

    @staticmethod
    def _compact_tool_catalog(
        registry,
        *,
        focus: str = "",
    ) -> str:
        """Render the same registry as a smaller, alias-aware tool index.

        The executor still accepts every registered spelling.  This is only a
        prompt optimization for observation rounds: repeating dozens of
        near-identical names and six alias families consumed roughly 4.4k
        characters on every turn. Exact names remain visible, while a prefix
        is written once and aliases share one signature. The full catalog
        remains available through ``/tools`` and ``tools.catalog`` for
        operator/debugging surfaces.
        """
        try:
            discovered = registry.discover()
        except Exception:
            return ""

        specs = {
            str(item.get("name")): item
            for item in discovered
            if isinstance(item, dict) and item.get("name")
        }

        def signature(spec: Dict[str, Any]) -> str:
            schema = spec.get("input_schema") or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            args = ", ".join(
                f"{key}{'*' if key in required else ''}:"
                f"{(props[key] or {}).get('type', 'string')}"
                for key in sorted(props)
            )
            return f"({args})"

        focus_text = str(focus or "").lower()
        always = {"context", "fs", "filesystem", "git", "tests", "receipt", "shell", "tools"}
        keyword_families = {
            "odyssey": {"odyssey"},
            "huggingface": {"huggingface", "hf", "model", "download", "weights"},
            "web": {"web", "internet", "research", "source", "prior art", "claude", "aider", "mcp"},
            "github": {"github", "repo", "repository", "issue", "commit"},
            "gravity": {"gravity", "compress", "compression", "representation", "context"},
            "frontier": {"frontier", "cloud", "escalate", "swarm", "lane"},
            "doctor": {"doctor", "audit", "measure", "measurement"},
            "accelerator": {"accelerator", "gpu", "ane", "benchmark", "timing"},
            "modellake": {"modellake", "model lake", "acquire", "storage"},
            "specimens": {"specimen", "sealed", "registry"},
            "architecture": {"architecture", "topology", "organ"},
            "forbidden_fruit": {"forbidden fruit", "mlcompute"},
            "grok": {"grok"},
            "acquisition": {"acquisition", "disk", "ssd", "headroom"},
            "roadmap": {"roadmap", "civilization"},
            "vmcp": {"vmcp", "visionmcp"},
            "benchmark": {"benchmark", "pytest", "verification", "verify"},
        }
        selected_prefixes = set(always)
        for prefix, words in keyword_families.items():
            if any(word in focus_text for word in words):
                selected_prefixes.add(prefix)
        for name in specs:
            if name.lower() in focus_text:
                selected_prefixes.add(name.split(".", 1)[0])

        used = set()
        alias_lines: List[str] = []
        for name, spec in sorted(specs.items()):
            if name.split(".", 1)[0] not in selected_prefixes:
                continue
            if name.startswith("git.checkout-safe") and not any(
                word in focus_text
                for word in ("checkout", "revert", "restore", "reset", "discard")
            ):
                # A refused destructive capability is still noise on ordinary
                # read rounds. It re-enters only when the goal explicitly asks
                # about rollback semantics, and even then no mutation is done.
                continue
            aliases = spec.get("aliases", [])
            advertised = [name]
            advertised.extend(alias for alias in aliases if "/" not in alias)
            if aliases:
                alias_lines.append("|".join(advertised) + signature(spec))
                used.add(name)

        by_prefix: Dict[str, List[str]] = {}
        for name in sorted(specs):
            if name in used:
                continue
            prefix, separator, short = name.partition(".")
            if prefix not in selected_prefixes:
                continue
            if not separator:
                by_prefix.setdefault("", []).append(name + signature(specs[name]))
            else:
                by_prefix.setdefault(prefix, []).append(short + signature(specs[name]))

        lines = [
            "EXACT DOTTED TOOL NAMES; aliases joined by |; prefix before : applies to each entry:",
            "Call tools.catalog(focus=...) for exact signatures in an omitted domain.",
        ]
        lines.extend(alias_lines)
        for prefix in sorted(by_prefix):
            entries = by_prefix[prefix]
            if prefix:
                lines.append(f"{prefix}: " + "; ".join(entries))
            else:
                lines.extend(entries)
        return "\n".join(lines)

    @staticmethod
    def _typed_arguments(registry, name: str, pairs: Any) -> Dict[str, Any]:
        """Flat string pairs back into the types each tool's schema declares.

        The model can only emit strings (nesting is what it could not escape),
        but a tool that wants `limit: 5` or `recursive: true` gets INVALID
        ARGUMENTS from a `"5"`. So coerce per the tool's own input schema and
        leave anything unrecognised alone -- the registry still validates, so a
        bad coercion surfaces as a readable tool error, not a wrong call.
        """
        args: Dict[str, Any] = {}
        if not isinstance(pairs, list):
            return args
        spec = registry.get(name)
        schema = getattr(spec, "input_schema", None) or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        props = props if isinstance(props, dict) else {}
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            key = str(pair.get("name") or "").strip()
            if not key:
                continue
            value: Any = pair.get("value")
            declared = props.get(key) or {}
            want = declared.get("type") if isinstance(declared, dict) else None
            want = want[0] if isinstance(want, list) and want else want
            text = "" if value is None else str(value)
            try:
                if want in ("integer", "number"):
                    value = int(text) if want == "integer" else float(text)
                elif want == "boolean":
                    value = text.strip().lower() in ("true", "1", "yes", "on")
                elif want in ("array", "object"):
                    value = json.loads(text)
                else:
                    value = text
            except (ValueError, TypeError):
                value = text  # let the registry reject it with a readable error
            args[key] = value
        return args

    def _prompt_with_observations(
        self,
        prompt: str,
        observations: List[Dict[str, Any]],
        *,
        final: bool = False,
        compact_catalog: bool = False,
        tools_allowed: bool = True,
    ) -> str:
        """Tool output rides beside the goal, NOT inside `evidence`.

        Evidence items are file snapshots: hashed, size/mtime stamped, and
        re-read when they change under us. Tool output is none of those things.
        Putting it in the same list would let unhashed, model-directed content
        pose as deterministic evidence and quietly weaken the freshness gate.
        """
        registry = self._tool_registry()
        # An EMPTY TOOL BUNDLE, scoped to one WorkUnit. A goal that already
        # carries its objective, its target's exact bytes and its failing spec
        # needs no exploratory tool, and offering six is not neutral: measured,
        # a 249-token self-contained goal that said "do not read any file" still
        # spent six tool calls and arrived at the mutation with a 4,943-token
        # prompt, nearly all of it observations of a file already quoted to it.
        #
        # Saying "do not use tools" while advertising them is not scoping. This
        # is. Off by default; the caller opts in per unit.
        if os.environ.get("HCLI_NO_TOOLS") == "1" or not tools_allowed:
            self._tool_catalog_chars = 0
            self._tool_catalog_mode = "none"
            parts = [
                prompt,
                "TOOL ACCESS IS CLOSED FOR THIS ROUND. Do not emit tool_calls. "
                "Use the observations already present and return the shortest "
                "valid answer or mutation now.",
            ]
            if (
                "ROLE: implementation" in str(prompt)
                or "OBJECTIVE: repair" in str(prompt)
            ):
                parts.append(
                    "IMPLEMENTATION CLOSE: This is an implementation/repair "
                    "WorkUnit. Do not answer with measurements, a plan, or a "
                    "status summary. Emit kind=mutation now with the smallest "
                    "justified executable-behavior operation. Emit exactly one "
                    "operation and keep its combined old/new replacement text "
                    "under 900 UTF-8 bytes. Use the existing "
                    "hcli/test_engine_tool_loop.py proving test. Never emit a "
                    "comment-only or invented-test mutation. `hcli/engine.py` "
                    "already exists: never use `create` for it; use `replace` "
                    "with an exact anchor. Old_text is byte-exact: include "
                    "real newlines, never a literal backslash-n, and never "
                    "trailing spaces after the final newline."
                )
            if observations:
                parts.append(self._observations_block(observations, final=True))
            return "\n\n".join(part for part in parts if part)
        if compact_catalog and getattr(self, "_agentic_execution", False):
            catalog = _AGENTIC_TOOL_CATALOG
            catalog_mode = "agentic-minimal"
        else:
            catalog = (
                self._compact_tool_catalog(
                    registry,
                    # The catalog is part of the stable prefix. Including the
                    # names of tools already observed made its contents change
                    # on every round, defeating prefix reuse exactly where
                    # Odyssey pays for long prefill. Observation text remains
                    # the only mutable suffix below.
                    focus=str(prompt or ""),
                )
                if compact_catalog
                else self._tool_catalog(registry)
            )
            catalog_mode = "compact" if compact_catalog else "full"
        self._tool_catalog_chars = len(catalog)
        self._tool_catalog_mode = catalog_mode
        parts = [prompt]
        if catalog:
            parts.append(
                "AVAILABLE TOOLS (name(arg:type), * means required):\n" + catalog
            )
        block = self._observations_block(observations, final=final)
        if block:
            parts.append(block)
        return "\n\n".join(parts)

    def _clamp_observation(self, text: str) -> str:
        """One tool result must never exceed a fraction of the usable window.

        `MAX_EVIDENCE_CHARS_PER_FILE` was 24,000 characters -- about 8,000
        tokens -- against a usable input of 5,632. A single `fs.read` of a large
        file was therefore 1.4x the entire context on its own, so the reduction
        ladder could shed every other observation and still not fit. Measured:
        demand stuck at 12,469 against a 8,192 window with one observation left.

        Derived from the live budget rather than hardcoded, so it stays correct
        if the window changes. A quarter each, so a handful of results coexist
        with the goal and the schema instruction.
        """
        text = str(text or "")
        limit = self.MAX_EVIDENCE_CHARS_PER_FILE
        try:
            usable = int(self._context_budget().usable_input_tokens)
            if usable > 0:
                limit = min(limit, max(1200, usable * _CHARS_PER_TOKEN // 4))
        except Exception:
            pass
        if len(text) <= limit:
            return text
        # Say what to DO about it. The notice used to report that bytes were
        # dropped and stop there, which is the deep-code-unreachable failure in
        # a new costume: hcli/tool_registry.py is 2,341 lines and a whole-file
        # read shows 169 of them, so a target at line 582 is 413 lines past the
        # cut. Measured consequence: the model asked to read the file to obtain
        # an exact anchor, received the head, and sent "x" as the anchor.
        #
        # fs.read takes start_line/end_line and fs.search reports the line a
        # symbol is on, but nothing told the model that at the moment it
        # mattered, and the system prompt names neither -- deliberately, since
        # every token there is re-prefilled on every call of every goal.
        return text[:limit] + (
            f"\n[... {len(text) - limit} characters truncated to fit the context"
            f" window. This is the HEAD of the result only; what you are looking"
            f" for may be past the cut. Use fs.search to find the line a symbol"
            f" is on, then fs.read that file again with start_line and end_line"
            f" to read a window around it.]"
        )

    def _observations_block(
        self,
        observations: List[Dict[str, Any]],
        *,
        final: bool = False,
    ) -> str:
        """The APPEND-ONLY tail. Nothing stable may follow it.

        Observations are the only part of the prompt that grows between rounds,
        so the resident's KV prefix survives exactly up to where this begins.
        Measured on the real builder: with the stable blocks (evidence, the
        durable checkpoint) placed AFTER this, mean reusable prefix was 0.544;
        with this last, 0.924 -- same content, same order of reading for the
        model, 1.7x more prefill the resident can skip.
        """
        parts: List[str] = []
        if observations:
            rendered = "\n\n".join(
                f"----- {o['tool']} [{'ok' if o['ok'] else 'FAILED'}] -----\n{o['text']}"
                for o in observations
            )
            parts.append(f"OBSERVATIONS (tool results, this goal):\n{rendered}")
        if final:
            parts.append(
                "TOOL BUDGET EXHAUSTED. Answer from the observations above. "
                "Do not request more tools."
            )
        return "\n\n".join(parts)

    def _compact_closed_observations(
        self,
        observations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bound already-paid tool evidence before the mutation turn."""
        limit = int(self.CLOSED_OBSERVATION_CHARS)
        compacted: List[Dict[str, Any]] = []
        for observation in observations:
            item = dict(observation)
            text = str(item.get("text") or "")
            if len(text) > limit:
                marker = "\n[... closed-turn observation elided ...]\n"
                room = max(0, limit - len(marker))
                head = room // 2
                tail = room - head
                text = text[:head] + marker + (text[-tail:] if tail else "")
            item["text"] = text
            compacted.append(item)
        return compacted

    def execute(
        self,
        prompt: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        compiled: Any = None,
        *,
        context_memory: Any = None,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()

        if not prompt:
            return {
                "kind": "answer",
                "content": "",
                "operations": [],
                "tests": [],
                "status": "completed",
            }

        goal_id = str(uuid.uuid4())
        self._active_goal_id = goal_id
        # The goal, kept for the evidence window. _gather_evidence receives a
        # string that has already been reduced to path tokens by the time it
        # reaches _focused_excerpt -- measured, it was literally
        # "hcli/tool_registry.py hcli/tests/test_...py" -- so focusing on it
        # anchors on whichever lines mention those paths most densely, which is
        # the tool registration block, not the function the goal names.
        self._active_goal_text = prompt
        self._model_calls = []
        self._tool_calls_seen = {}
        self._last_model_text = None
        self._last_call_plan = {}
        self._agentic_execution = False
        self._reset_evidence_efficiency()

        started = datetime.now(timezone.utc)

        self._emit(
            "goal_started",
            {
                "goal_id": goal_id,
                "text": prompt,
            },
        )

        supplied_evidence = evidence
        supplied_compiled = compiled
        evidence: List[Dict[str, Any]] = list()
        try:
            if self._cancelled:
                return self._cancel_result(goal_id, evidence)

            # Handle the narrow, unambiguous directory read before evidence
            # budgeting.  Budget discovery may ask the controller to start a
            # runtime; a model-free capability must get first refusal on a
            # cold folder, not after the resident has already loaded.
            if supplied_evidence is None and supplied_compiled is None:
                deterministic_directory = self._try_deterministic_directory_answer(
                    prompt,
                    goal_id,
                    evidence,
                    started,
                )
                if deterministic_directory is not None:
                    return deterministic_directory

            self._emit(
                "evidence_gathering_started",
                {"goal_id": goal_id},
            )
            if supplied_evidence is None:
                evidence = self._gather_evidence(prompt)
            else:
                evidence = list(supplied_evidence)
            self._emit(
                "evidence_gathering_finished",
                {
                    "goal_id": goal_id,
                    "file_count": len(evidence),
                },
            )

            if supplied_compiled is None:
                try:
                    compiled = self.goal_compiler.compile(prompt)
                except Exception:
                    compiled = {}
            else:
                compiled = supplied_compiled

            self._emit(
                "goal_compiled",
                {
                    "goal_id": goal_id,
                    "evidence_files": [
                        item["path"]
                        for item in evidence
                    ],
                    "workunits": self._compiled_workunit_count(
                        compiled
                    ),
                },
            )

            if self._cancelled:
                return self._cancel_result(goal_id, evidence)

            # The agentic loop. Everything above gathers evidence DETERMINISTICALLY
            # and then asks once; a model that needed one more file could only
            # answer "no evidence provided". Now a tool_use reply is executed and
            # fed back, bounded, until the model answers or the budget runs out.
            observations: List[Dict[str, Any]] = []
            conversation_history: List[Dict[str, Any]] = []
            self._agentic_execution = True
            # An evidence-complete resident lane must enter the closed-turn budget
            # path on its FIRST call. HCLI_NO_TOOLS used to suppress the catalog
            # text but left the reducer flag false, so the initial call could
            # consume the whole worker packet before the reducer was allowed to
            # protect the source snapshot.
            tools_closed = os.environ.get("HCLI_NO_TOOLS") == "1"
            # Built ONCE, outside the loop: observations travel as conversation
            # history rather than as a rebuilt prompt body, so every stable byte
            # stays prefix-reusable between rounds.
            cognition_prompt = self._prompt_with_observations(
                prompt,
                [],
                compact_catalog=True,
                tools_allowed=not tools_closed,
            )
            for round_index in range(self.MAX_TOOL_ROUNDS):
                # Observations go LAST in the payload, not into the prompt
                # body: they are the only part that grows between rounds, and
                # every stable byte after them is re-prefilled for nothing.
                self._tools_closed_for_round = tools_closed
                try:
                    raw = self._call_model(
                        cognition_prompt,
                        evidence,
                        compiled,
                        context_memory=context_memory,
                        history=conversation_history,
                    )
                finally:
                    self._tools_closed_for_round = False
                result = self._sanitize_result(raw)
                if tools_closed:
                    # There is no executable retrieval path in this lane. A
                    # tool_use reply is therefore an answer failure, not a
                    # reason to reopen the catalog or spend another round.
                    if result.get("kind") == "tool_use":
                        result["kind"] = "answer"
                    break
                if result.get("kind") != "tool_use":
                    break
                calls = result.get("tool_calls") or []
                if not calls:
                    # Asked to look and named nothing. Treat as an answer rather
                    # than spinning: an empty tool_use is not progress.
                    result["kind"] = "answer"
                    break
                if self._cancelled:
                    return self._cancel_result(goal_id, evidence)
                assistant_text = getattr(self, "_last_model_text", None)
                if isinstance(assistant_text, str) and assistant_text:
                    conversation_history.append(
                        {"role": "assistant", "content": assistant_text}
                    )
                round_observations = self._run_tool_calls(calls, goal_id)
                observations.extend(round_observations)
                conversation_history.append(
                    {
                        "role": "user",
                        "content": self._observations_block(observations),
                    }
                )
                # A round containing a failed call has already identified a
                # path/tool mismatch; continuing to decode beside one lucky
                # success only replays the same confused plan. Close the catalog
                # before the next model call so a local resident cannot spend
                # several minutes on a mixed success/failure loop. The post-loop
                # block below still gives it one final no-tools call, so this
                # break does not discard its remaining chance to act.
                if round_observations and (
                    any(not item.get("ok") for item in round_observations)
                    or all(
                        item.get("repeat") or not item.get("ok")
                        for item in round_observations
                    )
                    or len(observations) >= 1
                ):
                    self._emit(
                        "tool_loop_closed",
                        {
                            "goal_id": goal_id,
                            "reason": (
                                "failed_call"
                                if any(not item.get("ok") for item in round_observations)
                                else (
                                    "bounded_observation_round"
                                    if len(observations) >= 1
                                    else "repeated_or_failed"
                                )
                            ),
                            "observation_count": len(round_observations),
                            "successful": sum(
                                bool(item.get("ok")) for item in round_observations
                            ),
                        },
                    )
                    tools_closed = True
                    break
            else:
                # Budget exhausted still asking to look. Answer from what was
                # actually observed rather than reporting nothing. The prompt is
                # the SAME stable cognition_prompt, so the closing call still
                # reuses the prefix; only the history tail differs.
                self._tools_closed_for_round = True
                try:
                    result = self._sanitize_result(
                        self._call_model(
                            cognition_prompt,
                            evidence,
                            compiled,
                            context_memory=context_memory,
                            history=conversation_history
                            + [
                                {
                                    "role": "user",
                                    "content": self._observations_block(
                                        self._compact_closed_observations(
                                            observations
                                        ),
                                        final=True,
                                    ),
                                }
                            ],
                        )
                    )
                finally:
                    self._tools_closed_for_round = False
                if result.get("kind") == "tool_use":
                    result["kind"] = "answer"

            if tools_closed and result.get("kind") == "tool_use":
                # The loop breaker above intentionally does not discard the
                # model's one remaining chance to act. Remove the catalog and
                # make the no-tools instruction explicit so this call cannot
                # start another expensive duplicate round.
                self._tools_closed_for_round = True
                try:
                    final_prompt = self._prompt_with_observations(
                        prompt,
                        [],
                        compact_catalog=False,
                        tools_allowed=False,
                    )
                    self._emit(
                        "final_turn_prepared",
                        {
                            "goal_id": goal_id,
                            "prompt_chars": len(final_prompt),
                            "observations_replayed": len(observations),
                            "tools_allowed": False,
                        },
                    )
                    final_trailing = self._observations_block(
                        self._compact_closed_observations(observations),
                        final=True,
                    )
                    result = self._sanitize_result(
                        self._call_model(
                            final_prompt,
                            evidence,
                            compiled,
                            # Preserve the bounded observations that caused
                            # closure; the fit ladder sheds older blocks when
                            # the mutation contract needs more room.
                            trailing=final_trailing,
                            context_memory=context_memory,
                        )
                    )
                finally:
                    self._tools_closed_for_round = False
                if result.get("kind") == "tool_use":
                    result["kind"] = "answer"

            self._agentic_execution = False

            if self._cancelled:
                return self._cancel_result(goal_id, evidence)

            kind = result.get("kind")

            if kind == "answer":
                result["status"] = "completed"
                result["goal_id"] = goal_id

                receipt = self._write_receipt(
                    goal_id=goal_id,
                    goal=prompt,
                    result=result,
                    evidence=evidence,
                    validation={
                        "ok": True,
                        "kind": "read_only",
                        "evidence": "none",
                        "accepted_work": False,
                    },
                    rolled_back=False,
                    started=started,
                )

                result["receipt"] = receipt

                self._emit(
                    "goal_completed",
                    {
                        "goal_id": goal_id,
                        "status": "completed",
                    },
                )

                self._emit(
                    "final_response",
                    {
                        "goal_id": goal_id,
                        "content": result.get(
                            "content",
                            "",
                        ),
                        "status": "completed",
                    },
                )

                self.last_result = result
                return result

            if kind != "mutation":
                raise EngineError(
                    f"Unsupported model result kind: {kind!r}"
                )

            operations = result.get(
                "operations",
                [],
            )

            if not isinstance(operations, list):
                raise EngineError(
                    "operations must be a list"
                )

            if not 1 <= len(operations) <= self.MAX_OPERATIONS:
                raise EngineError(
                    "mutation operation count must be "
                    f"1..{self.MAX_OPERATIONS}"
                )

            tests = result.get(
                "tests",
                [],
            )

            if not isinstance(tests, list):
                tests = []

            self._emit(
                "mutation_prepared",
                {
                    "goal_id": goal_id,
                    "operations": len(operations),
                },
            )

            paths = self._operation_paths(
                operations
            )

            snapshot = self._snapshot(
                paths
            )

            rolled_back = False
            validation: Optional[Dict[str, Any]] = None
            pre_validation: Optional[Dict[str, Any]] = None
            apply_result: Optional[Dict[str, Any]] = None

            try:
                apply_result = self._apply_operations(
                    operations
                )

                self._emit(
                    "mutation_applied",
                    {
                        "goal_id": goal_id,
                        "paths": [
                            str(p.relative_to(self.root))
                            for p in paths
                        ],
                    },
                )

                if self._cancelled:
                    raise EngineError(
                        "Goal cancelled before validation"
                    )

                self._emit(
                    "validation_started",
                    {
                        "goal_id": goal_id,
                    },
                )

                if tests:
                    try:
                        pre_validation = (
                            self._run_proving_tests_against_pre_mutation_producers(
                                snapshot,
                                paths,
                                tests,
                            )
                        )
                    except Exception as exc:
                        pre_validation = {
                            "ok": False,
                            "reason": (
                                "pre_mutation_exception:"
                                f"{type(exc).__name__}"
                            ),
                            "error": str(exc),
                            "checks": [],
                        }

                validation = self._validate(
                    paths,
                    tests,
                    pre_mutation=pre_validation,
                )

                if apply_result and apply_result.get("files"):
                    validation["files"] = apply_result["files"]

                ok = (
                    bool(validation)
                    if isinstance(validation, bool)
                    else bool(
                        validation.get("ok")
                        if isinstance(
                            validation,
                            dict,
                        )
                        else False
                    )
                )

                if not ok:
                    reason = ""
                    if isinstance(validation, dict):
                        reason = str(
                            validation.get("reason") or ""
                        )
                    # Empty tests / no passing tests: record NO_EVIDENCE
                    # but keep the mutation. A refused or failing test
                    # still rolls back.
                    if reason != "NO_EVIDENCE":
                        # The reason was extracted three lines up and then
                        # thrown away, so every failure here reached the
                        # mission log as the bare string "Deterministic
                        # validation failed". That opacity blocked three
                        # separate obligations: a unit could not be diagnosed
                        # without re-running it under a debugger. Carry the
                        # reason, and whatever structured detail the validator
                        # actually produced, into the message.
                        raise EngineError(
                            validation_failure_message(validation)
                        )

                status = self._status_from_validation(validation)
                event_name = (
                    "validation_passed"
                    if status == "completed"
                    else "validation_recorded"
                )
                self._emit(
                    event_name,
                    {
                        "goal_id": goal_id,
                        "validation": validation,
                    },
                )

            except BaseException as exc:
                self._restore(
                    snapshot
                )

                rolled_back = True

                self._emit(
                    "rollback",
                    {
                        "goal_id": goal_id,
                        "reason": str(exc),
                    },
                )

                if validation is None and (
                    isinstance(exc, NoOpMutation)
                    or "NO_OP_MUTATION" in str(exc)
                ):
                    validation = {
                        "ok": False,
                        "reason": "NO_OP_MUTATION",
                        "files": getattr(exc, "files", []) or [],
                    }

                if validation is not None:
                    self._emit(
                        "validation_failed",
                        {
                            "goal_id": goal_id,
                            "validation": validation,
                        },
                    )

                result["rolled_back"] = True
                result.update(_error_fields(exc))
                result["status"] = (
                    "cancelled"
                    if self._cancelled
                    else "failed"
                )
                result["goal_id"] = goal_id

                receipt = self._write_receipt(
                    goal_id=goal_id,
                    goal=prompt,
                    result=result,
                    evidence=evidence,
                    validation=validation,
                    rolled_back=True,
                    started=started,
                )

                result["receipt"] = receipt

                self._emit(
                    "goal_completed",
                    {
                        "goal_id": goal_id,
                        "status": result["status"],
                    },
                )

                self._emit(
                    "final_response",
                    {
                        "goal_id": goal_id,
                        "content": (
                            result.get("content")
                            or result.get("error")
                            or "Goal failed"
                        ),
                        "status": result["status"],
                    },
                )

                self.last_result = result
                return result

            result["rolled_back"] = False
            result["status"] = status
            result["goal_id"] = goal_id

            receipt = self._write_receipt(
                goal_id=goal_id,
                goal=prompt,
                result=result,
                evidence=evidence,
                validation=validation,
                rolled_back=False,
                started=started,
            )

            result["receipt"] = receipt

            self._emit(
                "goal_completed",
                {
                    "goal_id": goal_id,
                    "status": status,
                },
            )

            self._emit(
                "final_response",
                {
                    "goal_id": goal_id,
                    "content": result.get(
                        "content",
                        "",
                    ),
                    "status": status,
                },
            )

            self.last_result = result
            return result

        except BaseException as exc:
            fields = _error_fields(exc)
            result = {
                "kind": "error",
                "content": "",
                "operations": [],
                "tests": [],
                "error": fields["error"],
                "error_type": fields["error_type"],
                "error_traceback": fields["error_traceback"],
                "goal_id": goal_id,
                "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            }

            receipt = self._write_receipt(
                goal_id=goal_id,
                goal=prompt,
                result=result,
                evidence=evidence,
                validation=None,
                rolled_back=False,
                started=started,
            )
            result["receipt"] = receipt

            self._emit(
                "error",
                {
                    "goal_id": goal_id,
                    "message": fields["error"],
                    "error_type": fields["error_type"],
                },
            )

            self._emit(
                "goal_completed",
                {
                    "goal_id": goal_id,
                    "status": result["status"],
                },
            )

            self.last_result = result
            return result

        finally:
            self._active_goal_id = None
            self._cancelled = False

    # -----------------------------------------------------------------
    # Goal / evidence
    # -----------------------------------------------------------------

    def _compiled_workunit_count(
        self,
        compiled: Any,
    ) -> int:
        if not isinstance(compiled, dict):
            return 0

        dag = compiled.get("workunits")

        units = getattr(
            dag,
            "units",
            None,
        )

        if isinstance(units, dict):
            return len(units)

        return 0

    def _extract_path_tokens(
        self,
        text: str,
    ) -> List[str]:
        candidates: List[str] = []

        for match in _PATH_TOKEN_RE.finditer(
            text or ""
        ):
            value = (
                match.group("qpath")
                or match.group("path")
                or ""
            ).strip()

            if not value:
                continue

            if value.startswith(
                (
                    "http://",
                    "https://",
                )
            ):
                continue

            candidates.append(value)

        result: List[str] = []
        seen = set()

        for value in candidates:
            value = value.rstrip(
                ".,:;)]}"
            )

            if value not in seen:
                seen.add(value)
                result.append(value)

        return result

    def _is_instruction_document(
        self,
        path: Path,
        prompt: str,
    ) -> bool:
        """Return True only when nested file-reference expansion is justified.

        Ordinary source/readme/docs are evidence, not implicit manifests.

        Nested expansion is reserved for explicit task/spec/mission documents
        so a request such as "read README.md" cannot recursively ingest an
        entire repository merely because README names source files.
        """
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False

        upper_name = path.name.upper()

        instruction_markers = (
            "TASK",
            "MISSION",
            "SPEC",
            "PLAN",
            "GOAL",
            "ULTRAGOAL",
            "ACCEPTANCE",
            "FULFILL",
            "SELFHOST",
            "HANDOFF",
        )

        explicit_instruction_verb = any(
            token in (prompt or "").lower()
            for token in (
                "execute ",
                "execute it",
                "follow ",
                "implement ",
                "fulfill ",
                "according to ",
                "use the instructions",
            )
        )

        under_control_dir = (
            len(rel.parts) > 0
            and rel.parts[0] == ".hcli"
        )

        named_like_instruction = any(
            marker in upper_name
            for marker in instruction_markers
        )

        return bool(
            explicit_instruction_verb
            and (
                under_control_dir
                or named_like_instruction
            )
        )

    def _context_budget(self) -> ContextBudget:
        pool = None
        if self.runtime_provider is not None:
            try:
                pool = self.runtime_provider()
            except Exception:
                pool = None
        if pool is not None:
            existing = getattr(pool, "context_budget", None)
            if isinstance(existing, ContextBudget):
                return existing
        n_parallel = max(1, int(self.runtime_count))
        model_path = None
        repo_root = None
        if pool is not None:
            topo = getattr(pool, "topology", None)
            requested = getattr(pool, "requested_n", None)
            admitted = getattr(pool, "admitted_n", None)
            if topo == "process":
                n_parallel = 1
            elif topo == "slot":
                n_parallel = max(
                    1, int(admitted or requested or n_parallel)
                )
            model_path = getattr(pool, "model_path", None)
            repo_root = getattr(pool, "repo_root", None)
        gen = None
        explicit, _source = self.config.model_tokens()
        if explicit is not None:
            gen = int(explicit)
        return resolve_context_budget(
            model_path=model_path,
            n_parallel=n_parallel,
            repo_root=str(repo_root) if repo_root is not None else None,
            generation_reserve=gen,
        )

    def _evidence_char_budget(self) -> int:
        """Conservative prompt-evidence budget derived from the authority.

        This is intentionally approximate because HCLI must not require a
        tokenizer merely to admit deterministic evidence. The admitted
        character count is always capped to usable_input_tokens so inlined
        evidence cannot exceed what the live slot can actually send.
        """
        budget = self._context_budget()
        ratio = getattr(self, "_chars_per_token", None) or float(_CHARS_PER_TOKEN)
        cap = int(
            max(0, int(budget.usable_input_tokens))
            * _MAX_OBSERVATION_SHARE
            * ratio
        )
        cap = min(self.MAX_TOTAL_EVIDENCE_CHARS, cap)

        explicit = os.environ.get(
            "HCLI_EVIDENCE_CHAR_BUDGET"
        )

        if explicit:
            try:
                return max(0, min(int(explicit), cap))
            except ValueError:
                pass

        # A WorkUnit already carries the exact named paths and a bounded
        # packet. Keep only a small inline cache for the next decision; the
        # complete files remain disk authority and are available through the
        # worker's tools. The real O003 A/B reached 20.951 s at 400 chars and
        # the 1,200-char setting retains more specimen evidence while staying
        # near the sub-30-second prefill target.
        if getattr(self, "_worker_execution", False):
            return max(0, min(1200, cap))

        return max(0, cap)

    def _reset_evidence_efficiency(self) -> None:
        self._evidence_gather_ran = False
        self._evidence_dup_bytes = 0
        self._evidence_reread_ran = False
        self._evidence_reread_bytes = 0
        self._context_efficiency: Dict[str, Any] = {}
        # The observation cut is monotone within one goal so the resident can
        # reuse its KV prefix. It must start fresh for the next goal: carrying
        # a prior goal's cut forward can skip every shedding rung and refuse a
        # closed turn that would fit after dropping the oldest observation.
        self._observation_floor = 0

    def _read_evidence_text(self, path: Path) -> str:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    def _stamp_evidence_identity(
        self,
        item: Dict[str, Any],
        path: Path,
    ) -> None:
        st = path.stat()
        content = str(item.get("content") or "")
        item["mtime_ns"] = int(st.st_mtime_ns)
        item["size"] = int(st.st_size)
        item["sha256"] = _sha256_bytes(
            content.encode("utf-8")
        )

    def _assert_evidence_fresh(
        self,
        evidence: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Re-read any inlined item whose source file changed under us.

        Invalidation is mtime_ns + size. sha256 identifies the snapshot
        that will be sent; it is not the invalidation predicate.
        """
        items = evidence if evidence is not None else []
        if not self._evidence_reread_ran:
            self._evidence_reread_bytes = 0
            self._evidence_reread_ran = True

        for item in items:
            rel = str(item.get("path") or "")
            if not rel:
                continue
            try:
                path = self._safe_path(
                    rel,
                    allow_missing=False,
                )
            except EngineError as exc:
                raise EngineError(
                    f"evidence vanished: {rel}"
                ) from exc
            if not path.is_file():
                raise EngineError(
                    f"evidence vanished: {rel}"
                )

            st = path.stat()
            stamped_mtime = item.get("mtime_ns")
            stamped_size = item.get("size")
            stale = (
                stamped_mtime is None
                or stamped_size is None
                or int(st.st_mtime_ns) != int(stamped_mtime)
                or int(st.st_size) != int(stamped_size)
            )
            if not stale:
                continue

            try:
                text = self._read_evidence_text(path)
            except Exception as exc:
                raise EngineError(
                    f"evidence unreadable: {rel}"
                ) from exc

            self._evidence_reread_bytes += len(
                text.encode("utf-8")
            )
            item["content"] = text[
                : self.MAX_EVIDENCE_CHARS_PER_FILE
            ]
            self._stamp_evidence_identity(item, path)

        return items

    def _snapshot_context_efficiency(
        self,
        evidence: Optional[List[Dict[str, Any]]],
        *,
        root_tokens: Any = "unknown",
        worker_tokens: Any = "unknown",
        evidence_bytes_inlined: Any = "unknown",
    ) -> Dict[str, Any]:
        identity: List[Dict[str, Any]] = []
        for item in evidence or []:
            rec: Dict[str, Any] = {
                "path": item.get("path"),
            }
            for key in ("mtime_ns", "size", "sha256"):
                rec[key] = item[key] if key in item else "unknown"
            identity.append(rec)

        return {
            "root_tokens": root_tokens,
            "worker_tokens": worker_tokens,
            "evidence_bytes_inlined": evidence_bytes_inlined,
            "tool_catalog_chars": getattr(self, "_tool_catalog_chars", "unknown"),
            "tool_catalog_mode": getattr(self, "_tool_catalog_mode", "unknown"),
            "bytes_re_read_stale": (
                int(self._evidence_reread_bytes)
                if self._evidence_reread_ran
                else "unknown"
            ),
            "duplicated_bytes_avoided": (
                int(self._evidence_dup_bytes)
                if self._evidence_gather_ran
                else "unknown"
            ),
            "evidence_identity": identity,
        }

    def _gather_evidence(
        self,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Gather deterministic evidence without uncontrolled recursion."""
        self._evidence_gather_ran = True
        self._evidence_dup_bytes = 0

        queue = [
            (token, True)
            for token in self._extract_path_tokens(
                prompt
            )
        ]

        seen = set()
        evidence: List[Dict[str, Any]] = []

        total_chars = 0
        char_budget = self._evidence_char_budget()

        while (
            queue
            and len(evidence) < self.MAX_EVIDENCE_FILES
            and total_chars < char_budget
        ):
            token, explicit_from_prompt = queue.pop(0)

            try:
                path = self._safe_path(
                    token,
                    allow_missing=False,
                )
            except Exception:
                continue

            key = str(path)

            if key in seen:
                for existing in evidence:
                    existing_key = str(
                        (self.root / existing["path"]).resolve()
                    )
                    if existing_key == key:
                        self._evidence_dup_bytes += len(
                            str(
                                existing.get("content") or ""
                            ).encode("utf-8")
                        )
                        break
                continue

            seen.add(key)

            if not path.is_file():
                continue

            remaining = (
                char_budget
                - total_chars
            )

            if remaining <= 0:
                break

            per_file_limit = min(
                self.MAX_EVIDENCE_CHARS_PER_FILE,
                remaining,
            )

            try:
                content = self._read_evidence_text(path)
            except Exception:
                continue

            content = _focused_excerpt(
                content,
                getattr(self, "_active_goal_text", None) or prompt,
                per_file_limit,
                str(path),
            )

            rel = str(
                path.relative_to(
                    self.root
                )
            )

            item = {
                "path": rel,
                "content": content,
            }
            self._stamp_evidence_identity(item, path)
            evidence.append(item)

            total_chars += len(content)

            # Only explicit task/spec/mission documents may expand nested
            # references. README.md, source code, documentation, etc. are
            # terminal evidence by default.
            if (
                explicit_from_prompt
                and self._is_instruction_document(
                    path,
                    prompt,
                )
            ):
                for nested in self._extract_path_tokens(
                    content
                ):
                    queue.append(
                        (
                            nested,
                            False,
                        )
                    )

        return evidence

    # -----------------------------------------------------------------
    # Local model
    # -----------------------------------------------------------------

    def _runtime_endpoint(
        self,
    ) -> Tuple[str, Dict[str, Any]]:
        if self.runtime_provider is None:
            raise EngineError(
                "No RuntimePool provider is configured"
            )

        pool = self.runtime_provider()

        runtimes = list(
            getattr(
                pool,
                "runtimes",
                [],
            )
        )

        if not runtimes:
            raise EngineError(
                "RuntimePool has no admitted runtimes"
            )

        runtime = next(
            (
                candidate
                for candidate in runtimes
                if getattr(candidate, "active", True)
            ),
            runtimes[0],
        )

        backend = getattr(runtime, "backend", None)

        port = getattr(
            runtime,
            "port",
            None,
        )

        # HTTP servers expose a port, but a native resident (or another
        # in-process/IPC backend) does not.  The Engine only needs a stable
        # provenance label here; actual inference is always delegated to the
        # RuntimePool below.  Keep the endpoint generic so adding a backend
        # does not require teaching AgentOS its transport.
        backend_endpoint = None
        endpoint_fn = getattr(backend, "endpoint", None)
        if callable(endpoint_fn):
            try:
                backend_endpoint = endpoint_fn()
            except Exception:
                backend_endpoint = None
        if port is None and not backend_endpoint:
            backend_endpoint = "RuntimePool.complete"

        provenance = {
            "index": getattr(
                runtime,
                "index",
                0,
            ),
            "pid": getattr(
                runtime,
                "pid",
                None,
            ),
            "port": port,
            "backend": getattr(backend, "__class__", type(None)).__name__
            if backend is not None
            else None,
            "endpoint": backend_endpoint,
        }

        endpoint = (
            str(backend_endpoint)
            if backend_endpoint and not str(backend_endpoint).startswith(("http://", "https://"))
            else f"http://127.0.0.1:{port}/v1/chat/completions"
            if port is not None
            else str(backend_endpoint)
        )
        provenance["endpoint"] = endpoint
        return endpoint, provenance

    def _resolve_pool(self) -> Any:
        if self.runtime_provider is None:
            return None
        try:
            return self.runtime_provider()
        except Exception:
            return None

    def _prefix_key(self, payload: Dict[str, Any]) -> str:
        """Stable routing key so sequential similar prompts share a runtime.

        System text is shared across HCLI calls; the leading user bytes keep
        distinct work units from pinning every decode onto runtime 0 once
        overlap exists, while repairs of the same unit still affinity-hit.
        """
        messages = payload.get("messages") or []
        system = ""
        user = ""
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = str(message.get("content") or "")
                if role == "system" and not system:
                    system = content
                elif role == "user" and not user:
                    user = content
                if system and user:
                    break
        blob = (system + "\n" + user[:256]).encode("utf-8", "replace")
        return hashlib.sha256(blob).hexdigest()[:24]

    def _snapshot_pool_provenance(self, pool: Any) -> Dict[str, Any]:
        for runtime in getattr(pool, "runtimes", []) or []:
            if not getattr(runtime, "active", True):
                continue
            backend = getattr(runtime, "backend", None)
            endpoint = None
            endpoint_fn = getattr(backend, "endpoint", None)
            if callable(endpoint_fn):
                try:
                    endpoint = endpoint_fn()
                except Exception:
                    endpoint = None
            port = getattr(runtime, "port", None)
            if endpoint is None and port is not None:
                endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
            if endpoint is None:
                endpoint = "RuntimePool.complete"
            return {
                "index": getattr(runtime, "index", 0),
                "pid": getattr(runtime, "pid", None),
                "port": port,
                "backend": type(backend).__name__ if backend is not None else None,
                "endpoint": endpoint,
            }
        return {
            "index": None,
            "pid": None,
            "port": None,
            "backend": None,
            "endpoint": "RuntimePool.complete",
        }

    def _provenance_for_index(
        self,
        pool: Any,
        runtime_index: Any,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        provenance = dict(fallback)
        if runtime_index is None:
            return provenance
        provenance["index"] = runtime_index
        for runtime in getattr(pool, "runtimes", []) or []:
            if getattr(runtime, "index", None) == runtime_index:
                backend = getattr(runtime, "backend", None)
                provenance["pid"] = getattr(runtime, "pid", None)
                port = getattr(runtime, "port", None)
                provenance["port"] = port
                provenance["backend"] = (
                    type(backend).__name__ if backend is not None else None
                )
                endpoint_fn = getattr(backend, "endpoint", None)
                endpoint = None
                if callable(endpoint_fn):
                    try:
                        endpoint = endpoint_fn()
                    except Exception:
                        endpoint = None
                if endpoint is None and port is not None:
                    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
                provenance["endpoint"] = endpoint or "RuntimePool.complete"
                break
        return provenance

    def _endpoint_from_provenance(self, provenance: Dict[str, Any]) -> str:
        endpoint = provenance.get("endpoint")
        if endpoint and not str(endpoint).startswith(("http://", "https://")):
            return str(endpoint)
        port = provenance.get("port")
        if port is None:
            return str(provenance.get("endpoint") or "RuntimePool.complete")
        return f"http://127.0.0.1:{port}/v1/chat/completions"

    def _cached_tokens_from(self, data: Any) -> Optional[int]:
        if not isinstance(data, dict):
            return None
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage.get("prompt_tokens_details"), dict)
            else {}
        )
        for blob, key in (
            (details, "cached_tokens"),
            (data.get("timings") if isinstance(data.get("timings"), dict) else {}, "cache_n"),
            (data, "tokens_cached"),
            (usage, "cache_n"),
        ):
            if key in blob and blob[key] is not None:
                try:
                    return int(blob[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _enter_model_call(self) -> None:
        persist: Optional[int] = None
        with self._model_inflight_lock:
            self._model_inflight += 1
            if self._model_inflight > self.max_model_in_flight:
                self.max_model_in_flight = self._model_inflight
                persist = self.max_model_in_flight
        if persist is not None:
            store_observed_overlap(self.root, persist)

    def _leave_model_call(self) -> None:
        with self._model_inflight_lock:
            self._model_inflight = max(0, self._model_inflight - 1)

    def _estimate_prompt_tokens(
        self,
        messages: List[Dict[str, Any]],
    ) -> int:
        text = "\n".join(str(m.get("content") or "") for m in messages)
        tokenizer = _exact_tokenizer()
        if tokenizer is not None:
            try:
                self._last_estimate_exact = True
                return max(1, len(tokenizer.encode(text).ids))
            except Exception:
                pass
        self._last_estimate_exact = False
        ratio = getattr(self, "_chars_per_token", None) or float(_CHARS_PER_TOKEN)
        return max(1, int(len(text) / ratio))

    def _calibrate_chars_per_token(self, real_prompt_tokens: int) -> None:
        """Learn the ratio from a count the runtime actually measured.

        Keeps the smallest ratio seen, which yields the largest token estimate
        and so the most conservative completion budget.
        """
        rendered = getattr(self, "_last_rendered_prompt", "") or ""
        if not rendered or not isinstance(real_prompt_tokens, int):
            return
        if real_prompt_tokens <= 0:
            return
        observed = len(rendered) / float(real_prompt_tokens)
        observed = max(_CHARS_PER_TOKEN_FLOOR, min(_CHARS_PER_TOKEN_CEILING, observed))
        current = getattr(self, "_chars_per_token", None) or float(_CHARS_PER_TOKEN)
        self._chars_per_token = min(current, observed)

    def _commit_posted_prompt_estimate(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Overwrite the assembly snapshot with an estimate of `payload`.

        `_build_model_payload` snapshots the messages it assembled.
        A hook installed at executors.py:359 then assigns
        `messages[1]["content"] = user[len("GOAL:\\n"):]` on that same
        dict before `_post_completion`. context_efficiency must describe
        the posted bytes, not the pre-hook snapshot.
        """
        # The exact bytes that go on the wire, after the executors hook has
        # rewritten messages[1]. Comparing anything earlier measures a prompt
        # that was never sent.
        try:
            self._last_rendered_prompt = "\n".join(
                str((m or {}).get("content") or "")
                for m in (payload.get("messages") or [])
                if isinstance(m, dict)
            )
        except Exception:
            self._last_rendered_prompt = ""
        observed = int(
            self._estimate_prompt_tokens(
                payload.get("messages") or []
            )
        )
        # Re-derive the completion budget against the POSTED prompt. It was
        # resolved before `contract.apply` injected the schema instruction, so
        # the payload grew by ~713 tokens after the budget was set and the sum
        # overflowed max_seq_len. The prompt is known exactly here; the budget
        # must follow it rather than a stale estimate.
        if payload.get("max_tokens") is not None:
            refreshed, source = self._resolve_max_tokens(observed)
            if int(refreshed) < int(payload.get("max_tokens") or 0):
                payload["max_tokens"] = int(refreshed)
                self._last_call_plan = {
                    **(self._last_call_plan or {}),
                    "max_tokens": int(refreshed),
                    "max_tokens_source": source,
                }
        plan = dict(self._last_call_plan or {})
        plan["prompt_tokens_est"] = observed
        ce = dict(self._context_efficiency or {})
        if ce:
            ce["root_tokens"] = observed
            ce["worker_tokens"] = observed
            plan["context_efficiency"] = ce
            self._context_efficiency = ce
        self._last_call_plan = plan
        return plan

    def _resolve_enable_thinking(
        self,
        override: Optional[bool],
    ) -> bool:
        if override is not None:
            return bool(override)
        return self.config.enable_thinking(default=False)

    def _resolve_response_schema(
        self,
        override: Optional[bool],
    ) -> bool:
        if override is not None:
            return bool(override)
        return self.config.response_schema_on(default=True)

    def _active_backend(self) -> Any:
        pool = self._resolve_pool()
        if pool is None:
            return None
        for runtime in getattr(pool, "runtimes", []) or []:
            backend = getattr(runtime, "backend", None)
            if backend is not None:
                return backend
        return getattr(pool, "backend", None)

    def _schema_contract(
        self,
        backend: Any,
        schema: Optional[Dict[str, Any]] = None,
    ) -> StructuredOutputContract:
        active_schema = schema or HCLI_RESULT_SCHEMA
        contract = make_structured_output_contract(
            backend,
            active_schema,
        )
        if contract is not None:
            return contract
        return StructuredOutputContract(
            schema=active_schema,
            instruction=schema_instruction(active_schema),
            max_attempts=structured_output_attempts(),
            degraded_features=["response_format"],
        )

    def _message_from_completion(
        self,
        result: CompletionResult,
    ) -> Dict[str, Any]:
        data = result.raw
        if not isinstance(data, dict):
            return {}
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {}
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        return message if isinstance(message, dict) else {}

    @staticmethod
    def _plain_text_from_value(value: Any) -> Optional[str]:
        """Extract the provider text without guessing at a structured result."""
        if isinstance(value, str):
            return value
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
        raw = getattr(value, "raw", value)
        if isinstance(raw, dict):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0] if isinstance(choices[0], dict) else {}
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
            if isinstance(raw.get("content"), str):
                return raw["content"]
        return None

    def _require_plain_text(self, value: Any, *, endpoint: str) -> str:
        """Return non-empty provider text or preserve an explicit failure."""
        text = self._plain_text_from_value(value)
        if isinstance(text, str) and text.strip():
            return text
        finish_reason = getattr(value, "finish_reason", None)
        completion_tokens = getattr(value, "completion_tokens", None)
        prompt_tokens = getattr(value, "prompt_tokens", None)
        raise EngineError(
            "provider returned empty text "
            f"(finish_reason={finish_reason!r}, "
            f"completion_tokens={completion_tokens!r}, "
            f"prompt_tokens={prompt_tokens!r}, endpoint={endpoint!r})"
        )

    def _invoke_completion(
        self,
        payload: Dict[str, Any],
        timeout: float,
        *,
        endpoint: str,
        provenance: Dict[str, Any],
        plan: Dict[str, Any],
        prefix_key: str,
        use_pool: bool,
        pool: Any,
    ) -> Tuple[CompletionResult, Dict[str, Any], str]:
        """One HTTP/pool completion. Records the call. Never retries."""
        started = time.perf_counter()
        data: Optional[Dict[str, Any]] = None
        result_obj: Optional[CompletionResult] = None
        runtime_index: Any = provenance.get("index")
        try:
            self._enter_model_call()
            try:
                if use_pool:
                    result_obj = pool.complete(
                        payload,
                        timeout=timeout,
                        prefix_key=prefix_key,
                    )
                    raw = getattr(result_obj, "raw", result_obj)
                    if not isinstance(raw, dict):
                        raise EngineError(
                            "llama-server returned invalid JSON"
                        )
                    data = raw
                    runtime_index = getattr(
                        result_obj, "runtime_index", runtime_index
                    )
                    provenance = self._provenance_for_index(
                        pool, runtime_index, provenance
                    )
                    provenance["via"] = "RuntimePool.complete"
                    endpoint = self._endpoint_from_provenance(provenance)
                    if not isinstance(result_obj, CompletionResult):
                        result_obj = completion_from_openai(data, [])
                    elif result_obj.raw is not data:
                        result_obj.raw = data
                else:
                    data = self._post_completion(
                        endpoint,
                        payload,
                        timeout,
                    )
                    result_obj = completion_from_openai(data, [])
            finally:
                self._leave_model_call()
        except urllib.error.HTTPError as exc:
            wall = round(time.perf_counter() - started, 3)
            self._record_model_call(
                endpoint=endpoint,
                finish_reason=None,
                prompt_tokens=plan.get("prompt_tokens_est"),
                completion_tokens=None,
                wall_s=wall,
                max_tokens=plan.get("max_tokens"),
                max_tokens_source=plan.get("max_tokens_source"),
                runtime_index=runtime_index,
                prefix_key=prefix_key,
            )
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise EngineError(
                f"llama-server HTTP {exc.code}: "
                f"{detail[:1200]}"
            ) from exc
        except EngineError:
            wall = round(time.perf_counter() - started, 3)
            self._record_model_call(
                endpoint=endpoint,
                finish_reason=None,
                prompt_tokens=plan.get("prompt_tokens_est"),
                completion_tokens=None,
                wall_s=wall,
                max_tokens=plan.get("max_tokens"),
                max_tokens_source=plan.get("max_tokens_source"),
                runtime_index=runtime_index,
                prefix_key=prefix_key,
            )
            raise
        except Exception as exc:
            wall = round(time.perf_counter() - started, 3)
            self._record_model_call(
                endpoint=endpoint,
                finish_reason=None,
                prompt_tokens=plan.get("prompt_tokens_est"),
                completion_tokens=None,
                wall_s=wall,
                max_tokens=plan.get("max_tokens"),
                max_tokens_source=plan.get("max_tokens_source"),
                runtime_index=runtime_index,
                prefix_key=prefix_key,
            )
            raise EngineError(
                f"llama-server request failed: {exc}"
            ) from exc

        wall = round(time.perf_counter() - started, 3)

        try:
            choice0 = data["choices"][0]
            message = choice0["message"]
        except Exception as exc:
            self._record_model_call(
                endpoint=endpoint,
                finish_reason=None,
                prompt_tokens=plan.get("prompt_tokens_est"),
                completion_tokens=None,
                wall_s=wall,
                max_tokens=plan.get("max_tokens"),
                max_tokens_source=plan.get("max_tokens_source"),
                runtime_index=runtime_index,
                cached_tokens=self._cached_tokens_from(data),
                prefix_key=prefix_key,
            )
            raise EngineError(
                "llama-server response missing choices[0].message"
            ) from exc

        finish_reason = choice0.get("finish_reason")
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            self._calibrate_chars_per_token(prompt_tokens)
        if prompt_tokens is None:
            prompt_tokens = plan.get("prompt_tokens_est")
        completion_tokens = usage.get("completion_tokens")

        self._record_model_call(
            endpoint=endpoint,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_s=wall,
            max_tokens=plan.get("max_tokens"),
            max_tokens_source=plan.get("max_tokens_source"),
            runtime_index=runtime_index,
            cached_tokens=self._cached_tokens_from(data),
            prefix_key=prefix_key,
            native=(data.get("hawking") if isinstance(data, dict) else None),
        )

        # The builder question, asked on the artifact the builder produces.
        # Same conversation only: comparing prompts across goals measures
        # nothing, because nothing was supposed to be shared.
        try:
            native = data.get("hawking") if isinstance(data, dict) else None
            # The budget the connector GRANTED, which is what the resident was
            # actually allowed. `plan["max_tokens"]` is the pre-reduction value
            # and diverges from it: receipt 5b2e060a recorded plan 2048 against a
            # payload of 1446, and the retry instruction built from the plan told
            # the model it had 2048 and "the runtime stopped 710 tokens SHORT",
            # when the runtime had delivered its budget exactly.
            granted = (native or {}).get("max_new_tokens_granted")
            if isinstance(granted, int) and granted > 0:
                self._last_granted_max_tokens = granted
            self._prefix_probe.observe(
                self._active_goal_id or "",
                self._last_rendered_prompt or "",
                prompt_tokens=prompt_tokens,
                prefix_reused_tokens=(native or {}).get("prefix_reused_tokens"),
                prefill_tokens_stepped=(native or {}).get("prefill_tokens_stepped"),
                active_context_tokens=plan.get("prompt_tokens_est"),
            )
        except Exception:
            pass

        if result_obj is None:
            result_obj = completion_from_openai(data, [])
        if result_obj.text is None and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                result_obj.text = content
        if result_obj.finish_reason is None:
            result_obj.finish_reason = finish_reason
        if result_obj.prompt_tokens is None:
            result_obj.prompt_tokens = prompt_tokens
        if result_obj.completion_tokens is None:
            result_obj.completion_tokens = completion_tokens
        return result_obj, provenance, endpoint

    def _parse_structured_reply(
        self,
        result: CompletionResult,
        plan: Dict[str, Any],
        thinking_on: bool,
    ) -> Dict[str, Any]:
        message = self._message_from_completion(result)
        content = result.text
        if content is None:
            content = message.get("content", "") if message else ""

        if not thinking_on and self._thinking_leaked(content, message):
            raise EngineError(
                "llama-server ignored chat_template_kwargs.enable_thinking=false "
                "(reply contained a <think> block or reasoning_content). "
                "Start llama-server with --jinja so the reasoning policy "
                "can take effect."
            )

        finish_reason = result.finish_reason
        prompt_tokens = result.prompt_tokens
        if prompt_tokens is None:
            prompt_tokens = plan.get("prompt_tokens_est")
        completion_tokens = result.completion_tokens

        try:
            parsed = self._extract_json_object(content)
        except EngineError:
            if str(finish_reason) == "length":
                budget = (
                    getattr(self, "_last_granted_max_tokens", None)
                    or plan.get("max_tokens")
                )
                raise EngineError(
                    _truncation_message(budget, completion_tokens, prompt_tokens)
                )
            raise EngineError(
                "Model did not return a valid structured JSON object "
                f"(finish_reason={finish_reason!r}, "
                f"completion_tokens={completion_tokens}, "
                f"prompt_tokens={prompt_tokens})"
            )
        return parsed

    def _complete_with_schema_contract(
        self,
        contract: StructuredOutputContract,
        payload: Dict[str, Any],
        timeout: float,
        *,
        endpoint: str,
        provenance: Dict[str, Any],
        plan: Dict[str, Any],
        prefix_key: str,
        use_pool: bool,
        pool: Any,
        thinking_on: bool,
    ) -> Dict[str, Any]:
        """Prompt-side schema: validate every reply, retry bounded, fail hard."""

        def complete_fn(
            to_send: Dict[str, Any],
            timeout_arg: Optional[float] = None,
        ) -> CompletionResult:
            result, _, _ = self._invoke_completion(
                to_send,
                float(timeout if timeout_arg is None else timeout_arg),
                endpoint=endpoint,
                provenance=provenance,
                plan=plan,
                prefix_key=prefix_key,
                use_pool=use_pool,
                pool=pool,
            )
            message = self._message_from_completion(result)
            content = result.text
            if content is None:
                content = message.get("content", "") if message else ""
            if not thinking_on and self._thinking_leaked(content, message):
                raise EngineError(
                    "llama-server ignored chat_template_kwargs.enable_thinking=false "
                    "(reply contained a <think> block or reasoning_content). "
                    "Start llama-server with --jinja so the reasoning policy "
                    "can take effect."
                )
            if str(result.finish_reason) == "length":
                try:
                    contract.validate(content)
                except Exception:
                    # A PATCH BLOCK is complete even when the JSON around it is
                    # not. This check has to sit here rather than in the
                    # extractor: the truncation violation is raised before any
                    # extraction runs, so a block-form reply was rejected three
                    # times over without the parser for it ever being reached.
                    if _patch_block_to_operations(content) is not None:
                        return result
                    # The granted budget, not the plan's pre-reduction one.
                    # Quoting the plan made a model that EXHAUSTED its budget be
                    # told it had stopped short of a larger one and should
                    # "answer far more briefly" -- advice against a ceiling it
                    # had already hit, spent three attempts over.
                    budget = (
                        getattr(self, "_last_granted_max_tokens", None)
                        or plan.get("max_tokens")
                    )
                    prompt_tokens = result.prompt_tokens
                    if prompt_tokens is None:
                        prompt_tokens = plan.get("prompt_tokens_est")
                    # A reply cut off mid-object is a SCHEMA VIOLATION the
                    # contract can retry, not an engine fault. Raising
                    # EngineError here escaped contract.enforce, so the
                    # attempt was never counted and the live receipt read
                    # attempts=0 against max_attempts=3 -- a retry budget
                    # that was never spent. The reason doubles as the retry
                    # instruction: enforce() appends it to the next payload.
                    raise SchemaViolation(
                        _truncation_message(
                            budget, result.completion_tokens, prompt_tokens
                        )
                        + "; answer far more briefly -- shortest valid values, "
                        "no prose, no markdown fence, no repeated fields",
                        text=content,
                    )
            if "response_format" in to_send:
                raise EngineError(
                    "degraded structured-output path sent response_format"
                )
            syntax = _python_syntax_violation(content)
            if syntax is not None:
                # A dropped bracket is a SCHEMA VIOLATION the contract can
                # retry, not a terminal unit failure. Measured behaviour: a
                # 2-line source patch was correct while a 15-line test rewrite
                # in the same reply dropped three closing parens, py_compile
                # failed AFTER the mutation was applied, and the whole unit --
                # including the correct half -- was rolled back and lost.
                #
                # The model never saw the error. Now it does, with the line and
                # the caret, on the next attempt, which is the one thing that
                # makes the failure fixable by the thing that caused it.
                raise SchemaViolation(syntax, text=content)
            return result

        _boundary_trace(
            "structured_parser_begin",
            attempts=getattr(contract, "max_attempts", None),
            prompt_tokens=plan.get("prompt_tokens_est"),
            goal_id=self._active_goal_id,
            prompt_sha256=hashlib.sha256(
                (self._last_rendered_prompt or "").encode("utf-8")
            ).hexdigest()[:16],
        )
        try:
            result = contract.enforce(complete_fn, payload, timeout)
        except StructuredOutputExhausted as exc:
            last_violation = (
                exc.errors[-1] if exc.errors else (exc.reason or "no valid JSON")
            )
            plan["structured_output"] = _degraded_structured_record(
                contract,
                attempts=int(exc.attempts),
                exhausted=True,
                last_violation=last_violation,
                errors=list(exc.errors),
                last_text=exc.last_text,
            )
            self._last_call_plan = plan
            _boundary_trace(
                "structured_parser_end",
                outcome="exhausted",
                attempts=int(exc.attempts),
                error=str(exc),
            )
            raise

        # Preserve the exact provider text for the next tool round. The
        # resident already has this assistant turn in its recurrent/KV state;
        # replaying it as a real assistant message is what makes the following
        # user observation a pure append and unlocks prefix reuse.
        self._last_model_text = result.text

        attempts = int(
            getattr(result, "schema_attempts", None)
            or 1
        )
        plan["structured_output"] = _degraded_structured_record(
            contract,
            attempts=attempts,
            exhausted=False,
        )
        self._last_call_plan = plan

        parsed = None
        if isinstance(result.raw, dict):
            parsed = result.raw.get("_structured")
        if not isinstance(parsed, dict):
            parsed = contract.validate(result.text)
        _boundary_trace(
            "structured_parser_end",
            outcome="success",
            attempts=attempts,
            result_kind=parsed.get("kind") if isinstance(parsed, dict) else None,
        )
        return parsed

    def _resolve_max_tokens(
        self,
        prompt_tokens_est: int,
    ) -> Tuple[int, str]:
        explicit, source = self.config.model_tokens()
        if explicit is not None and source is not None:
            return max(1, int(explicit)), source
        ctx = self._context_budget().per_request_ctx
        # The margin must SCALE. `_estimate_prompt_tokens` divides characters by
        # a constant; the real tokenizer disagreed by 5.8% on live prompts --
        # estimated 5,488 where the resident counted 5,804 -- and a flat 96
        # tokens cannot cover an error proportional to length. Measured
        # overflow: 5,804 + 2,557 against a 8,192 window.
        # An exact count needs only the chat template's overhead covered; an
        # estimated one needs the whole measured disagreement.
        error = (
            _CTX_EXACT_MARGIN
            if getattr(self, "_last_estimate_exact", False)
            else _CTX_ESTIMATE_ERROR
        )
        margin = max(_CTX_ESTIMATE_MARGIN, int(prompt_tokens_est * error))
        remaining = int(ctx) - int(prompt_tokens_est) - margin
        # The floor may not push the request PAST the window. `max(floor, ...)`
        # granted 512 completion tokens even when the prompt had already used
        # the context, and the runtime refused the whole call:
        #   "prompt has 5792 tokens and max_new_tokens is 2612;
        #    resident max_seq_len is 8192"
        # A request that cannot fit is not worth making, and asking for a floor
        # that does not exist turns a tight fit into a hard failure.
        derived = min(_MAX_TOKENS_CEILING, remaining)
        if derived >= _MAX_TOKENS_FLOOR:
            return derived, "derived"
        return max(1, derived), "derived_clamped_to_window"

    @staticmethod
    def _render_context_memory(memory: Any) -> str:
        """Serialize only bounded, structured memory into the next prompt."""
        if not isinstance(memory, dict) or not memory:
            return ""
        try:
            safe = json.loads(json.dumps(
                memory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ))
        except (TypeError, ValueError):
            return ""
        limit = 16000

        def shrink(value: Any, *, depth: int = 0, text_limit: int = 600) -> Any:
            if isinstance(value, str):
                return value[:text_limit] + ("…" if len(value) > text_limit else "")
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if depth >= 4:
                return "[omitted]"
            if isinstance(value, list):
                return [
                    shrink(item, depth=depth + 1, text_limit=text_limit)
                    for item in value[:12]
                ]
            if isinstance(value, dict):
                # Constraints and state survive before conversational tail or
                # raw receipt detail. This is semantic gravity, not a blind
                # character slice that can produce invalid JSON.
                order = (
                    "schema",
                    "active_goal",
                    "mission",
                    "ledger",
                    "steering",
                    "staging",
                    "prior_knowledge",
                    "goal_bank",
                    "receipts",
                    "recent",
                    "retention",
                )
                keys = [key for key in order if key in value]
                keys.extend(key for key in value if key not in keys)
                return {
                    str(key): shrink(value[key], depth=depth + 1, text_limit=text_limit)
                    for key in keys[:20]
                }
            return str(value)[:text_limit]

        text = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(text) <= limit:
            return text
        for text_limit in (600, 360, 220, 120, 72):
            candidate = shrink(safe, text_limit=text_limit)
            text = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len(text) <= limit:
                return text
        # The final fallback is intentionally small and always valid JSON.
        core = {
            key: safe.get(key)
            for key in ("schema", "active_goal", "mission", "ledger", "staging", "prior_knowledge", "goal_bank")
            if key in safe
        }
        return json.dumps(shrink(core, text_limit=48), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _prompt_with_context_memory(cls, prompt: str, memory: Any) -> str:
        """Compatibility path for legacy ``model_client.complete`` providers."""
        rendered = cls._render_context_memory(memory)
        if not rendered:
            return prompt
        return (
            f"{prompt}\n\nDURABLE CONTEXT CHECKPOINT (bounded operator memory; "
            f"verify against current disk state):\n{rendered}"
        )

    # The raw goal may be far larger than the resident's whole input budget: the
    # sovereign ultragoal that failed measured 12,456 tokens against 5,632 usable
    # on sealed-3.14. `del compiled` used to throw away the compiled goal and
    # embed the raw text verbatim, so an oversized goal could only ever be
    # refused. SOURCE IS NOT ACTIVE CONTEXT: the exact bytes are preserved on
    # disk, hashed, and referenced, and the resident receives a compact kernel
    # plus the handle. It retrieves exact wording with fs.read when it needs it.
    GOAL_BUDGET_FRACTION = 0.5

    def _persist_goal_source(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Write the exact goal bytes once, keyed by content hash. Never deletes."""
        try:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            directory = self.root / ".hcli" / "sources"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"goal-{digest[:16]}.txt"
            if not path.is_file():
                _atomic_write_text(path, prompt)
            return {
                "path": os.path.relpath(path, self.root),
                "sha256": digest,
                "chars": len(prompt),
            }
        except OSError:
            return None

    def _goal_block(self, prompt: str, compiled: Any) -> str:
        """Raw goal when it fits the budget; a compiled kernel when it does not."""
        if compiled is None:
            return f"GOAL:\n{prompt}"
        try:
            usable = int(self._context_budget().usable_input_tokens)
        except Exception:
            return f"GOAL:\n{prompt}"
        limit = max(256, int(usable * self.GOAL_BUDGET_FRACTION))
        if len(prompt) // _CHARS_PER_TOKEN <= limit:
            return f"GOAL:\n{prompt}"

        source = self._persist_goal_source(prompt)
        get = compiled.get if hasattr(compiled, "get") else (lambda k, d=None: d)
        lines = ["GOAL (compiled kernel; the full source is EXACT and retrievable):"]
        summary = str(get("goal_summary") or "").strip()
        if summary:
            lines.append(f"  objective: {summary}")
        for label, key in (
            ("invariants", "invariants"),
            ("acceptance criteria", "acceptance_criteria"),
            ("obligations", "obligations"),
            ("referenced files", "referenced_files"),
        ):
            items = [str(x).strip() for x in (get(key) or []) if str(x).strip()]
            if items:
                lines.append(f"  {label}:")
                lines.extend(f"    - {item}" for item in items[:12])
        if source is not None:
            lines.append(
                f"  SOURCE: {source['path']}  "
                f"({source['chars']} chars, sha256 {source['sha256'][:16]})"
            )
            lines.append(
                "  The compiled kernel above is a SUMMARY. When you need exact "
                f"wording, read spans with fs.read(path=\"{source['path']}\"). "
                "Do not assume detail that is not stated here."
            )
        else:
            lines.append(
                "  SOURCE: could not be persisted; this kernel is all that is available."
            )
        return "\n".join(lines)

    # A turn that does not fit must SHRINK before it is refused. Preflight used
    # to raise ContextPreflightError straight into `resources.py`, which grades
    # it IMPOSSIBLE_CONTRACT -- so one oversized turn ended the goal with no
    # attempt to recover, which is exactly what an unattended overnight run
    # cannot survive. Reduce what is re-derivable, in order of least loss:
    # deterministic evidence can be re-read from disk with fs.read, and the
    # durable checkpoint can be re-read from mission state. The GOAL is never
    # reduced here -- `_goal_block` already compiled it and the exact source is
    # on disk. Nothing is truncated mid-token; whole items are dropped.
    EVIDENCE_REDUCTION_STEPS = (1.0, 0.5, 0.25, 0.0)

    def _fit_payload_to_budget(
        self,
        build: Callable[..., Dict[str, Any]],
        evidence: Any,
        context_memory: Any,
        trailing: str = "",
        reserve: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Build the payload, shrinking re-derivable context until it fits.

        `trailing` -- the accumulated tool observations -- is re-derivable too,
        and it was the one growing block the ladder could not touch. Twelve
        model calls succeeded at ~2,600 prompt tokens and then the thirteenth
        was refused at 14,135, because observations accumulate across rounds and
        nothing could shed them. Evidence would be dropped to zero while the
        observations that actually caused the overflow were left untouched.

        Observations are shed NEWEST-KEEPING: the last tool result is the one
        the model asked for and is about to reason over, and the earliest are
        the ones it has already used. This runs only after evidence and the
        durable checkpoint are gone, because a file snapshot is cheaper to
        re-derive than a tool call that has already been paid for.
        """
        try:
            budget = self._context_budget()
        except Exception:
            return build(evidence, context_memory), None

        items = list(evidence or ())
        # With NO TOOLS the ladder's premise is inverted. It sheds evidence
        # first because "a file snapshot can be re-read for free" -- true only
        # while fs.read exists. Under an empty tool bundle the snapshot is the
        # ONLY source of truth, and dropping it leaves the model to invent the
        # file from memory. Measured: evidence_files listed tool_registry.py,
        # evidence_bytes_inlined was 0, and the model sent an anchor reading
        # "    if start is None and end is None:" -- a plausible line that is
        # not in the file.
        #
        # So keep the first evidence item and let the rungs below shrink the
        # rest. Tool catalogs and observations are re-derivable; the explicit
        # task evidence is the authority the model must not lose merely because
        # it asked for one extra read. Something must still give when the
        # budget is short, but it must not be the only source of truth.
        tools_closed = bool(getattr(self, "_tools_closed_for_round", False))
        keep_floor = 1 if items else 0
        # The final no-tools turn is a decision turn, not another retrieval
        # turn. It still needs the bounded result that caused closure: dropping
        # that result made the resident answer read-only from the disk snapshot
        # even though the failed path was the evidence that should have driven
        # the mutation. Keep the newest useful block if the full tail will not
        # fit; the ladder below sheds older blocks monotonically.
        blocks = _observation_blocks(trailing)
        # Where the kept observations START, as an ABSOLUTE index that only ever
        # moves forward. Keeping "the last N" instead re-cut the block at a
        # different observation every turn, so the rendered prompt changed at
        # the point observations begin and the resident's KV prefix could be
        # reused only up to there. Measured: five consecutive calls pinned at
        # 1398 reused tokens -- the system prompt, tools and goal -- while the
        # prompts grew past 4700, every later token re-stepped at 580 dispatches
        # each. A floor that only advances makes each turn the previous turn
        # plus an append, which is exactly what a prefix cache can reuse.
        floor = min(getattr(self, "_observation_floor", 0), max(len(blocks) - 1, 0))

        attempts: List[Tuple[Any, ...]] = []
        for fraction in self.EVIDENCE_REDUCTION_STEPS:
            keep = (
                items[: max(keep_floor, int(len(items) * fraction))]
                if items else []
            )
            label = (
                "full" if fraction == 1.0
                else f"evidence {len(keep)}/{len(items)}"
            )
            attempts.append((keep, context_memory, label, floor))
        # Last resorts: drop the durable checkpoint too, then shed observations
        # oldest-first. Observations go last because a tool call has already
        # been paid for, while a file snapshot can be re-read for free.
        attempts.append((items[:keep_floor], None, f"evidence {keep_floor} + no checkpoint", floor))
        if len(blocks) > 1:
            for keep_n in (len(blocks) * 3 // 4, len(blocks) // 2, len(blocks) // 4, 1):
                if keep_n < 1 or keep_n >= len(blocks):
                    continue
                advanced = len(blocks) - keep_n
                # Never rewind: re-admitting an observation already shed would
                # lengthen the prompt again AND move the cut backwards, losing
                # the prefix twice over.
                if advanced <= floor:
                    continue
                attempts.append((
                    items[:keep_floor], None,
                    f"evidence {keep_floor} + observations {keep_n}/{len(blocks)}",
                    advanced,
                ))

        # LAST RESORT: drop evidence entirely rather than refuse the call.
        # keep_floor exists so the model is not left inventing a file it cannot
        # read, but a floor that cannot fit refuses the turn outright --
        # measured: "context preflight failed (root): demand 10580 exceeds
        # per-request ctx", which is strictly worse than a degraded prompt.
        if keep_floor:
            attempts.append(([], None, "evidence 0 (floor abandoned to fit)", floor))
            if len(blocks) > 1:
                for keep_n in (len(blocks) * 3 // 4, len(blocks) // 2, len(blocks) // 4, 1):
                    if keep_n < 1 or keep_n >= len(blocks):
                        continue
                    advanced = len(blocks) - keep_n
                    if advanced <= floor:
                        continue
                    attempts.append((
                        [], None,
                        f"evidence 0 + observations {keep_n}/{len(blocks)}",
                        advanced,
                    ))

        # Real tool rounds use chat history, not the legacy ``trailing`` field.
        # Keep the immediate assistant/tool-result pair when possible, but make
        # older rounds re-derivable just like old observations.  Before this
        # ladder knew about history, an O003 tool round could pass its first
        # model call and then refuse the next one at 25K tokens because the
        # reducer had no way to shed the history it had just added.
        history_values: List[Optional[List[Dict[str, Any]]]] = [history]
        if history:
            raw_latest_pair = list(history[-2:])
            if raw_latest_pair != history:
                history_values.append(raw_latest_pair)

            # A raw latest pair can still be too large even after every older
            # turn is dropped: tool output is deliberately bounded, but the
            # base O003 packet plus the schema reserve can leave only a small
            # remainder.  Preserve a compact decision trace before abandoning
            # the observation entirely.  The tool can always be re-run from
            # the durable evidence; an identical base prompt cannot tell the
            # resident why its previous request was made.
            compacted: List[Dict[str, Any]] = []
            for item in history[-2:]:
                if not isinstance(item, dict):
                    continue
                value = dict(item)
                content = value.get("content")
                if isinstance(content, str):
                    cap = 1200 if value.get("role") == "assistant" else 2400
                    if len(content) > cap:
                        value["content"] = content[:cap] + (
                            f"\n[history content truncated from {len(content)} "
                            "characters; use the durable evidence/tool result "
                            "rather than reconstructing the omitted tail]"
                        )
                compacted.append(value)
            if compacted and compacted != raw_latest_pair:
                history_values.append(compacted)
            for keep_n in (1, 0):
                if keep_n < len(history):
                    history_values.append(list(history[-keep_n:]) if keep_n else [])

        try:
            import inspect

            accepts_history = "history" in inspect.signature(build).parameters
        except Exception:
            accepts_history = False

        def invoke_build(
            keep: Any,
            memory: Any,
            observed: str,
            hist: Optional[List[Dict[str, Any]]],
        ) -> Dict[str, Any]:
            if accepts_history:
                return build(keep, memory, observed, history=hist)
            if observed:
                return build(keep, memory, observed)
            return build(keep, memory)

        last = None
        for keep, memory, label, cut in attempts:
            # The two-argument form is the contract every existing caller uses.
            # A third argument appears only once observations are being shed, so
            # a build() that knows nothing about `trailing` keeps working.
            observed = _join_observations(blocks[cut:]) if cut else ""
            payload = invoke_build(keep, memory, observed, history_values[0])
            demand = self._estimate_prompt_tokens(payload.get("messages") or []) + reserve
            fits = preflight(budget, demand, kind="root").ok
            if tools_closed and demand > self.CLOSED_TURN_TARGET_TOKENS:
                fits = False
            if fits:
                self._observation_floor = cut
                if label == "full" and not cut:
                    return payload, None
                return payload, {
                    "reduced_to": label,
                    "prompt_tokens_est": demand,
                    "usable_input_tokens": int(budget.usable_input_tokens),
                    "dropped_evidence": len(items) - len(keep),
                    "observation_floor": cut,
                }
            last = (payload, demand)

        # If the complete chat history still cannot fit after re-derivable
        # context has been shed, retain the most recent pair, then only the
        # latest observation, before refusing.  This is deliberately after the
        # normal evidence/memory ladder: a prior assistant turn and its latest
        # tool result are more decision-relevant than a durable checkpoint.
        for history_index, hist in enumerate(history_values[1:], start=1):
            payload = invoke_build([], None, "", hist)
            demand = self._estimate_prompt_tokens(payload.get("messages") or []) + reserve
            if preflight(budget, demand, kind="root").ok:
                self._observation_floor = 0
                kept = len(hist or [])
                compacted_history = bool(
                    hist
                    and any(
                        isinstance(item, dict)
                        and isinstance(item.get("content"), str)
                        and "history content truncated from " in item["content"]
                        for item in hist
                    )
                )
                return payload, {
                    "reduced_to": (
                        f"compact history {kept}/{len(history or [])}"
                        if compacted_history
                        else f"history {kept}/{len(history or [])}"
                    ),
                    "prompt_tokens_est": demand,
                    "usable_input_tokens": int(budget.usable_input_tokens),
                    "dropped_evidence": len(items),
                    "dropped_history": len(history or []) - kept,
                    "history_compacted": compacted_history,
                    "observation_floor": 0,
                }
            last = (payload, demand)

        # Still over budget with nothing re-derivable left. Refuse honestly, and
        # say what was already given up so the refusal is actionable.
        #
        # And say what the demand is MADE OF. An offline reconstruction of a
        # failing unit's payload measured 1,714 tokens while the live refusal
        # reported 8,707 -- a gap that cannot be closed by reading the code,
        # because the only thing that knows is the payload that was actually
        # built. Per-message sizes are cheap and they end that argument.
        payload, demand = last
        try:
            breakdown = ", ".join(
                f"{(m or {}).get('role', '?')}={len(str((m or {}).get('content') or '')) // _CHARS_PER_TOKEN}"
                for m in (payload.get("messages") or [])
                if isinstance(m, dict)
            )
            self._emit("context_refused", {
                "demand": demand,
                "usable_input_tokens": int(budget.usable_input_tokens),
                "messages": breakdown,
                "evidence_items": len(items),
                "observation_blocks": len(blocks),
            })
        except Exception:
            pass
        result = preflight(budget, demand, kind="root")
        raise ContextPreflightError(result)

    def _build_model_payload(
        self,
        prompt: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        compiled: Any = None,
        *,
        context_memory: Any = None,
        enable_thinking: Optional[bool] = None,
        response_schema: Optional[bool] = None,
        trailing: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Mission worker packets are already the compiled, bounded control
        # context. Wrapping one in the legacy ``GOAL:\n`` envelope would make
        # the worker seam look like a root-goal call again and defeats the
        # packet compiler's no-parent-dump invariant. Keep ordinary/root calls
        # on the established goal-block path.
        is_worker_packet = (
            str(prompt or "").lstrip().startswith("PHASE:")
            and "\nWORKUNIT:" in str(prompt)
        )
        goal_block = prompt if is_worker_packet else self._goal_block(prompt, compiled)
        evidence = self._assert_evidence_fresh(evidence)
        evidence_text = "\n\n".join(
            (
                f"===== {item['path']} =====\n"
                f"{item['content']}"
            )
            for item in evidence
        )
        user = (
            f"{goal_block}\n\n"
            f"DETERMINISTIC EVIDENCE:\n"
            f"{evidence_text or '(none)'}"
        )
        memory_text = self._render_context_memory(context_memory)
        if memory_text:
            user += (
                "\n\nDURABLE CONTEXT CHECKPOINT (bounded operator memory; "
                "verify against current disk state):\n"
                + memory_text
            )
        # LAST, and it must stay last. `trailing` is the only part of the prompt
        # that grows between rounds of one goal, so every stable byte placed
        # after it would be re-prefilled every round for no reason. The stable
        # blocks used to sit here: mean reusable prefix 0.544 against 0.924 with
        # the growth at the end.
        if trailing and not history:
            user += "\n\n" + trailing
        system_prompt = (
            _AGENTIC_SYSTEM_PROMPT
            if getattr(self, "_agentic_execution", False)
            else _SYSTEM_PROMPT
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user,
            },
        ]
        if history:
            # Keep the original user turn (including the schema instruction)
            # as the stable prefix. The resident's previous assistant JSON and
            # tool observations are appended as real chat turns, so the next
            # rendered prompt begins with the exact resident context instead
            # of diverging at the old user-turn terminator.
            messages.extend(
                dict(item)
                for item in history
                if isinstance(item, dict)
            )
            if trailing:
                messages.append({"role": "user", "content": trailing})
        thinking = self._resolve_enable_thinking(enable_thinking)
        use_schema = self._resolve_response_schema(response_schema)
        prompt_tokens_est = self._estimate_prompt_tokens(messages)
        max_tokens, source = self._resolve_max_tokens(prompt_tokens_est)
        payload: Dict[str, Any] = {
            "model": self.model_name or "local",
            "messages": messages,
            "temperature": 0.15,
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {
                "enable_thinking": bool(thinking),
            },
        }
        if history:
            # StructuredOutputContract.apply() must inject its schema into the
            # stable base user turn, never into the mutable observation turn.
            payload["_schema_instruction_user_index"] = 1
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "hcli_result",
                    "strict": True,
                    "schema": HCLI_RESULT_SCHEMA,
                },
            }
        inlined = 0
        for item in evidence:
            inlined += len(
                str(item.get("content") or "").encode("utf-8")
            )
        # One-shot instrument: write the evidence actually POSTED. Byte counts
        # cannot answer "which 6027 characters", and inferring the window from
        # the code produced two wrong conclusions in a row.
        if os.environ.get("HCLI_DUMP_EVIDENCE"):
            try:
                with open(os.environ["HCLI_DUMP_EVIDENCE"], "a") as handle:
                    # APPEND, with the objective. Overwriting made every read
                    # ambiguous: the file held whichever call ran last, so a
                    # correct first call and a wrong later one were
                    # indistinguishable.
                    handle.write("\n########## CALL ##########\n")
                    obj = [ln for ln in str(prompt or "").splitlines()
                           if ln.lstrip().startswith("OBJECTIVE:")]
                    handle.write(f"PROMPT_CHARS: {len(str(prompt or ''))}\n")
                    handle.write(f"OBJECTIVE_LINE: {obj[0][:200] if obj else '(NONE FOUND)'}\n")
                    for item in evidence:
                        head = str(item.get("content") or "").splitlines()[:1]
                        handle.write(f"===== {item.get('path')} :: {head}\n")
            except Exception:  # an instrument must never end a goal
                pass
        # Snapshot of this assembled payload. _call_model re-measures
        # after any Engine._build_model_payload hook mutates messages.
        self._context_efficiency = self._snapshot_context_efficiency(
            evidence,
            root_tokens=int(prompt_tokens_est),
            worker_tokens=int(prompt_tokens_est),
            evidence_bytes_inlined=int(inlined),
        )
        self._last_call_plan = {
            "max_tokens": int(max_tokens),
            "max_tokens_source": source,
            "prompt_tokens_est": prompt_tokens_est,
            "enable_thinking": bool(thinking),
            "response_schema": bool(use_schema),
            "context_efficiency": self._context_efficiency,
        }
        return payload

    def _record_model_call(
        self,
        *,
        endpoint: str,
        finish_reason: Any,
        prompt_tokens: Any,
        completion_tokens: Any,
        wall_s: float,
        max_tokens: Any = None,
        max_tokens_source: Any = None,
        runtime_index: Any = None,
        cached_tokens: Any = None,
        prefix_key: Any = None,
        native: Any = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "endpoint": endpoint,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_s": wall_s,
        }
        if max_tokens is not None:
            entry["max_tokens"] = max_tokens
        if max_tokens_source is not None:
            entry["max_tokens_source"] = max_tokens_source
        if runtime_index is not None:
            entry["runtime_index"] = runtime_index
        if cached_tokens is not None:
            entry["cached_tokens"] = cached_tokens
        if prefix_key is not None:
            entry["prefix_key"] = prefix_key
        if native is not None:
            # The resident already reports these and `_record_model_call` was
            # dropping them, so the only evidence of KV reuse was a wall clock.
            for key in (
                "prefix_reused_tokens",
                "prefill_tokens_stepped",
                "prefill_step_count",
                "prefix_source",
                "resident_context_tokens_before",
                "shared_prefix_tokens",
                "checkpoint_missed",
                "checkpoint_restored_tokens",
                "prefix_checkpoint_taken_at",
                # Whether the resident's JSON mask actually ran. Without it a
                # malformed reply cannot be diagnosed: "the reply is NOT valid
                # JSON" means either the mask is off or the mask is wrong, and
                # those need opposite fixes. The resident has always reported
                # this field; the receipt has never carried it.
                "grammar_enforced",
                # Why generation ended. "never closed the JSON object" has
                # opposite causes -- the constraint believing it closed versus
                # the budget running out -- and the receipt could not tell them
                # apart.
                "stop_reason",
                # The body's own layer count. Dispatches per step is not
                # interpretable without it, and the host has no other source:
                # the sealed profile carries capabilities, not geometry.
                "layers",
                # The budget the connector actually GRANTED after _limits, and
                # the position limit it computed it from. The receipt has only
                # ever carried the ENGINE'S request, so a reply truncated at
                # 1446 against a recorded max_tokens of 2048 was unattributable
                # -- three reads of the request path could not say which number
                # the resident received. A field has to survive three hops and
                # these were dying on the second.
                "max_new_tokens_granted",
                "profile_max_seq_len",
                "prompt_tokens_used_for_budget",
                "prompt_token_count_source",
                "payload_max_tokens_received",
                "rendered_prompt_chars",
            ):
                value = native.get(key)
                if value is not None:
                    entry[key] = value
            trace = ((native.get("native_metrics") or {}).get("step_trace")) or None
            if isinstance(trace, dict):
                stepped = native.get("prefill_tokens_stepped")
                if not isinstance(stepped, int):
                    stepped = prompt_tokens if isinstance(prompt_tokens, int) else 0
                try:
                    from .prefill_profile import attribute, bucket_profile

                    profile = bucket_profile(trace, prefill_steps=int(stepped))
                    # The raw trace is one number per token. Only the shape is
                    # kept: a receipt is a trail, not a transcript.
                    entry["prefill_profile"] = profile
                    entry["prefill_attribution"] = attribute(profile, layers=entry.get("layers"))
                except Exception as exc:  # telemetry must never end a goal
                    entry["prefill_profile_error"] = f"{type(exc).__name__}: {exc}"
        self._model_calls.append(entry)

    def _post_completion(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                body = response.read()
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            raise EngineError(
                f"llama-server request failed: {exc}"
            ) from exc

        try:
            return json.loads(
                body.decode("utf-8", errors="replace")
            )
        except Exception as exc:
            raise EngineError(
                "llama-server returned invalid JSON"
            ) from exc

    def _thinking_leaked(
        self,
        content: Any,
        message: Dict[str, Any],
    ) -> bool:
        if isinstance(content, str) and _THINK_OPEN_RE.search(content):
            return True
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        return isinstance(reasoning, str) and bool(reasoning.strip())

    @_wall_phase("resident")
    def _call_model(
        self,
        prompt: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        compiled: Any = None,
        *,
        context_memory: Any = None,
        enable_thinking: Optional[bool] = None,
        response_schema: Optional[bool] = None,
        plain_text: bool = False,
        trailing: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        agentic = bool(getattr(self, "_agentic_execution", False))
        if self.model_client is not None:
            call = getattr(
                self.model_client,
                "complete",
                None,
            )

            if callable(call):
                self._assert_evidence_fresh(evidence)
                est = max(1, len(prompt or "") // _CHARS_PER_TOKEN)
                with self._model_call_scope(est):
                    value = call(
                        prompt=self._prompt_with_context_memory(prompt, context_memory),
                        evidence=evidence or [],
                        compiled=compiled,
                    )
                    if plain_text:
                        return self._require_plain_text(value, endpoint="model-client")
                    return value

            # A provider is not required to expose the legacy ``complete``
            # convenience method.  The model-neutral contract is
            # ``generate(GenerationRequest)``; adapt its normalized response
            # into HCLI's existing structured result boundary here.
            generate = getattr(self.model_client, "generate", None)
            if callable(generate):
                from .providers import GenerationRequest

                self._assert_evidence_fresh(evidence)
                payload = self._build_model_payload(
                    prompt,
                    evidence,
                    compiled,
                    context_memory=context_memory,
                    enable_thinking=enable_thinking,
                    response_schema=response_schema,
                    trailing=trailing,
                    history=history,
                )
                est = (self._last_call_plan or {}).get("prompt_tokens_est")
                with self._model_call_scope(est):
                    started = time.perf_counter()
                    response = generate(
                        GenerationRequest.from_mapping(payload),
                        timeout=float(os.environ.get("HCLI_MODEL_TIMEOUT", "1800")),
                    )
                    wall = time.perf_counter() - started
                    text = getattr(response, "text", None)
                    raw = getattr(response, "raw", response)
                    if isinstance(raw, dict):
                        structured = raw.get("_structured")
                        if isinstance(structured, dict):
                            return structured
                        choices = raw.get("choices")
                        if isinstance(choices, list) and choices:
                            choice = choices[0] if isinstance(choices[0], dict) else {}
                            message = choice.get("message")
                            if isinstance(message, dict):
                                text = message.get("content", text)
                            elif choice.get("text") is not None:
                                text = choice.get("text")
                        if text is None and isinstance(raw.get("content"), str):
                            text = raw["content"]
                    usage = getattr(response, "usage", None)
                    self._record_model_call(
                        endpoint=f"provider:{type(self.model_client).__name__}",
                        finish_reason=getattr(response, "finish_reason", None),
                        prompt_tokens=(usage or {}).get("prompt_tokens") if isinstance(usage, dict) else None,
                        completion_tokens=(usage or {}).get("completion_tokens") if isinstance(usage, dict) else None,
                        wall_s=wall,
                        max_tokens=payload.get("max_tokens"),
                        max_tokens_source="provider-contract",
                    )
                    if plain_text:
                        return self._require_plain_text(
                            response,
                            endpoint=f"provider:{type(self.model_client).__name__}",
                        )
                    return self._extract_json_object(text)

        backend = self._active_backend()
        use_schema = False if plain_text else self._resolve_response_schema(response_schema)
        supports_rf = backend_supports_response_format(backend)
        degrade = bool(use_schema and not supports_rf)

        thinking_arg = enable_thinking
        if degrade and enable_thinking is None:
            # mlx_lm.server accepts extra JSON keys and ignores them: a
            # response_format schema is not enforcement. A previous
            # production stall from this class of failure recovered only
            # after enable_thinking=false; a JSON schema alone was not
            # sufficient. Force thinking off on the degraded path unless
            # the caller explicitly overrode enable_thinking on this call.
            #
            # NOT a bug that config/env is ignored here: tests/
            # test_engine_schema_degrade.py::
            # test_enable_thinking_false_on_degraded_unless_overridden sets
            # config enable_thinking=True and asserts thinking is STILL forced
            # off. Only a per-call override wins, deliberately, because the
            # stall this guards against recovered only when thinking was off.
            thinking_arg = False

        def _build(
            ev: Any,
            cm: Any,
            tr: str = trailing,
            *,
            history: Optional[List[Dict[str, Any]]] = history,
        ) -> Dict[str, Any]:
            return self._build_model_payload(
                prompt,
                ev,
                compiled,
                context_memory=cm,
                enable_thinking=thinking_arg,
                response_schema=(False if degrade else response_schema),
                trailing=tr,
                history=history,
            )

        # Build the contract BEFORE fitting. `contract.apply` injects the schema
        # instruction -- about 713 tokens -- and it was being added AFTER the
        # ladder had already declared the payload a fit, so the post-contract
        # preflight refused a payload the reducer had just approved:
        #   demand 8739 exceeds per-request ctx 8192
        # The reducer must shrink against the size that will actually be posted.
        contract: Optional[StructuredOutputContract] = None
        reserve = 0
        if degrade:
            contract = self._schema_contract(
                backend,
                schema=(
                    HCLI_COMPACT_RESULT_SCHEMA
                    if history is not None
                    else HCLI_RESULT_SCHEMA
                ),
            )
            # Contract factories may clone the schema, so identity is not a
            # reliable indication that this is the compact worker contract.
            # `history is not None` is the call-site distinction: agentic
            # worker turns always pass a history list, while direct/root calls
            # retain the full result contract.
            if agentic and history is not None:
                contract.instruction = _AGENTIC_SCHEMA_INSTRUCTION
            reserve = len(str(contract.instruction or "")) // _CHARS_PER_TOKEN

        payload, reduction = self._fit_payload_to_budget(
            _build,
            evidence,
            context_memory,
            trailing=trailing,
            reserve=reserve,
            history=history,
        )
        if reduction:
            self._emit("context_reduced", reduction)
        if contract is not None:
            payload = contract.apply(payload)
        # Re-measure after return: executors.py:359 mutates
        # messages[1]["content"] in place (strips a leading "GOAL:\n")
        # after the inner snapshot, on the same dict that is posted.
        plan = self._commit_posted_prompt_estimate(payload)
        if contract is not None:
            plan["structured_output"] = _degraded_structured_record(
                contract,
                attempts=0,
                exhausted=False,
            )
        else:
            plan["structured_output"] = structured_output_record(
                mode="enforced"
            )
        self._last_call_plan = plan
        if getattr(self, "_tools_closed_for_round", False):
            self._emit(
                "closed_model_payload_prepared",
                {
                    "goal_id": self._active_goal_id,
                    "prompt_tokens": int(plan.get("prompt_tokens_est") or 0),
                    "prompt_chars": len(self._last_rendered_prompt),
                    "evidence_items": len(evidence or []),
                    "max_tokens": int(plan.get("max_tokens") or 0),
                },
            )
        pf = preflight(
            self._context_budget(),
            int(plan.get("prompt_tokens_est") or 1),
            kind="root",
        )
        if not pf.ok:
            raise ContextPreflightError(pf)

        _boundary_trace(
            "model_payload_ready",
            goal_id=self._active_goal_id,
            history_messages=len(history or []),
            message_roles=[
                str((message or {}).get("role") or "")
                for message in (payload.get("messages") or [])
                if isinstance(message, dict)
            ],
            prompt_sha256=hashlib.sha256(
                (self._last_rendered_prompt or "").encode("utf-8")
            ).hexdigest()[:16],
        )

        endpoint, provenance = self._runtime_endpoint()
        thinking_on = bool(
            payload.get("chat_template_kwargs", {}).get(
                "enable_thinking"
            )
        )

        timeout = float(
            os.environ.get(
                "HCLI_MODEL_TIMEOUT",
                "1800",
            )
        )

        pool = self._resolve_pool()
        use_pool = pool is not None and callable(getattr(pool, "complete", None))
        prefix_key = self._prefix_key(payload)
        if use_pool:
            provenance = self._snapshot_pool_provenance(pool)
            provenance["via"] = "RuntimePool.complete"
            endpoint = self._endpoint_from_provenance(provenance)
        else:
            endpoint, provenance = self._runtime_endpoint()

        with self._model_call_scope(plan.get("prompt_tokens_est")):
            self._emit(
                "workunit_started",
                {
                    "runtime": provenance,
                    "role": "primary",
                },
            )

            if contract is not None:
                parsed = self._complete_with_schema_contract(
                    contract,
                    payload,
                    timeout,
                    endpoint=endpoint,
                    provenance=provenance,
                    plan=plan,
                    prefix_key=prefix_key,
                    use_pool=use_pool,
                    pool=pool,
                    thinking_on=thinking_on,
                )
                self._emit(
                    "workunit_completed",
                    {
                        "runtime": provenance,
                        "role": "primary",
                    },
                )
                return parsed

            result, provenance, endpoint = self._invoke_completion(
                payload,
                timeout,
                endpoint=endpoint,
                provenance=provenance,
                plan=plan,
                prefix_key=prefix_key,
                use_pool=use_pool,
                pool=pool,
            )

            self._emit(
                "workunit_completed",
                {
                    "runtime": provenance,
                    "role": "primary",
                },
            )

            if plain_text:
                if not thinking_on and self._thinking_leaked(
                    result.text,
                    self._message_from_completion(result),
                ):
                    raise EngineError(
                        "provider ignored chat_template_kwargs.enable_thinking=false "
                        "for plain-text cognition"
                    )
                return self._require_plain_text(result, endpoint=endpoint)

            return self._parse_structured_reply(result, plan, thinking_on)

    def _extract_json_object(
        self,
        content: Any,
    ) -> Dict[str, Any]:
        if isinstance(content, dict):
            return _normalize_micro_mutation(content)

        text = str(
            content or ""
        ).strip()

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.I | re.S,
        ).strip()

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.I,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            ).strip()

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return _normalize_micro_mutation(parsed)
        except Exception:
            pass

        decoder = json.JSONDecoder()

        for index, char in enumerate(text):
            if char != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(
                    text[index:]
                )
            except Exception:
                continue

            if isinstance(parsed, dict):
                return _normalize_micro_mutation(parsed)

        # LAST: a patch block, which has nothing to close. This is reached only
        # when every JSON path has failed, so it costs nothing when the model
        # produces valid JSON and rescues the one failure mode that no error
        # message moved -- 1,338 to 1,446 tokens emitted on three consecutive
        # attempts with the object left open.
        block = _patch_block_to_operations(text)
        if block is not None:
            return block

        raise EngineError(
            "Model did not return a valid structured JSON object"
        )

    # -----------------------------------------------------------------
    # Result sanitization
    # -----------------------------------------------------------------

    def _strip_reasoning(
        self,
        value: Any,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: self._strip_reasoning(item)
                for key, item in value.items()
                if key.lower() not in _REASONING_KEYS
            }

        if isinstance(value, list):
            return [
                self._strip_reasoning(item)
                for item in value
            ]

        if isinstance(value, str):
            value = re.sub(
                r"<think>.*?</think>",
                "",
                value,
                flags=re.I | re.S,
            )

        return value

    def _sanitize_result(
        self,
        result: Any,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise EngineError(
                "Model result must be a JSON object"
            )

        result = self._strip_reasoning(
            result
        )

        kind = str(
            result.get(
                "kind",
                "",
            )
        ).strip().lower()

        if kind not in {
            "answer",
            "mutation",
            "tool_use",
        }:
            raise EngineError(
                "Model result kind must be answer, mutation or tool_use"
            )

        content = str(
            result.get(
                "content",
                "",
            )
        )

        operations = result.get(
            "operations",
            [],
        )

        tests = result.get(
            "tests",
            [],
        )

        if not isinstance(operations, list):
            operations = []

        if not isinstance(tests, list):
            tests = []

        raw_calls = result.get("tool_calls", [])
        tool_calls = [
            {
                "tool": str(call.get("tool") or ""),
                "arguments": call.get("arguments") or [],
            }
            for call in (raw_calls if isinstance(raw_calls, list) else [])
            if isinstance(call, dict) and str(call.get("tool") or "").strip()
        ]

        return {
            "kind": kind,
            "content": content,
            "tool_calls": tool_calls,
            "operations": operations,
            "tests": [
                str(item)
                for item in tests
                if isinstance(
                    item,
                    (str, Path),
                )
            ],
        }

    # -----------------------------------------------------------------
    # Path authority
    # -----------------------------------------------------------------

    def _safe_path(
        self,
        raw_path: Any,
        allow_missing: bool = True,
    ) -> Path:
        text = str(
            raw_path or ""
        ).strip()

        if not text:
            raise EngineError(
                "empty mutation path"
            )

        candidate = Path(text).expanduser()

        if candidate.is_absolute():
            raise EngineError(
                f"absolute path rejected: {text}"
            )

        # Canonicalize lexically first. Remaining `..` after normpath
        # means the candidate climbed out of the workspace root.
        normalized = os.path.normpath(candidate.as_posix())
        normalized_path = Path(normalized)
        if normalized_path.is_absolute() or any(
            part == ".." for part in normalized_path.parts
        ):
            raise EngineError(
                f"path escape rejected: {text}"
            )
        if normalized in {"", "."}:
            raise EngineError(
                f"path escape rejected: {text}"
            )

        joined = self.root / normalized_path

        # Containment is on the resolved path, never a lexical leftover.
        # A dangling symlink leaf is refused outright; a live symlink is
        # accepted only if its target stays inside the resolved root.
        if joined.is_symlink():
            if not joined.exists():
                raise EngineError(
                    f"dangling symlink leaf rejected: {text}"
                )
            full = joined.resolve()
        else:
            full = joined.resolve(strict=False)

        try:
            rel = full.relative_to(self.root)
        except ValueError as exc:
            raise EngineError(
                f"symlink/path escape rejected: {text}"
            ) from exc

        if not rel.parts or rel == Path("."):
            raise EngineError(
                f"path escape rejected: {text}"
            )

        protected = {
            prefix.casefold()
            for prefix in _PROTECTED_PATH_PREFIXES
        }
        hit = next(
            (
                part
                for part in rel.parts
                if part.casefold() in protected
            ),
            None,
        )
        if hit is not None:
            raise EngineError(
                f"{hit.casefold()} mutation rejected: {text}"
            )

        if not allow_missing and not full.exists():
            raise EngineError(
                f"path does not exist: {text}"
            )

        return full

    # -----------------------------------------------------------------
    # Mutation transaction
    # -----------------------------------------------------------------

    def _operation_paths(
        self,
        operations: List[Dict[str, Any]],
    ) -> List[Path]:
        paths: List[Path] = []
        seen = set()

        for operation in operations:
            if not isinstance(operation, dict):
                raise EngineError(
                    "operation must be an object"
                )

            path = self._safe_path(
                operation.get("path"),
                allow_missing=True,
            )

            key = str(path)

            if key not in seen:
                seen.add(key)
                paths.append(path)

        return paths

    def _snapshot(
        self,
        paths: Iterable[Path],
    ) -> Dict[str, Optional[bytes]]:
        snapshot: Dict[str, Optional[bytes]] = {}

        for path in paths:
            key = str(path)

            if path.exists():
                if not path.is_file():
                    raise EngineError(
                        f"mutation target is not a file: {path}"
                    )

                snapshot[key] = path.read_bytes()
            else:
                snapshot[key] = None

        return snapshot

    def _restore(
        self,
        snapshot: Dict[str, Optional[bytes]],
    ) -> None:
        for raw_path, original in snapshot.items():
            path = Path(raw_path)

            if original is None:
                if path.exists() and path.is_file():
                    path.unlink()

                continue

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_bytes(
                original
            )

    def _single_anchor(
        self,
        text: str,
        anchor: str,
        path: Path,
    ) -> None:
        count = text.count(
            anchor
        )

        if count == 1:
            return

        # An anchor that does not match is the LAST MILE of a correct patch,
        # and "found 0" tells the model nothing it can act on. Measured: a
        # mutation whose anchor was right except for one position where the
        # model emitted a literal backslash-n instead of a newline -- every
        # other line in the same string was correct. It could not see that.
        #
        # So show it the real bytes. The nearest actual text is what the anchor
        # has to equal, and quoting it turns an unfixable rejection into a
        # correctable one.
        detail = ""
        # Try EVERY line of the anchor, not just the first. The first line is
        # often the one carrying the defect, so probing only it finds nothing
        # and the model is told "no match" a second time with no new
        # information.
        at = -1
        for probe in (l.strip() for l in anchor.strip().splitlines()):
            if len(probe) > 5:
                at = text.find(probe)
                if at >= 0:
                    break
        if True:
            if at >= 0:
                start = max(0, text.rfind("\n", 0, at) + 1)
                actual = text[start:start + max(len(anchor), 240)]
                detail = (
                    f"; the file at that point actually reads:\n{actual!r}\n"
                    f"your anchor was:\n{anchor!r}\n"
                    f"copy the file's bytes exactly -- a single escaped "
                    f"newline written as a literal backslash-n will not match"
                )
            else:
                detail = "; no line of your anchor appears in the file at all"
        raise EngineError(
            f"anchor must occur exactly once in "
            f"{path.relative_to(self.root)}; "
            f"found {count}{detail}"
        )

    def _file_hashes(
        self,
        before: Dict[str, Optional[bytes]],
    ) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        for key, before_bytes in before.items():
            path = Path(key)
            after_bytes: Optional[bytes]
            if path.exists() and path.is_file():
                after_bytes = path.read_bytes()
            else:
                after_bytes = None
            try:
                rel = str(path.relative_to(self.root))
            except ValueError:
                rel = str(path)
            files.append(
                {
                    "path": rel,
                    "sha256_before": _sha256_bytes(before_bytes),
                    "sha256_after": _sha256_bytes(after_bytes),
                    "changed": before_bytes != after_bytes,
                }
            )
        return files

    def _apply_operations(
        self,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        paths = self._operation_paths(operations)
        before: Dict[str, Optional[bytes]] = {}
        for path in paths:
            key = str(path)
            if path.exists() and path.is_file():
                before[key] = path.read_bytes()
            else:
                before[key] = None

        for index, operation in enumerate(
            operations
        ):
            op = str(
                operation.get(
                    "op",
                    "",
                )
            ).strip()

            path = self._safe_path(
                operation.get("path"),
                allow_missing=True,
            )

            old_text = _operation_text(operation, "old_text") or ""

            resolved_new = _operation_text(operation, "new_text")
            has_new_text = resolved_new is not None
            new_text = resolved_new or ""

            if op == "create":
                if not has_new_text:
                    raise EngineError(
                        "create requires new_text"
                    )

                if path.exists():
                    raise EngineError(
                        f"create target already exists: "
                        f"{path.relative_to(self.root)}"
                    )

                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ATOMIC. A SIGKILL during an in-place write_text left a
                # 1374-byte truncated hybrid of the WORK PRODUCT -- neither the
                # old file nor the new one, measured by a real kill. This is the
                # mutation the mission was accepted for; losing it to a crash
                # loses accepted work, which is the worst consequence rank in
                # the persistence audit.
                _atomic_write_text(path, new_text)

                continue

            if op == "replace_file":
                if not path.exists():
                    raise EngineError(
                        f"replace_file target missing: "
                        f"{path.relative_to(self.root)}"
                    )

                current = path.read_text(
                    encoding="utf-8",
                    errors="strict",
                )
                if current == new_text:
                    raise NoOpMutation(
                        "replace_file content unchanged"
                    )

                # Record what this write DOES to the file, onto the operation
                # the receipt echoes. Without it the receipt's only account of a
                # whole-file replace is the model's own summary of it.
                radius = blast_radius(current, new_text)
                try:
                    operation["blast_radius"] = radius
                except Exception:  # noqa: BLE001 - a frozen mapping must not fail the write
                    pass
                if radius["mostly_deleted"]:
                    self._emit(
                        "mutation_blast_radius",
                        {
                            "path": str(path.relative_to(self.root)),
                            **radius,
                        },
                    )

                # ATOMIC. A SIGKILL during an in-place write_text left a
                # 1374-byte truncated hybrid of the WORK PRODUCT -- neither the
                # old file nor the new one, measured by a real kill. This is the
                # mutation the mission was accepted for; losing it to a crash
                # loses accepted work, which is the worst consequence rank in
                # the persistence audit.
                _atomic_write_text(path, new_text)

                continue

            if not path.exists():
                raise EngineError(
                    f"operation target missing: "
                    f"{path.relative_to(self.root)}"
                )

            text = path.read_text(
                encoding="utf-8",
                errors="strict",
            )

            if op == "replace":
                if not old_text:
                    raise EngineError(
                        "replace requires old_text"
                    )

                self._single_anchor(
                    text,
                    old_text,
                    path,
                )

                if old_text == new_text:
                    raise NoOpMutation(
                        "replace old_text equals new_text"
                    )

                text = text.replace(
                    old_text,
                    new_text,
                    1,
                )

            elif op == "insert_before":
                if not old_text:
                    raise EngineError(
                        "insert_before requires old_text anchor"
                    )

                if not new_text:
                    raise NoOpMutation(
                        "insert_before with empty new_text"
                    )

                self._single_anchor(
                    text,
                    old_text,
                    path,
                )

                text = text.replace(
                    old_text,
                    new_text + old_text,
                    1,
                )

            elif op == "insert_after":
                if not old_text:
                    raise EngineError(
                        "insert_after requires old_text anchor"
                    )

                if not new_text:
                    raise NoOpMutation(
                        "insert_after with empty new_text"
                    )

                self._single_anchor(
                    text,
                    old_text,
                    path,
                )

                text = text.replace(
                    old_text,
                    old_text + new_text,
                    1,
                )

            elif op == "append":
                if not new_text:
                    raise NoOpMutation(
                        "append with empty new_text"
                    )

                text += new_text

            else:
                raise EngineError(
                    f"unsupported mutation op "
                    f"{op!r} at index {index}"
                )

            # ATOMIC, same reason as above: this is the work product.
            _atomic_write_text(path, text)

        files = self._file_hashes(before)
        if not any(item.get("changed") for item in files):
            raise NoOpMutation(
                files=files,
            )
        # Bytes changing is not enough for a Python mutation: a model can add
        # a marker/comment or whitespace, pass py_compile and still claim
        # executable progress. Reject that at the transaction boundary and
        # restore the snapshot so direct callers get the same atomic result as
        # the execute() rollback path.
        for raw_path, before_bytes in before.items():
            path = Path(raw_path)
            if before_bytes is None or path.suffix.lower() != ".py":
                continue
            try:
                before_tree = ast.parse(
                    before_bytes.decode("utf-8"),
                    filename=str(path),
                )
                after_tree = ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                )
            except (UnicodeDecodeError, SyntaxError):
                continue
            if ast.dump(before_tree, include_attributes=False) == ast.dump(
                after_tree,
                include_attributes=False,
            ):
                self._restore(before)
                raise NoOpMutation(
                    "Python mutation changes no executable syntax",
                    files=files,
                )

        return {
            "files": files,
            "changed": True,
        }

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _status_from_validation(self, validation: Any) -> str:
        if isinstance(validation, bool):
            return "completed" if validation else "failed"
        if not isinstance(validation, dict):
            return "failed"
        if validation.get("ok") is True:
            return "completed"
        if str(validation.get("reason") or "") == "NO_EVIDENCE":
            return "unverified"
        return "failed"

    def _test_subprocess_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for key in _TEST_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                env[key] = value
        # Contained test subprocesses must see the same `hcli` this
        # process loaded. The package parent is repo-root (editable) or
        # site-packages (installed). Not a fossil-name sys.path hack.
        import hcli as _hcli_pkg

        container = str(Path(_hcli_pkg.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = container + os.pathsep + str(self.root)
        return env

    def _kill_process_group(self, proc: subprocess.Popen[Any]) -> None:
        if proc.pid is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _run_contained_subprocess(
        self,
        argv: List[str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.Popen(
            argv,
            cwd=str(self.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=self._test_subprocess_env(),
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            raise subprocess.TimeoutExpired(
                argv,
                timeout,
                output=stdout,
                stderr=stderr,
            )
        return subprocess.CompletedProcess(
            argv,
            proc.returncode,
            stdout,
            stderr,
        )

    def _pytest_importable(self) -> bool:
        cached = getattr(self, "_pytest_importable_cached", None)
        if cached is not None:
            return bool(cached)
        import importlib.util

        cached = importlib.util.find_spec("pytest") is not None
        self._pytest_importable_cached = cached
        return bool(cached)

    def _file_is_pytest_idiom(self, path: Path) -> bool:
        if not path.is_file() or path.suffix != ".py":
            return False
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except Exception:
            return False
        return _ast_is_pytest_idiom(tree)

    def _pytest_argv(self, path: Path) -> List[str]:
        try:
            display = str(path.relative_to(self.root))
        except ValueError:
            display = str(path)
        return [
            sys.executable,
            "-m",
            "pytest",
            display,
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "--color=no",
        ]

    def _admit_test(self, raw: str) -> Dict[str, Any]:
        raw = str(raw or "").strip()
        if not raw:
            return {
                "admitted": False,
                "reason": "NOT_ADMITTED",
                "argv": None,
            }

        tokens: Optional[List[str]]
        if raw.endswith(".py") and " " not in raw:
            tokens = [raw]
        else:
            try:
                tokens = shlex.split(raw)
            except Exception:
                return {
                    "admitted": False,
                    "reason": "NOT_ADMITTED",
                    "argv": None,
                }

        if not tokens:
            return {
                "admitted": False,
                "reason": "NOT_ADMITTED",
                "argv": None,
            }

        wants_pytest = False
        rest: List[str]
        path_from_python: Optional[str] = None

        if (
            _is_python_invoker(tokens[0])
            and len(tokens) >= 3
            and tokens[1] == "-m"
            and tokens[2] == "pytest"
        ):
            wants_pytest = True
            rest = tokens[3:]
        elif tokens[0] in {"pytest", "py.test"}:
            wants_pytest = True
            rest = tokens[1:]
        elif _is_python_invoker(tokens[0]) and len(tokens) == 2:
            path_from_python = tokens[1]
            rest = []
        elif len(tokens) == 1:
            path_from_python = tokens[0]
            rest = []
        else:
            return {
                "admitted": False,
                "reason": "NOT_ADMITTED",
                "argv": None,
            }

        path_token = path_from_python
        skip_next = False
        extras: List[str] = []
        for index, tok in enumerate(rest):
            if skip_next:
                skip_next = False
                continue
            if tok.startswith("-"):
                if tok in {"-q", "-qq", "-v", "-x", "-s"}:
                    continue
                if tok.startswith("--tb="):
                    continue
                if tok == "--color=no":
                    continue
                if (
                    tok == "-p"
                    and index + 1 < len(rest)
                    and rest[index + 1] == "no:cacheprovider"
                ):
                    skip_next = True
                    continue
                return {
                    "admitted": False,
                    "reason": "NOT_ADMITTED",
                    "argv": None,
                }
            extras.append(tok)

        if path_token is None:
            if len(extras) != 1:
                return {
                    "admitted": False,
                    "reason": "NOT_ADMITTED",
                    "argv": None,
                }
            path_token = extras[0]
        elif extras:
            return {
                "admitted": False,
                "reason": "NOT_ADMITTED",
                "argv": None,
            }

        try:
            path = self._safe_path(
                path_token,
                allow_missing=False,
            )
        except EngineError:
            return {
                "admitted": False,
                "reason": "NOT_ADMITTED",
                "argv": None,
            }

        if path.is_dir():
            if not wants_pytest:
                return {
                    "admitted": False,
                    "reason": "NOT_ADMITTED",
                    "argv": None,
                }
            if not self._pytest_importable():
                return {
                    "admitted": False,
                    "reason": "PYTEST_UNAVAILABLE",
                    "argv": None,
                }
            return {
                "admitted": True,
                "reason": None,
                "argv": self._pytest_argv(path),
                "runner": "pytest",
                "path": path,
            }

        if path.suffix != ".py":
            return {
                "admitted": False,
                "reason": "NOT_ADMITTED",
                "argv": None,
            }

        # A file that NAMES ITSELF a test is scored as a test, always.
        #
        # _file_is_pytest_idiom inspects CONTENT, so a file with no test
        # functions is not "pytest idiom" and used to fall through to the
        # `script` runner -- plain `python file.py`, scored on exit code alone.
        # Measured 2026-09-05: HCLI named tools/sovereign/test_g002_attribution.py
        # as its proving test, wrote a file containing only a docstring, and the
        # mutation was ACCEPTED on runner=script exit 0 with collected=None.
        # Its two previous attempts wrote real tests and were correctly rejected
        # (runner=pytest, collected 1, passed 0, TEST_FAILED) -- so the heuristic
        # routed the EMPTY file to the one runner that cannot notice it is empty.
        #
        # Under pytest an empty test file returns rc 5, which _pytest_evidence
        # already scores as NO_EVIDENCE. Forcing pytest by NAME closes the hole.
        use_pytest = (
            wants_pytest
            or _looks_like_a_test_filename(path)
            or self._file_is_pytest_idiom(path)
        )
        if use_pytest:
            if not self._pytest_importable():
                return {
                    "admitted": False,
                    "reason": "PYTEST_UNAVAILABLE",
                    "argv": None,
                }
            return {
                "admitted": True,
                "reason": None,
                "argv": self._pytest_argv(path),
                "runner": "pytest",
                "path": path,
            }

        return {
            "admitted": True,
            "reason": None,
            "argv": [sys.executable, str(path)],
            "runner": "script",
            "path": path,
        }

    def _pytest_evidence(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Dict[str, Any]:
        text = f"{stdout or ''}\n{stderr or ''}"
        collected: Optional[int] = None
        match = _COLLECTED_RE.search(text)
        if match:
            collected = int(match.group(1))

        executed = 0
        passed = 0
        for item in _SUMMARY_COUNT_RE.finditer(text):
            count = int(item.group(1))
            kind = item.group(2)
            executed += count
            if kind == "passed":
                passed += count

        collected_out = 0 if collected is None else collected
        executed_out = executed if executed else collected_out

        if returncode == 5 or collected == 0:
            return {
                "reason": "NO_EVIDENCE",
                "collected": collected_out,
                "executed": executed_out,
                "passed": passed,
            }

        ran = (collected is not None and collected >= 1) or executed >= 1
        if not ran or passed < 1:
            # Zero passing tests is not evidence, including an all-skipped
            # run that exits 0. A non-zero rc with collected failures is
            # still a test failure (must roll back), not a silent keep.
            reason = "NO_EVIDENCE"
            if returncode not in (0, 5) and ran:
                reason = "TEST_FAILED"
            return {
                "reason": reason,
                "collected": collected_out if collected is not None else executed,
                "executed": executed_out,
                "passed": passed,
            }

        return {
            "reason": None if returncode == 0 else "TEST_FAILED",
            "collected": collected if collected is not None else executed,
            "executed": executed if executed else collected,
            "passed": passed,
        }

    def _test_record_passed(self, rec: Dict[str, Any]) -> bool:
        if rec.get("admitted") is not True:
            return False
        if rec.get("reason") in {
            "NO_EVIDENCE",
            "PYTEST_UNAVAILABLE",
            "NOT_ADMITTED",
        }:
            return False
        if rec.get("runner") == "pytest":
            return (
                rec.get("exit_code") == 0
                and int(rec.get("collected") or 0) >= 1
                and int(rec.get("passed") or 0) >= 1
            )
        return rec.get("exit_code") == 0

    def _test_record_failed(self, rec: Dict[str, Any]) -> bool:
        if rec.get("admitted") is not True:
            return False
        if rec.get("reason") == "NO_EVIDENCE":
            return False
        if rec.get("runner") == "pytest":
            if int(rec.get("collected") or 0) < 1:
                return False
            return rec.get("exit_code") not in (0, None)
        return rec.get("exit_code") not in (0, None)

    def _compute_red_before_green(
        self,
        pre_mutation: Optional[Dict[str, Any]],
        post: Dict[str, Any],
    ) -> Tuple[Optional[bool], Optional[str]]:
        # What this CAN establish:
        #   False -- every pre-mutation proving test passed (already green)
        #   True  -- post tests passed AND pre tests were not green: an
        #            admitted failure, or an admitted run that did not pass
        #            (collection/import error scored NO_EVIDENCE). A proving
        #            test that cannot even collect against the pre-mutation
        #            producer is not green.
        # What this CANNOT establish (caller must fail closed, not accept):
        #   pre_mutation is None              -- pre_mutation_pass_not_run
        #   no post test records              -- no post-mutation test records
        #   no pre test records               -- pre_mutation_tests_did_not_run
        #   pre tests not admitted / mixed    -- could_not_establish
        #   pre not-green but post not green  -- post_mutation_tests_did_not_pass
        if pre_mutation is None:
            return None, "pre_mutation_pass_not_run"

        pre_tests = [
            c
            for c in pre_mutation.get("checks", [])
            if c.get("kind") == "test"
        ]
        post_tests = [
            c
            for c in post.get("checks", [])
            if c.get("kind") == "test"
        ]

        if not post_tests:
            return None, "no post-mutation test records"

        post_all_pass = all(
            self._test_record_passed(c) for c in post_tests
        )

        if not pre_tests:
            return None, "pre_mutation_tests_did_not_run"

        pre_all_pass = all(
            self._test_record_passed(c) for c in pre_tests
        )
        pre_any_fail = any(
            self._test_record_failed(c) for c in pre_tests
        )

        if pre_all_pass:
            return False, None
        if not post_all_pass:
            if pre_any_fail:
                return None, "post_mutation_tests_did_not_pass"
            return None, "could_not_establish"
        if pre_any_fail:
            return True, None
        pre_admitted = [
            c for c in pre_tests if c.get("admitted") is True
        ]
        if pre_admitted and all(
            not self._test_record_passed(c) for c in pre_admitted
        ):
            return True, None
        return None, "could_not_establish"

    def _safe_test_argv(
        self,
        raw: str,
    ) -> Optional[List[str]]:
        admitted = self._admit_test(raw)
        if not admitted.get("admitted"):
            return None
        argv = admitted.get("argv")
        if not argv:
            return None
        return list(argv)

    def _path_token_from_test_request(
        self,
        raw: str,
    ) -> Optional[Path]:
        text = str(raw or "").strip()
        if not text:
            return None
        token: Optional[str] = None
        if text.endswith(".py") and " " not in text:
            token = text
        else:
            try:
                parts = shlex.split(text)
            except Exception:
                parts = text.split()
            for tok in reversed(parts):
                if tok.endswith(".py") and not tok.startswith("-"):
                    token = tok
                    break
        if not token:
            return None
        try:
            return self._safe_path(
                token,
                allow_missing=True,
            )
        except EngineError:
            return None

    def _named_proving_test_paths(
        self,
        tests: List[str],
    ) -> List[Path]:
        found: List[Path] = []
        seen = set()
        for raw in tests:
            admitted = self._admit_test(raw)
            got = admitted.get("path")
            path: Optional[Path]
            if got is not None:
                path = Path(got)
            else:
                path = self._path_token_from_test_request(raw)
            if path is None:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
        return found

    def _snapshot_key_is_proving_test(
        self,
        raw: str,
        test_keys: set,
    ) -> bool:
        if raw in test_keys:
            return True
        try:
            return str(Path(raw).resolve()) in test_keys
        except Exception:
            return False

    def _run_proving_tests_against_pre_mutation_producers(
        self,
        snapshot: Dict[str, Optional[bytes]],
        paths: List[Path],
        tests: List[str],
    ) -> Dict[str, Any]:
        """Run the named proving tests against the pre-mutation producer.

        After apply, test files may be new or rewritten. Red-before-green
        asks whether THOSE tests pass without the producer changes. Running
        the old test files (if any) before apply cannot see a newly written
        proving test, which is how an import-smoke file was accepted as
        evidence of a producer change that never happened.

        A proving-test path is left at its post-mutation bytes; every other
        snapshot path is restored for the duration of this run.
        """
        test_keys = {
            str(p)
            for p in self._named_proving_test_paths(tests)
        }
        extra = set()
        for key in test_keys:
            try:
                extra.add(str(Path(key).resolve()))
            except Exception:
                pass
        test_keys |= extra

        producer_snapshot: Dict[str, Optional[bytes]] = {}
        for raw, original in snapshot.items():
            if self._snapshot_key_is_proving_test(raw, test_keys):
                continue
            producer_snapshot[raw] = original

        post_producer: Dict[str, Optional[bytes]] = {}
        for raw in producer_snapshot:
            path = Path(raw)
            if path.exists() and path.is_file():
                post_producer[raw] = path.read_bytes()
            else:
                post_producer[raw] = None

        try:
            if producer_snapshot:
                self._restore(producer_snapshot)
            existing = [
                p
                for p in paths
                if p.exists() and p.is_file()
            ]
            return self._validate(
                existing,
                tests,
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": (
                    "pre_mutation_exception:"
                    f"{type(exc).__name__}"
                ),
                "error": str(exc),
                "checks": [],
            }
        finally:
            if post_producer:
                self._restore(post_producer)

    def _validate(
        self,
        paths: Iterable[Path],
        tests: Optional[List[str]] = None,
        pre_mutation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        ok = True
        test_list = list(tests) if tests is not None else []

        for raw_path in paths:
            path = Path(raw_path)
            if path.exists():
                path = path.resolve()

            try:
                rel = str(path.relative_to(self.root))
            except ValueError:
                rel = str(path)

            if path.suffix == ".rs":
                # A Rust mutation used to record `no_checker_available` and pass,
                # so acceptance rested entirely on whatever test the reply named
                # -- and a Python test passes no matter what the Rust says.
                # Measured: a mutation that deleted the correctness guard in
                # qwen38_batched_prefill_allowed (reuse == 0 && snapshot_at
                # .is_none()) was ACCEPTED on `test hcli/test_engine_tool_loop.py
                # exit 0`. That change compiles and produces a faster wall with a
                # different answer, which is the one outcome the guard exists to
                # prevent. `cargo check` on the owning crate costs 13.5 s.
                checked = check_rust_file(path, self.root)
                checks.append(
                    {
                        "kind": "cargo_check",
                        "path": rel,
                        "package": checked.get("package"),
                        "exit_code": checked["exit_code"],
                        "stdout": checked["stdout"][-4000:],
                        "stderr": checked["stderr"][-4000:],
                    }
                )
                if checked["exit_code"] != 0:
                    ok = False
                continue

            if path.suffix != ".py":
                record = {"kind": "no_checker_available", "path": rel}
                if path.suffix.lower() in _UNCHECKABLE_SOURCE_SUFFIXES:
                    # Source the verifier cannot check must not be accepted on
                    # unrelated evidence. Silence here is not a passing check.
                    record["fatal"] = True
                    record["reason"] = (
                        f"{path.suffix} is source and this verifier has no checker "
                        "for it; a mutation to it cannot be accepted on a test in "
                        "another language"
                    )
                    ok = False
                checks.append(record)
                continue

            compiled = compile_python_file(path)
            record = {
                "kind": "py_compile",
                "path": rel,
                "exit_code": compiled["exit_code"],
                "stdout": compiled["stdout"][-4000:],
                "stderr": compiled["stderr"][-4000:],
            }

            checks.append(record)

            if compiled["exit_code"] != 0:
                ok = False

        if ok:
            if not test_list:
                ok = False
            else:
                for raw in test_list:
                    admitted = self._admit_test(raw)
                    if not admitted.get("admitted"):
                        checks.append(
                            {
                                "kind": "test",
                                "requested": raw,
                                "admitted": False,
                                "exit_code": None,
                                "reason": admitted.get("reason")
                                or "NOT_ADMITTED",
                            }
                        )
                        ok = False
                        continue

                    argv = list(admitted.get("argv") or [])
                    runner = admitted.get("runner") or "script"

                    try:
                        proc = self._run_contained_subprocess(
                            argv,
                            timeout=300,
                        )
                    except Exception as exc:
                        checks.append(
                            {
                                "kind": "test",
                                "requested": raw,
                                "admitted": True,
                                "runner": runner,
                                "exit_code": None,
                                "reason": f"TEST_ERROR:{type(exc).__name__}",
                                "stderr": str(exc)[-8000:],
                            }
                        )
                        ok = False
                        continue

                    record = {
                        "kind": "test",
                        "requested": raw,
                        "admitted": True,
                        "runner": runner,
                        "exit_code": proc.returncode,
                        "stdout": proc.stdout[-8000:],
                        "stderr": proc.stderr[-8000:],
                    }

                    if runner == "pytest":
                        evidence = self._pytest_evidence(
                            proc.stdout,
                            proc.stderr,
                            proc.returncode,
                        )
                        record["collected"] = evidence.get("collected")
                        record["executed"] = evidence.get("executed")
                        record["passed"] = evidence.get("passed")
                        if evidence.get("reason"):
                            record["reason"] = evidence["reason"]
                            ok = False
                        elif proc.returncode != 0:
                            record["reason"] = "TEST_FAILED"
                            ok = False
                    elif proc.returncode != 0:
                        record["reason"] = "TEST_FAILED"
                        ok = False

                    checks.append(record)

        result: Dict[str, Any] = {
            "ok": ok,
            "checks": checks,
        }

        if not test_list:
            result["ok"] = False
            result["reason"] = "NO_EVIDENCE"
        elif not ok:
            test_checks = [
                c for c in checks if c.get("kind") == "test"
            ]
            if test_checks and all(
                c.get("reason") == "NO_EVIDENCE"
                for c in test_checks
            ):
                result["reason"] = "NO_EVIDENCE"

        rbg, rbg_reason = self._compute_red_before_green(
            pre_mutation,
            result,
        )
        result["red_before_green"] = rbg
        # Continuity: receipts still carry this flag. It is NOT the
        # acceptance decision. A proving test that was already green
        # (red_before_green is False) or whose before-state cannot be
        # established (None) is refused below when this is a post-mutation
        # pass (pre_mutation provided). Direct _validate probes, including
        # the pre-mutation run itself, pass pre_mutation=None and are not
        # refused here.
        result["red_before_green_advisory"] = True
        if rbg is None and rbg_reason:
            result["red_before_green_reason"] = rbg_reason

        if result.get("ok") and pre_mutation is not None:
            if rbg is False:
                result["ok"] = False
                result["reason"] = "NOT_RED_BEFORE"
            elif rbg is None:
                result["ok"] = False
                result["reason"] = "RED_BEFORE_UNESTABLISHED"

        if pre_mutation is not None:
            result["pre_mutation"] = {
                "ok": pre_mutation.get("ok"),
                "reason": pre_mutation.get("reason"),
                "checks": [
                    {
                        key: rec[key]
                        for key in (
                            "kind",
                            "requested",
                            "admitted",
                            "exit_code",
                            "reason",
                            "collected",
                            "executed",
                            "passed",
                            "runner",
                        )
                        if key in rec
                    }
                    for rec in pre_mutation.get("checks", [])
                    if rec.get("kind") == "test"
                ],
            }

        return result

    # -----------------------------------------------------------------
    # Receipts
    # -----------------------------------------------------------------

    def _serialize_compiled(
        self,
        compiled: Any,
    ) -> Dict[str, Any]:
        if not isinstance(compiled, dict):
            return {}

        result: Dict[str, Any] = {}

        for key in (
            "invariants",
            "acceptance_criteria",
        ):
            value = compiled.get(key)

            if isinstance(value, list):
                result[key] = [
                    str(item)
                    for item in value
                ]

        dag = compiled.get(
            "workunits"
        )

        units = getattr(
            dag,
            "units",
            None,
        )

        if isinstance(units, dict):
            result["workunits"] = [
                (
                    unit.to_dict()
                    if hasattr(
                        unit,
                        "to_dict",
                    )
                    else {
                        "id": getattr(
                            unit,
                            "id",
                            None,
                        ),
                        "description": getattr(
                            unit,
                            "description",
                            "",
                        ),
                        "dependencies": list(
                            getattr(
                                unit,
                                "dependencies",
                                [],
                            )
                        ),
                    }
                )
                for unit in units.values()
            ]

        return result

    def _runtime_provenance(
        self,
    ) -> List[Dict[str, Any]]:
        if self.runtime_state_provider is None:
            return []

        try:
            pool = self.runtime_state_provider()
        except Exception:
            return []

        if pool is None:
            return []

        result = []

        for runtime in getattr(
            pool,
            "runtimes",
            [],
        ):
            backend = getattr(runtime, "backend", None)
            identity = {}
            if backend is not None:
                identity_fn = getattr(backend, "identity", None)
                if callable(identity_fn):
                    try:
                        value = identity_fn()
                        if isinstance(value, dict):
                            identity = value
                    except Exception as exc:  # noqa: BLE001 - receipt remains useful
                        identity = {"identity_error": f"{type(exc).__name__}: {exc}"}
            profile = None
            if backend is not None:
                try:
                    from .providers import profile_from_backend

                    profile = profile_from_backend(backend).to_dict()
                except Exception:
                    profile = None
            result.append(
                {
                    "index": getattr(
                        runtime,
                        "index",
                        None,
                    ),
                    "pid": getattr(
                        runtime,
                        "pid",
                        None,
                    ),
                    "port": getattr(
                        runtime,
                        "port",
                        None,
                    ),
                    "active": getattr(
                        runtime,
                        "active",
                        None,
                    ),
                    "provider": identity.get("provider") or identity.get("runtime"),
                    "model_identity": identity.get("model_identity"),
                    "identity": identity,
                    "profile": profile,
                }
            )

        return result

    def _write_receipt(
        self,
        goal_id: str,
        goal: str,
        result: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        validation: Any,
        rolled_back: bool,
        started: Optional[datetime] = None,
    ) -> str:
        started = started or datetime.now(
            timezone.utc
        )

        finished = datetime.now(
            timezone.utc
        )

        receipt_dir = (
            self.root
            / ".hcli"
            / "receipts"
        )

        receipt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = receipt_dir / (
            f"{goal_id}.json"
        )

        from .result_envelope import build_result_envelope

        result_envelope = build_result_envelope(
            goal=goal,
            result=result,
            evidence=evidence,
            validation=validation,
            runtime_provenance=self._runtime_provenance(),
            model_calls=self._model_calls,
            receipt_path=path,
        )

        receipt = {
            "goal_id": goal_id,
            "goal": goal,
            "model": self.model_name,
            "runtime_count": self.runtime_count,
            "runtime_provenance": self._runtime_provenance(),
            "evidence_files": [
                item["path"]
                for item in evidence
            ],
            "context_efficiency": (
                dict(self._context_efficiency)
                if self._context_efficiency
                else self._snapshot_context_efficiency(evidence)
            ),
            "kind": result.get("kind"),
            "operations": result.get(
                "operations",
                [],
            ),
            "tests": result.get(
                "tests",
                [],
            ),
            "validation": validation,
            "result_envelope": result_envelope,
            "rolled_back": bool(
                rolled_back
            ),
            "status": result.get(
                "status",
                "unknown",
            ),
            "timestamps": {
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
            },
            "model_calls": list(self._model_calls or []),
            "max_model_in_flight": int(self.max_model_in_flight),
        }

        so = None
        if isinstance(self._last_call_plan, dict):
            so = self._last_call_plan.get("structured_output")
        if isinstance(so, dict) and so.get("mode"):
            receipt["structured_output"] = so

        status = str(
            result.get("status") or "unknown"
        )
        kind = result.get("kind")
        is_failure = status in {
            "failed",
            "cancelled",
        } or (
            kind == "error"
            and status != "completed"
        )
        if is_failure:
            error = str(
                result.get("error") or ""
            ).strip()
            error_type = str(
                result.get("error_type") or ""
            ).strip()
            error_tb = str(
                result.get("error_traceback") or ""
            ).strip()
            if not error:
                error = "unknown error"
            if not error_type:
                error_type = "Error"
            if not error_tb:
                error_tb = error
            receipt["error"] = error
            receipt["error_type"] = error_type
            receipt["error_traceback"] = error_tb

        # ATOMIC. A SIGKILL mid-write left a 1648-byte truncated receipt that
        # raised JSONDecodeError on read -- and this file is the PROOF OF
        # ACCEPTANCE. An unreadable receipt makes verified work indistinguishable
        # from work that never ran.
        _boundary_trace(
            "receipt_write_begin",
            goal_id=goal_id,
            receipt_path=str(path),
            status=receipt.get("status"),
        )
        _atomic_write_text(
            path,
            json.dumps(
                self._strip_reasoning(
                    receipt
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        _boundary_trace(
            "receipt_write_end",
            goal_id=goal_id,
            receipt_path=str(path),
            bytes=path.stat().st_size if path.exists() else None,
        )

        return str(path)
class FrontierEngine:
    def ranked_candidate(self, frontier_input):
        if not isinstance(frontier_input, dict):
            raise TypeError("frontier input must be a dict")
        candidates = frontier_input.get("candidates", [])
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list")
        ranked = sorted(candidates, key=lambda c: (c.get("score", 0), str(c.get("id", ""))), reverse=True)
        return {"ranked": ranked, "count": len(ranked)}

    def bounded_result(self, frontier_input):
        if not isinstance(frontier_input, dict):
            raise TypeError("frontier input must be a dict")
        return {
            "kind": "answer",
            "content": "frontier",
            "operations": [],
            "tests": [],
            "status": "completed",
            "bounded": True,
        }

    def run(self, frontier_input=None):
        result = {
            "kind": "answer",
            "content": "frontier",
            "operations": [],
            "tests": [],
            "status": "completed",
        }
        if frontier_input is not None:
            result.update(self.ranked_candidate(fronter_input))
            result.update(self.bounded_result(frontier_input))
        return result
