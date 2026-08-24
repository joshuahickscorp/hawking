"""Verifier-first pipeline, ported from ~/.claude/workflows/superagent.js.

Four phases, each independently testable. None of them import Engine; a
``ModelCaller`` is injected so this module can run without a model and can
be wired into whichever real caller exists later.

Ported lines (superagent.js), quoted:

    Decompose this into the SMALLEST set of obligations that, all met, mean
    the goal is met. Not a task list — a list of propositions each of which
    is TRUE or FALSE and can be settled by running something or reading
    something.

    Return what you found with concrete refs — absolute paths with line
    numbers, exact commands with their output, URLs with dates. A ref
    someone else can re-check is worth more than a conclusion they have
    to trust.

    Settle this claim MECHANICALLY. Find the cheapest command or file read
    whose output differs depending on whether the claim is true, run it, and
    return that command plus its output as the proof. Do not grade the
    evidence above on how confident it sounds — confidence is not evidence.
    If no command or read can settle it, return UNVERIFIABLE with an empty
    command rather than guessing.

    spec 90: a failed pipeline index settles as a falsy slot. Dropping it
    would delete the obligation from the verdicts AND the synth prompt with
    nothing saying so; carry it through as UNVERIFIABLE.

    Answer the goal from THIS material only. Do not go looking for more —
    you have no search tools and adding an uncited claim here would launder
    a guess into a finding.

Deliberately NOT replicated (belong to a workflow orchestrator HCLI does
not have): PROFILES / fan-out scheduling, the critic pass, the UNVERIFIABLE
escalation judge, targetedTests-then-fullSuite, host ``pipeline`` /
``parallel`` primitives, and sa-* sub-agent dispatch. agent_role is the
principle of narrow-tool routing, not those named sub-agents.

Agent-role mapping (superagent agentType -> HCLI agent_role):

    sa-grep     locate    where is X, what calls Y, which files touch Z
    sa-read     read      fetch known paths verbatim
    sa-test     test      run one exact command; fast path if an angle is
                          already a command (skips the model)
    sa-research research  external sources: docs, changelogs, specs
    sa-synth    reason    reason over evidence already in hand; no search
    sa-verify   settle    settle one specific falsifiable claim
    (default)   general   unassigned; generic concrete-refs prompt

The imperative heuristic in ``plan`` is a first-token check against a small
verb list (Implement/Add/Fix/Build and a few siblings). It is not a parser.
False negatives: "Please implement X", "Implementing X is required".
False positives: a proposition whose first word is exactly one of those
verbs ("Add is a reserved keyword").

Check-5 decision: OVERRIDE. If the model judges TRUE but the harness saw a
nonzero exit, the verdict is forced to FALSE. A passing command (exit 0)
does not flip FALSE up to TRUE — that would let an unrelated ``true``
promote a claim. Limitation: an inverted discriminator (nonzero means the
claim holds) will be overridden incorrectly; wrap such checks so
confirmation exits 0.
"""
from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

AGENT_ROLES = frozenset(
    {"locate", "read", "test", "research", "reason", "settle", "general"}
)

_ROLE_ALIASES = {
    "sa-grep": "locate",
    "grep": "locate",
    "locator": "locate",
    "sa-read": "read",
    "sa-test": "test",
    "sa-research": "research",
    "sa-synth": "reason",
    "synth": "reason",
    "sa-verify": "settle",
    "verify": "settle",
}

# First-token only. See module docstring for the limits of this heuristic.
_IMPERATIVE_STARTERS = frozenset(
    {
        "implement",
        "add",
        "fix",
        "build",
        "create",
        "update",
        "remove",
        "delete",
        "refactor",
        "write",
    }
)

_COMMAND_FIRST = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "cargo",
        "go",
        "npm",
        "npx",
        "node",
        "bash",
        "sh",
        "zsh",
        "make",
        "uv",
        "uvx",
        "ruff",
        "mypy",
        "git",
    }
)

# Shell builtins / tautologies that exit 0 (or 1) without inspecting the world.
_VACUOUS_FIRST = frozenset({"true", "false", ":", "exit"})



_ROLE_LATITUDE = {
    "locate": (
        "Role=locate (sa-grep). Report where things live: paths and line "
        "numbers. Do not run tests."
    ),
    "read": (
        "Role=read (sa-read). Fetch known paths verbatim with line numbers. "
        "Do not search elsewhere."
    ),
    "test": (
        "Role=test (sa-test). The decisive artifact is one exact command "
        "and its output."
    ),
    "research": (
        "Role=research (sa-research). Cite external sources with URLs and dates."
    ),
    "reason": (
        "Role=reason (sa-synth). Reason over evidence already in hand. "
        "Do not search."
    ),
    "settle": (
        "Role=settle (sa-verify). Gather the cheapest ref that would let a "
        "later mechanical check settle the claim. Do not render the verdict."
    ),
    "general": (
        "Role=general. Prefer the tightest concrete ref you can find."
    ),
}

_FAILURE_OUTPUT = (
    "its pipeline stage settled as a failure — nothing was gathered or checked"
)

PLAN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["obligations"],
    "properties": {
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "statement",
                    "agent_role",
                    "angles",
                    "consequential",
                ],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "short kebab id, unique",
                    },
                    "statement": {
                        "type": "string",
                        "description": (
                            "ONE falsifiable proposition that is true when "
                            "this obligation is met"
                        ),
                    },
                    "agent_role": {
                        "type": "string",
                        "enum": sorted(AGENT_ROLES),
                    },
                    "angles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-3 distinct ways to attack it, MOST DECISIVE FIRST"
                        ),
                    },
                    "consequential": {
                        "type": "boolean",
                        "description": (
                            "true if being wrong here is expensive, "
                            "irreversible, or contradicts something else"
                        ),
                    },
                },
            },
        }
    },
}

PROPOSE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["command"],
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "the exact command or file:line read that would settle it; "
                "empty if nothing could"
            ),
        },
        "true_if": {"type": "string"},
        "false_if": {"type": "string"},
    },
}

JUDGE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["TRUE", "FALSE", "UNVERIFIABLE"],
        },
        "why": {"type": "string"},
    },
}


class PlanError(ValueError):
    """Planner returned no usable propositions, or still a task list after retry."""


class ModelCaller(Protocol):
    def __call__(self, prompt: str, *, schema: Optional[dict] = None) -> Any: ...


@dataclass
class Obligation:
    id: str
    statement: str
    angles: List[str]
    consequential: bool
    agent_role: str = "general"


@dataclass
class Verdict:
    obligation_id: str
    verdict: str
    command: str
    output: str
    evidence: str
    exit_code: Optional[int] = None


def plan(
    goal: str, caller: ModelCaller, *, max_obligations: int = 12
) -> List[Obligation]:
    """Decompose GOAL into propositions, not a task list.

    Rejects (retry once, then raise) any statement whose first token is a
    bare imperative verb. See module docstring for the heuristic's limits.
    """
    prompt = _plan_prompt(goal)
    raw = _call(caller, prompt, PLAN_SCHEMA)
    obligations = _coerce_obligations(raw)
    if not obligations:
        raise PlanError("verifier_pipeline: planner returned no obligations")
    bad = [o for o in obligations if _unusable_statement(o.statement)]
    if bad:
        retry_prompt = _plan_retry_prompt(goal, bad)
        raw = _call(caller, retry_prompt, PLAN_SCHEMA)
        obligations = _coerce_obligations(raw)
        if not obligations:
            raise PlanError("verifier_pipeline: planner returned no obligations")
        still = [o for o in obligations if _unusable_statement(o.statement)]
        if still:
            rejected = "; ".join(o.statement for o in still)
            raise PlanError(
                "planner returned task-list obligations after retry: " + rejected
            )
    cap = max_obligations if max_obligations and max_obligations > 0 else len(obligations)
    return obligations[:cap]


def execute(
    obligation: Obligation,
    caller: ModelCaller,
    *,
    fan: int = 1,
    run_command: Optional[Callable[[str], Tuple[int, str]]] = None,
    goal: str = "",
) -> str:
    """Gather concrete refs. Returns a bare evidence string, never a verdict.

    ``agent_role="test"`` plus an angle that already names a command skips
    the model. If ``run_command`` is injected, that command is run here;
    otherwise the named command is returned as the ref.
    """
    fan_n = fan if isinstance(fan, int) and fan > 0 else 1
    if obligation.agent_role == "test":
        cmds = [a for a in (obligation.angles or []) if _looks_like_command(a)]
        if cmds:
            return _test_fast_path(cmds[:fan_n], run_command)

    angles: Sequence[str] = obligation.angles or [obligation.statement]
    chunks: List[str] = []
    for angle in list(angles)[:fan_n]:
        prompt = (
            f"OBLIGATION: {obligation.statement}\n\n"
            f"ATTACK IT THIS WAY: {angle}\n"
        )
        if goal:
            prompt += f"\nOverall goal for context only: {goal}\n"
        prompt += "\n" + _ROLE_LATITUDE.get(
            obligation.agent_role, _ROLE_LATITUDE["general"]
        )
        prompt += (
            "\n\nReturn what you found with concrete refs — absolute paths "
            "with line numbers, exact commands with their output, URLs with "
            "dates. A ref someone else can re-check is worth more than a "
            "conclusion they have to trust.\n\n"
            "Do not render a TRUE/FALSE verdict — that is verify's job alone."
        )
        chunks.append(_evidence_as_str(_call(caller, prompt)))
    return "\n\n---\n\n".join(chunks)


def verify(
    obligation: Obligation,
    evidence: str,
    caller: ModelCaller,
    run_command: Callable[[str], Tuple[int, str]],
) -> Verdict:
    """Settle the claim mechanically: propose, run, then judge the real tuple.

    An empty proposed command is UNVERIFIABLE and does not invoke
    ``run_command``. The model's claimed output is ignored. A TRUE judgement
    against a nonzero exit is overridden to FALSE (see module docstring).
    """
    evidence_text = evidence if isinstance(evidence, str) else str(evidence)
    propose_prompt = (
        f"CLAIM: {obligation.statement}\n\n"
        f"EVIDENCE GATHERED SO FAR:\n{evidence_text}\n\n"
        "Settle this claim MECHANICALLY. Find the cheapest command or file "
        "read whose output differs depending on whether the claim is true. "
        "Propose that command — the harness will run it — and return that "
        "command as the proof. Do not grade the evidence above on how "
        "confident it sounds — confidence is not evidence. If no command or "
        "read can settle it, return UNVERIFIABLE with an empty command "
        "rather than guessing.\n\n"
        'Return JSON {"command": "<exact shell command or empty string>"}. '
        "Do not invent the command's output; you will not run it."
    )
    proposed = _call(caller, propose_prompt, PROPOSE_SCHEMA)
    command = _extract_command(proposed)
    if not command:
        return Verdict(
            obligation_id=obligation.id,
            verdict="UNVERIFIABLE",
            command="",
            output=(
                "no command or read can settle it; empty command returned "
                "rather than guessing"
            ),
            evidence=evidence_text,
            exit_code=None,
        )

    admitted, refuse_reason = command_is_admissible(command)
    if not admitted:
        return Verdict(
            obligation_id=obligation.id,
            verdict="FALSE",
            command=command,
            output=refuse_reason,
            evidence=evidence_text,
            exit_code=None,
        )

    try:
        code, raw_output = run_command(command)
    except subprocess.TimeoutExpired as exc:
        return Verdict(
            obligation_id=obligation.id,
            verdict="FALSE",
            command=command,
            output=f"TIMEOUT: {exc}",
            evidence=evidence_text,
            exit_code=-1,
        )
    except Exception as exc:
        return Verdict(
            obligation_id=obligation.id,
            verdict="FALSE",
            command=command,
            output=f"COMMAND_ERROR:{type(exc).__name__}: {exc}",
            evidence=evidence_text,
            exit_code=-1,
        )
    code = int(code)
    raw_output = "" if raw_output is None else str(raw_output)
    proof = f"exit_code={code}\n{raw_output}"

    judge_prompt = (
        f"CLAIM: {obligation.statement}\n\n"
        "The harness RAN this command (do not invent a different output):\n"
        f"command: {command}\n"
        f"exit_code: {code}\n"
        f"output:\n{raw_output}\n\n"
        "Ground your verdict in this (exit_code, output) tuple. TRUE only if "
        "the actual output settles the claim as true. FALSE only if it "
        "settles the claim as false. UNVERIFIABLE if the output does not "
        "actually discriminate.\n\n"
        'Return JSON {"verdict": "TRUE"|"FALSE"|"UNVERIFIABLE"}.'
    )
    judged = _normalize_verdict(_call(caller, judge_prompt, JUDGE_SCHEMA))
    if judged == "TRUE" and code != 0:
        judged = "FALSE"
        proof = (
            proof
            + f"\n[mechanical-override] model said TRUE but command exited {code}; "
            "verdict forced to FALSE"
        )
    return Verdict(
        obligation_id=obligation.id,
        verdict=judged,
        command=command,
        output=proof,
        evidence=evidence_text,
        exit_code=code,
    )


def synthesize(goal: str, verdicts: List[Verdict], caller: ModelCaller) -> str:
    """Answer the goal from collected verdicts only. No command runner."""
    payload = []
    for item in verdicts:
        rests = "real command" if (item.command or "").strip() else "judgement"
        payload.append(
            {
                "id": item.obligation_id,
                "verdict": item.verdict,
                "proof": item.command,
                "output": item.output,
                "evidence": item.evidence,
                "rests_on": rests,
            }
        )
    prompt = (
        f"GOAL: {goal}\n\n"
        "Every obligation, its verdict, and the proof behind it:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Answer the goal from THIS material only. Do not go looking for more "
        "— you have no search tools and adding an uncited claim here would "
        "launder a guess into a finding.\n"
        "Lead with the answer. Then, plainly: which obligations rest on a "
        "real command and which rest on judgement, what the surviving "
        "attacks cost the answer, and what is still not known."
    )
    raw = _call(caller, prompt)
    return raw if isinstance(raw, str) else _evidence_as_str(raw)


def run_pipeline(
    goal: str,
    caller: ModelCaller,
    run_command: Callable[[str], Tuple[int, str]],
    *,
    fan: int = 1,
) -> dict:
    """Plan, execute, verify, synthesize. Failed slots stay UNVERIFIABLE."""
    obligations = plan(goal, caller, max_obligations=12)
    verdicts: List[Verdict] = []
    for ob in obligations:
        evidence = ""
        try:
            evidence = execute(
                ob, caller, fan=fan, run_command=run_command, goal=goal
            )
        except Exception as exc:
            verdicts.append(
                Verdict(
                    obligation_id=ob.id,
                    verdict="UNVERIFIABLE",
                    command="",
                    output=f"{_FAILURE_OUTPUT}: {exc}",
                    evidence="",
                )
            )
            continue
        try:
            verdicts.append(verify(ob, evidence, caller, run_command))
        except Exception as exc:
            verdicts.append(
                Verdict(
                    obligation_id=ob.id,
                    verdict="UNVERIFIABLE",
                    command="",
                    output=f"{_FAILURE_OUTPUT}: {exc}",
                    evidence=evidence if isinstance(evidence, str) else str(evidence),
                )
            )
    answer = synthesize(goal, verdicts, caller)
    return {
        "goal": goal,
        "answer": answer,
        "verdicts": verdicts,
        "obligations": obligations,
    }


def _plan_prompt(goal: str) -> str:
    return (
        f"GOAL: {goal}\n\n"
        "Decompose this into the SMALLEST set of obligations that, all met, "
        "mean the goal is met. Not a task list — a list of propositions each "
        "of which is TRUE or FALSE and can be settled by running something "
        "or reading something.\n\n"
        "For each obligation pick the tightest agent_role that can settle it:\n"
        "  locate     where is X, what calls Y, which files touch Z\n"
        "  read       fetch known paths verbatim\n"
        "  test       run one exact command and report PASS/FAIL\n"
        "  research   external sources: docs, changelogs, specs\n"
        "  reason     reason over evidence already in hand\n"
        "  settle     settle one specific falsifiable claim\n"
        "Do not send a locate job to reason or a reasoning job to locate.\n\n"
        "angles: order them by how DECISIVE they are, most decisive first — "
        "only the first few will be run.\n"
        "consequential: true when being wrong is expensive, irreversible, or "
        "would contradict something already established.\n\n"
        "A statement is a proposition (\"retry logic is present in foo.py\"), "
        "never an imperative (\"Implement the retry logic\", \"Fix Y\", "
        "\"Add X\", \"Build Z\")."
    )


def _plan_retry_prompt(goal: str, rejected: Sequence[Obligation]) -> str:
    listed = "\n".join(f"- {o.statement}" for o in rejected)
    return (
        f"{_plan_prompt(goal)}\n\n"
        "Your previous obligations included items that are TASKS "
        "(imperatives), not propositions. Rejected:\n"
        f"{listed}\n\n"
        "Decompose this again. Every statement must read as a TRUE/FALSE "
        "claim, not an instruction."
    )


def _call(
    caller: ModelCaller, prompt: str, schema: Optional[dict] = None
) -> Any:
    try:
        return caller(prompt, schema=schema)
    except TypeError:
        return caller(prompt)


def _coerce_obligations(raw: Any) -> List[Obligation]:
    data = _as_data(raw)
    items: Any = None
    if isinstance(data, dict):
        items = data.get("obligations")
    elif isinstance(data, list):
        items = data
    if not items:
        return []
    out: List[Obligation] = []
    for i, item in enumerate(items):
        if isinstance(item, Obligation):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        out.append(_obligation_from_raw(item, i))
    return out


def _obligation_from_raw(raw: dict, index: int) -> Obligation:
    statement = str(raw.get("statement") or "").strip()
    oid = str(raw.get("id") or f"ob-{index + 1}").strip()
    angles_raw = raw.get("angles") or []
    if isinstance(angles_raw, str):
        angles = [angles_raw] if angles_raw.strip() else []
    else:
        angles = [str(a) for a in angles_raw if str(a).strip()]
    role = raw.get("agent_role") or raw.get("agentType") or "general"
    return Obligation(
        id=oid or f"ob-{index + 1}",
        statement=statement,
        angles=angles,
        consequential=bool(raw.get("consequential", False)),
        agent_role=_normalize_role(str(role)),
    )


def _normalize_role(role: str) -> str:
    key = role.strip().lower()
    key = _ROLE_ALIASES.get(key, key)
    if key not in AGENT_ROLES:
        return "general"
    return key


def _unusable_statement(statement: str) -> bool:
    if not statement.strip():
        return True
    return _is_imperative_statement(statement)


def _is_imperative_statement(statement: str) -> bool:
    return _first_token(statement).lower() in _IMPERATIVE_STARTERS


def _first_token(statement: str) -> str:
    text = statement.strip()
    text = re.sub(r"^[\s#>*\-]+", "", text)
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    text = text.strip().strip("\"'`")
    if not text:
        return ""
    token = text.split()[0]
    return re.sub(r"[^A-Za-z]+$", "", token)


def _looks_like_command(text: str) -> bool:
    s = text.strip()
    if not s or s.endswith("?"):
        return False
    admitted, _reason = command_is_admissible(s)
    return admitted


def _test_fast_path(
    cmds: Sequence[str],
    run_command: Optional[Callable[[str], Tuple[int, str]]],
) -> str:
    parts: List[str] = []
    for cmd in cmds:
        if run_command is None:
            parts.append(
                "named command (test-role fast path, runner not injected): "
                + cmd
            )
            continue
        code, output = run_command(cmd)
        parts.append(
            f"command: {cmd}\nexit_code={int(code)}\noutput:\n"
            f"{'' if output is None else output}"
        )
    return "\n\n---\n\n".join(parts)


def _as_data(response: Any) -> Any:
    if not isinstance(response, str):
        return response
    text = _strip_fence(response.strip())
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_command(raw: Any) -> str:
    data = _as_data(raw)
    if data is None:
        return ""
    if isinstance(data, dict):
        cmd = data.get("command")
        if cmd is None:
            return ""
        return str(cmd).strip()
    if isinstance(data, str):
        stripped = data.strip()
        if not stripped or stripped.upper().startswith("UNVERIFIABLE"):
            return ""
        return stripped
    return str(data).strip()


def _normalize_verdict(raw: Any) -> str:
    """Accept only the defined enum. Any other string is UNVERIFIABLE.

    Prefix matches ("TRUE-ISH") and aliases ("VERIFIED") used to collapse
    to TRUE, which is a truthy pass of a non-enum verdict.
    """
    data = _as_data(raw)
    text = ""
    if isinstance(data, dict):
        text = str(data.get("verdict") or "")
    elif isinstance(data, str):
        text = data
    else:
        text = str(data or "")
    upper = text.strip().upper()
    if upper == "TRUE":
        return "TRUE"
    if upper == "FALSE":
        return "FALSE"
    return "UNVERIFIABLE"


def _evidence_as_str(raw: Any) -> str:
    data = _as_data(raw)
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("evidence", "refs", "found", "content"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        filtered = {k: v for k, v in data.items() if k not in {"verdict", "why"}}
        return json.dumps(filtered, ensure_ascii=False)
    return str(data)


def _tokens(command: str) -> List[str]:
    try:
        return shlex.split(command.strip(), posix=True)
    except ValueError:
        return command.strip().split()


def _call_is_exit_zero(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "SystemExit":
        return True
    if not isinstance(node, ast.Call):
        return False
    name = ""
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    # `os._exit(0)` and `quit()` end the process just as cleanly as sys.exit,
    # so a body whose only effect is one of them cannot fail either.
    if name not in {"exit", "SystemExit", "_exit", "quit"}:
        return False
    if not node.args:
        return True
    arg0 = node.args[0]
    return isinstance(arg0, ast.Constant) and arg0.value in (0, None, False)


def _stmt_is_vacuous(stmt: ast.stmt) -> bool:
    """True when this statement cannot make the command fail."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(stmt, ast.Assert):
        # `assert True` and `assert 1` are decoration, not a check. A constant
        # test can never be false, so the assertion cannot discriminate.
        test = stmt.test
        if isinstance(test, ast.Constant):
            return bool(test.value)
        return False
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        return _call_is_exit_zero(stmt.exc)
    if isinstance(stmt, ast.Expr):
        value = stmt.value
        if isinstance(value, ast.Constant):
            # a bare literal or docstring
            return True
        return _call_is_exit_zero(value)
    return False


def _is_vacuous_python_body(body: str) -> bool:
    """A `python -c` body that cannot fail for a reason depending on the claim.

    Walks EVERY statement rather than bailing out when there is more than one:
    the earlier version returned False for any multi-statement body, so
    `import os; os._exit(0)` and `assert True` were both admitted.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    rest = [
        stmt
        for stmt in tree.body
        if not isinstance(stmt, (ast.Import, ast.ImportFrom))
    ]
    if not rest:
        return True
    return all(_stmt_is_vacuous(stmt) for stmt in rest)


def _is_vacuous_shell_body(body: str) -> bool:
    s = body.strip().rstrip(";").strip()
    s = re.sub(r"\s+", " ", s)
    if s in {"true", "false", ":", "exit", "exit 0", "true;", ":;"}:
        return True
    if s.startswith("exit 0"):
        return True
    return False



def _split_top_level(command: str, sep: str) -> List[str]:
    """Split on `sep` only outside single/double quotes.

    Naive `str.split` would cut inside `python3 -c "a || b"`, which is a quoted
    argument and not a shell combinator.
    """
    out: List[str] = []
    buf: List[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if command.startswith(sep, i):
            # `&&` must not be seen inside `&`-splitting and vice versa
            if sep == ";" or not command.startswith(sep * 2, i) or sep in ("&&", "||"):
                out.append("".join(buf))
                buf = []
                i += len(sep)
                continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [p.strip() for p in out if p.strip()]


def command_is_admissible(command: str) -> Tuple[bool, str]:
    """Refuse tautologies and anything whose first token is not in _COMMAND_FIRST.

    ``true``, ``:``, ``exit 0``, ``python3 -c 'raise SystemExit(0)'`` and
    ``sh -c true`` cannot fail for a reason that depends on the claim.
    """
    s = (command or "").strip()
    if not s:
        return False, "EMPTY_COMMAND"

    # Shell combinators first. `cmd || true` and `cmd ; true` launder ANY
    # failure into an exit 0, which is strictly worse than a tautology: the
    # check still runs, still fails, and the result is thrown away. Both were
    # admitted before this guard.
    for sep in ("||", ";"):
        parts = _split_top_level(s, sep)
        if len(parts) > 1:
            if sep == "||":
                return False, "SUCCESS_LAUNDERING"
            # `a ; b` exits with b's status, so b decides and must itself be a
            # real check.
            return command_is_admissible(parts[-1])
    parts = _split_top_level(s, "&&")
    if len(parts) > 1:
        # `a && b` fails if either fails, so it is admissible when any conjunct
        # is a real check -- but every conjunct must still be a known command.
        verdicts = [command_is_admissible(p) for p in parts]
        if all(not ok for ok, _ in verdicts):
            return verdicts[0]
        return True, ""

    tokens = _tokens(s)
    if not tokens:
        return False, "EMPTY_COMMAND"
    base = tokens[0].rsplit("/", 1)[-1].lower()
    if base in _VACUOUS_FIRST or tokens[0] == ":":
        return False, "VACUOUS_COMMAND"
    if base not in _COMMAND_FIRST:
        return False, "COMMAND_NOT_ADMITTED"
    if base in {"sh", "bash", "zsh"} and len(tokens) >= 3 and tokens[1] in {"-c", "-lc"}:
        if _is_vacuous_shell_body(tokens[2]):
            return False, "VACUOUS_COMMAND"
        # Recurse: `sh -c "python3 -c 'raise SystemExit(0)'"` is exactly as
        # vacuous as the command it wraps, and wrapping must not launder it.
        inner_ok, inner_why = command_is_admissible(tokens[2])
        if not inner_ok:
            return False, inner_why
    if base in {"python", "python3"} and "-c" in tokens:
        idx = tokens.index("-c")
        if idx + 1 < len(tokens) and _is_vacuous_python_body(tokens[idx + 1]):
            return False, "VACUOUS_COMMAND"
    return True, ""


def ast_has_tests(tree: ast.AST) -> bool:
    """True if the module defines pytest-idiom tests at top level.

    A ``__main__`` guard does not hide these: files with ``test_*`` functions
    or ``Test*`` classes must run under pytest, never as ``python file.py``.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            return True
    return False


def should_run_with_pytest(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return ast_has_tests(tree)


def count_static_assertions(tree: ast.AST) -> int:
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            n += 1
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and str(func.attr).startswith("assert"):
                n += 1
            elif isinstance(func, ast.Name) and str(func.id).startswith("assert"):
                n += 1
    return n


def _pytest_passed_count(output: str) -> int:
    match = re.search(r"(\d+)\s+passed", output or "")
    if not match:
        return 0
    return int(match.group(1))


def _run_script_counting_asserts(path: Path, timeout: float = 30.0) -> Tuple[int, int, str]:
    """Run ``path`` as __main__ in a child, counting executed assert lines."""
    runner = r"""
import ast, runpy, sys
from pathlib import Path
path = sys.argv[1]
src = Path(path).read_text(encoding="utf-8")
tree = ast.parse(src, filename=path)
assert_lines = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Assert) and getattr(node, "lineno", None):
        assert_lines.add(node.lineno)
    elif isinstance(node, ast.Call) and getattr(node, "lineno", None):
        func = node.func
        if isinstance(func, ast.Attribute) and str(func.attr).startswith("assert"):
            assert_lines.add(node.lineno)
        elif isinstance(func, ast.Name) and str(func.id).startswith("assert"):
            assert_lines.add(node.lineno)
executed = 0
resolved = Path(path).resolve()
def tracer(frame, event, arg):
    global executed
    if event != "line":
        return tracer
    try:
        fn = Path(frame.f_code.co_filename).resolve()
    except Exception:
        return tracer
    if fn == resolved and frame.f_lineno in assert_lines:
        executed += 1
    return tracer
sys.settrace(tracer)
rc = 0
try:
    runpy.run_path(path, run_name="__main__")
except SystemExit as exc:
    code = exc.code
    if code is None:
        rc = 0
    elif isinstance(code, int):
        rc = code
    else:
        rc = 1
except Exception:
    rc = 1
finally:
    sys.settrace(None)
sys.stdout.write("\n__HCLI_ASSERT_COUNT__=%d\n" % executed)
raise SystemExit(rc)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return -1, 0, f"TIMEOUT: {exc}"
    blob = (proc.stdout or "") + (proc.stderr or "")
    count = 0
    match = re.search(r"__HCLI_ASSERT_COUNT__=(\d+)", blob)
    if match:
        count = int(match.group(1))
    return int(proc.returncode), count, blob


def evaluate_python_test_file(
    path: Union[str, Path],
    *,
    run: bool = True,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Classify and optionally run a Python test file.

    Pytest-idiom files (``test_*`` / ``Test*``) are never run as scripts,
    even when they have a ``__main__`` guard. A file that executes zero
    assertions is ``NO_EVIDENCE``, not a pass.
    """
    dest = Path(path)
    try:
        source = dest.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(dest))
    except SyntaxError as exc:
        return {
            "ok": False,
            "reason": "SYNTAX_ERROR",
            "runner": None,
            "forced_pytest": False,
            "assertion_count": 0,
            "output": str(exc),
        }
    except OSError as exc:
        return {
            "ok": False,
            "reason": f"COMMAND_ERROR:{type(exc).__name__}",
            "runner": None,
            "forced_pytest": False,
            "assertion_count": 0,
            "output": str(exc),
        }

    has_tests = ast_has_tests(tree)
    n_assert = count_static_assertions(tree)
    force_pytest = has_tests

    if n_assert < 1:
        return {
            "ok": False,
            "reason": "NO_EVIDENCE",
            "runner": "pytest" if force_pytest else "script",
            "forced_pytest": force_pytest,
            "assertion_count": 0,
            "output": "zero assertions; not evidence",
        }

    if force_pytest:
        result: Dict[str, Any] = {
            "ok": False,
            "reason": None,
            "runner": "pytest",
            "forced_pytest": True,
            "assertion_count": n_assert,
            "output": "",
        }
        if not run:
            result["ok"] = None
            return result
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(dest),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                    "--color=no",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(dest.parent),
            )
        except subprocess.TimeoutExpired as exc:
            result["reason"] = "TIMEOUT"
            result["output"] = str(exc)
            return result
        except Exception as exc:
            result["reason"] = f"COMMAND_ERROR:{type(exc).__name__}"
            result["output"] = str(exc)
            return result
        output = (proc.stdout or "") + (proc.stderr or "")
        passed = _pytest_passed_count(output)
        result["output"] = output
        result["exit_code"] = int(proc.returncode)
        result["passed"] = passed
        if proc.returncode != 0:
            lowered = output.lower()
            if "failed" in lowered or "error" in lowered:
                result["reason"] = "TEST_FAILED"
            else:
                result["reason"] = "NO_EVIDENCE"
            return result
        if passed < 1:
            result["reason"] = "NO_EVIDENCE"
            return result
        result["ok"] = True
        return result

    result = {
        "ok": False,
        "reason": None,
        "runner": "script",
        "forced_pytest": False,
        "assertion_count": n_assert,
        "output": "",
    }
    if not run:
        result["ok"] = None
        return result
    rc, executed, output = _run_script_counting_asserts(dest, timeout=timeout)
    result["output"] = output
    result["exit_code"] = rc
    result["assertion_count"] = executed
    if rc == -1:
        result["reason"] = "TIMEOUT"
        return result
    if executed < 1:
        result["reason"] = "NO_EVIDENCE"
        return result
    if rc != 0:
        result["reason"] = "TEST_FAILED"
        return result
    result["ok"] = True
    return result
