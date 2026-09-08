"""Durable identity, plans, and referents for the browser surface.

WHY THIS EXISTS. `hcli serve` was intellectually stateless: Open WebUI held the
conversation and resent it whole every turn, so HCLI had no session, no memory
of a plan, and no way to resolve "apply it". Everything needed to fix that was
already on disk and unwired -- SessionStore, KnowledgeStore, PasteCache, and the
CONTINUATION.json field set that `resident --from-checkpoint` already consumes.
This module is the record that ties them together, not a fourth store.

IDENTITY WITHOUT CLIENT COOPERATION. Open WebUI sends no session id. It does
send the whole conversation, and the FIRST user turn of a conversation never
changes -- so its digest is a stable key for as long as that conversation
exists, needs no protocol change, and cannot collide across repositories
because the workspace is folded in. A client that later sends a real id can
override it.

CONVERSATION IS AN INTERFACE, NOT MEMORY. The human may scroll weeks of chat.
That does not mean the body should be shown weeks of chat. What survives here is
what HCLI must still KNOW: the objective, the active plan and its step, the
referents "it"/"that plan" resolve to, the authority granted, and handles to
evidence. History stays on disk; this is the working set's index into it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .persist import atomic_write_json

SCHEMA = "hcli.chat.session.v1"
PLAN_SCHEMA = "hcli.chat.plan.v1"


def session_key(messages: Any, workspace: str, explicit: Optional[str] = None) -> str:
    """A stable id for this conversation.

    Derived from the first user turn because that is the one part of a
    conversation that never changes, and from the workspace so the same opening
    question in two repositories is two sessions.
    """
    if explicit:
        return str(explicit)[:64]
    first = ""
    for message in (messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            first = str(message.get("content") or "")
            break
    digest = hashlib.sha256(
        (os.path.realpath(workspace) + "\n" + first).encode("utf-8", "replace")
    ).hexdigest()
    return f"chat-{digest[:16]}"


@dataclass
class Plan:
    """A plan the human asked for, kept so it outlives the context window.

    The prose the human reads is authority and lives in `body`. `steps` and
    `current_step` are the compact execution view -- what S035 s5 calls
    compiling the plan into a working set, so a long build carries the step it
    is on rather than the whole document.
    """

    id: str
    title: str
    body: str
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    status: str = "draft"          # draft | approved | executing | done
    repo_commit: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": PLAN_SCHEMA, **self.__dict__}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Plan":
        fields = {k: v for k, v in (raw or {}).items()
                  if k in cls.__dataclass_fields__}
        return cls(**fields)

    def compiled(self) -> str:
        """The execution working set: goal, position, next step. Not the prose."""
        lines = [f"PLAN {self.id}  {self.title}", f"STATUS {self.status}"]
        if self.steps:
            lines.append(f"STEP {min(self.current_step + 1, len(self.steps))}"
                         f"/{len(self.steps)}")
            for index, step in enumerate(self.steps):
                mark = "x" if index < self.current_step else (
                    ">" if index == self.current_step else " ")
                lines.append(f"  [{mark}] {step}")
        if self.repo_commit:
            lines.append(f"BASIS {self.repo_commit}")
        return "\n".join(lines)


@dataclass
class ChatSession:
    """What HCLI must still know about this conversation."""

    id: str
    workspace: str
    resident: str = ""
    objective: str = ""
    authority: str = "read"        # read | write
    plans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_plan: str = ""
    referents: Dict[str, str] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    turns: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    last_checkpoint: str = ""

    # -- persistence ---------------------------------------------------------
    @staticmethod
    def path_for(workspace: str, session_id: str) -> Path:
        return (Path(workspace) / ".hcli" / "chat" / f"{session_id}.json")

    @classmethod
    def load(cls, workspace: str, session_id: str) -> "ChatSession":
        path = cls.path_for(workspace, session_id)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                fields = {k: v for k, v in raw.items()
                          if k in cls.__dataclass_fields__}
                return cls(**fields)
            except Exception:
                pass  # a corrupt record must not end the conversation
        now = time.time()
        return cls(id=session_id, workspace=str(workspace),
                   created_at=now, updated_at=now)

    def save(self) -> Path:
        self.updated_at = time.time()
        path = self.path_for(self.workspace, self.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema": SCHEMA, **self.__dict__})
        return path

    # -- plans ---------------------------------------------------------------
    def next_plan_id(self) -> str:
        return f"P{len(self.plans) + 1}"

    def add_plan(self, title: str, body: str, steps: Optional[List[str]] = None,
                 repo_commit: str = "") -> Plan:
        plan = Plan(id=self.next_plan_id(), title=title.strip()[:120], body=body,
                    steps=list(steps or []), repo_commit=repo_commit,
                    created_at=time.time(), updated_at=time.time())
        self.plans[plan.id] = plan.to_dict()
        self.active_plan = plan.id
        self.referents["it"] = plan.id
        self.referents["plan"] = plan.id
        return plan

    def plan(self, plan_id: Optional[str] = None) -> Optional[Plan]:
        key = plan_id or self.active_plan
        raw = self.plans.get(key or "")
        return Plan.from_dict(raw) if raw else None

    def update_plan(self, plan: Plan) -> None:
        plan.updated_at = time.time()
        self.plans[plan.id] = plan.to_dict()

    def resolve(self, phrase: str) -> "Resolution":
        """What "it" / "that plan" / "P2" points at.

        AMBIGUITY IS ANSWERED, NOT GUESSED. One candidate resolves silently --
        that is the whole point of natural reference. Several candidates return
        them all so the caller can ask, because picking one at random is how a
        builder edits the wrong thing.
        """
        text = (phrase or "").strip().lower()
        explicit = [pid for pid in self.plans if pid.lower() in text.split()]
        if len(explicit) == 1:
            return Resolution(explicit[0], self.plan(explicit[0]), [])
        if len(explicit) > 1:
            return Resolution("", None, explicit)
        titled = [pid for pid, raw in self.plans.items()
                  if str(raw.get("title", "")).lower() in text and raw.get("title")]
        if len(titled) == 1:
            return Resolution(titled[0], self.plan(titled[0]), [])
        if self.active_plan and self.active_plan in self.plans:
            return Resolution(self.active_plan, self.plan(self.active_plan), [])
        if len(self.plans) == 1:
            only = next(iter(self.plans))
            return Resolution(only, self.plan(only), [])
        return Resolution("", None, sorted(self.plans))


@dataclass
class Resolution:
    plan_id: str
    plan: Optional[Plan]
    candidates: List[str]

    @property
    def ok(self) -> bool:
        return self.plan is not None

    def question(self) -> str:
        """What to ask when a reference genuinely cannot be resolved."""
        if self.ok or not self.candidates:
            return ""
        return "Which one? " + ", ".join(self.candidates)


def working_set(session: ChatSession) -> str:
    """The durable part of the prompt: objective, plan position, authority.

    Deliberately compact and deliberately STABLE in its leading lines, because
    it sits in the prefix the native resident reuses. A session with nothing
    durable yet contributes nothing rather than an empty scaffold.
    """
    plan = session.plan()
    lines: List[str] = []
    if session.objective:
        lines.append(f"OBJECTIVE {session.objective}")
    if plan:
        lines.append(plan.compiled())
    if session.authority == "write":
        lines.append("AUTHORITY write (repo-scoped, typed operations only)")
    if not lines:
        return ""
    return ("Durable state for this conversation -- it survives the context "
            "window, so you do not need the transcript to know where you are:\n"
            + "\n".join(lines))


#: A reply is treated as a PLAN when the human asked for one and the answer has
#: the shape of one. Deliberately narrow: S035 s32 -- not every sentence
#: deserves permanent state, and memory should compound capability rather than
#: accumulate noise.
_PLAN_ASKED = ("plan", "roadmap", "design", "approach", "strategy", "steps")
_APPLY_ASKED = ("apply it", "apply that", "apply the plan", "go ahead",
                "do it", "implement it", "build it", "make it so",
                "start on it", "execute it", "proceed")


def asked_for_a_plan(text: str) -> bool:
    lowered = (text or "").lower()
    if not any(word in lowered for word in _PLAN_ASKED):
        return False
    return any(verb in lowered for verb in
               ("write", "give", "make", "draft", "propose", "come up",
                "outline", "think through", "sketch"))


def asked_to_apply(text: str) -> bool:
    """An instruction to execute, not a question about executing.

    The fixed phrase list missed "apply P1", which is unmistakably an
    instruction. So the bare verb counts too -- but only as a WORD and only
    when the sentence is not a question, so "what would applying it involve?"
    stays a question and does not start a build.
    """
    import re

    lowered = (text or "").lower().strip()
    if any(phrase in lowered for phrase in _APPLY_ASKED):
        return True
    if lowered.endswith("?"):
        return False
    # IMPERATIVE, not merely mentioned. "tell me about apply" contains the verb
    # and asks for nothing to happen; this flips a plan to executing, so the
    # verb has to lead the sentence or follow a discourse marker.
    lead = re.sub(r"^(ok|okay|now|then|yes|right|good|please|and|so|let'?s|go ahead(,| and)?)[\s,]+",
                  "", lowered)
    return bool(re.match(r"(apply|implement|execute)\b", lead))


def extract_steps(body: str, limit: int = 24) -> List[str]:
    """Ordered steps out of a plan's prose, so the working set can carry position.

    Numbered and bulleted forms only. A plan with no recognisable steps keeps
    its prose and simply has no step counter -- inventing steps out of
    paragraphs would put words in the plan's mouth.
    """
    import re

    steps: List[str] = []
    for line in (body or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:\d+[.)]|[-*•]|step\s+\d+[:.)]?)\s+(.{3,160})$",
                         stripped, re.I)
        if match:
            text = match.group(1).strip().rstrip(".")
            if text and text.lower() not in {s.lower() for s in steps}:
                steps.append(text)
        if len(steps) >= limit:
            break
    return steps


def _last_user(messages: Any) -> str:
    for message in reversed(list(messages or [])):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def remember_turn(session: ChatSession, messages: Any, answer: str,
                  knowledge: Any = None) -> Dict[str, Any]:
    """Promote what this turn established into durable state.

    Returns what it did, so a caller can report it rather than guess.
    """
    asked = _last_user(messages)
    outcome: Dict[str, Any] = {}

    if asked_for_a_plan(asked) and len(answer or "") > 200:
        steps = extract_steps(answer)
        title = asked.strip().rstrip("?.").split("\n")[0][:80]
        plan = session.add_plan(title, answer, steps=steps,
                                repo_commit=_head_commit(session.workspace))
        outcome["plan_created"] = plan.id
        if knowledge is not None:
            try:
                knowledge.record_note(
                    f"plan {plan.id}: {title} ({len(steps)} steps)")
            except Exception:
                pass

    if asked_to_apply(asked):
        resolution = session.resolve(asked)
        if resolution.ok and resolution.plan is not None:
            plan = resolution.plan
            if plan.status in ("draft", "approved"):
                plan.status = "executing"
                session.update_plan(plan)
                session.active_plan = plan.id
                session.objective = session.objective or plan.title
            outcome["plan_executing"] = plan.id
        elif resolution.candidates:
            outcome["ambiguous"] = resolution.candidates

    if not session.objective and asked:
        session.objective = asked.strip().split("\n")[0][:160]
    return outcome


def _head_commit(workspace: str) -> str:
    """The commit a plan was written against. A plan has a basis or it has none."""
    import subprocess
    try:
        done = subprocess.run(["git", "-C", str(workspace), "rev-parse",
                               "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        return done.stdout.strip() if done.returncode == 0 else ""
    except Exception:
        return ""
