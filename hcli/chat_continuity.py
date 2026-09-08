"""Crossing a context window, and crossing a process boundary.

TWO DIFFERENT BOUNDARIES, ONE RULE. A context window fills; a process exits.
Neither is the end of the objective. S035 s29/s31: a platform boundary is a
YIELD, and process lifetime is an implementation detail -- what must survive is
the objective, the plan and its step, the evidence, and the exact next action.

WHAT THIS DOES NOT DO. It does not summarise the conversation with a model.
Compaction here is EVICTION with a pointer: the turns leaving the working set
are written to the session's own archive and replaced by a compact statement of
where the work stands, which the durable plan and the evidence handles already
carry. A model-written summary would be a second, less reliable authority over
material that is sitting on disk in exact form.

PREFIX REUSE SURVIVES IT. The invariant leading region -- tool contract, repo
identity -- is untouched. Compaction removes turns from the MIDDLE and appends
the current position, so the prefix the resident reuses is the same prefix it
reused last turn. Rewriting the front of the prompt every turn would trade a
30x physical win for a semantic tidy-up.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .persist import atomic_write_json

#: Roughly four characters per token. Compaction is triggered on the estimate
#: rather than a tokenizer call, because the decision only needs to be
#: approximately right and a tokenizer round trip per turn is not free.
CHARS_PER_TOKEN = 4
#: Leave room for the answer. The body's own window is the ceiling; this is the
#: share of it the conversation may occupy before turns start leaving.
CONTEXT_SHARE = 0.55
KEEP_RECENT_TURNS = 6


def estimate_tokens(messages: Any) -> int:
    total = 0
    for message in (messages or []):
        if isinstance(message, dict):
            total += len(str(message.get("content") or "")) // CHARS_PER_TOKEN
    return total


@dataclass
class Compaction:
    """What happened, so a caller can report it instead of guessing."""

    compacted: bool
    before_tokens: int
    after_tokens: int
    evicted_turns: int
    archive: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def archive_turns(session: Any, turns: List[Dict[str, str]]) -> str:
    """Write evicted turns where they can be read back, exactly."""
    root = Path(session.workspace) / ".hcli" / "chat" / "archive"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{session.id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(json.dumps(
                {"at": time.time(), "role": turn.get("role"),
                 "content": turn.get("content")}, default=str) + "\n")
    return str(path)


def compact(messages: List[Dict[str, str]], session: Any, *,
            window: int, keep: int = KEEP_RECENT_TURNS) -> Tuple[List[Dict[str, str]], Compaction]:
    """Evict middle turns when the conversation outgrows its share of the window.

    The leading system material stays first and unchanged -- that is the prefix
    the resident reuses. The recent turns stay because they are the live thread.
    What goes is the middle, to the archive, replaced by one line saying so.
    """
    before = estimate_tokens(messages)
    budget = int(window * CONTEXT_SHARE)
    if before <= budget or len(messages) <= keep + 2:
        return messages, Compaction(False, before, before, 0)

    leading = [m for m in messages[:1] if m.get("role") == "system"]
    body = messages[len(leading):]
    recent = body[-keep:]
    middle = body[:-keep]
    if not middle:
        return messages, Compaction(False, before, before, 0)

    archive = archive_turns(session, middle)
    marker = {
        "role": "system",
        "content": (
            f"[{len(middle)} earlier turns of this conversation are archived at "
            f"{archive}. They are not lost: the objective, plan and evidence "
            f"handles above are what they established. Ask for a specific "
            f"detail and it can be read back.]"),
    }
    kept = [*leading, marker, *recent]
    after = estimate_tokens(kept)
    return kept, Compaction(True, before, after, len(middle), archive)


# --- crossing a process boundary -------------------------------------------

CHECKPOINT_SCHEMA = "hcli.chat.checkpoint.v1"


def checkpoint(session: Any, *, next_action: str = "",
               resident: str = "", note: str = "") -> Dict[str, Any]:
    """Everything needed to resume, in the shape CONTINUATION.json already uses.

    Deliberately the SAME field names the resident's own continuation record
    uses, so one reader can serve both and a durable objective does not depend
    on which surface wrote it.
    """
    plan = session.plan() if hasattr(session, "plan") else None
    record = {
        "schema": CHECKPOINT_SCHEMA,
        "session": session.id,
        "workspace": str(session.workspace),
        "objective": session.objective,
        "active_workunit": (f"{plan.id} step "
                            f"{min(plan.current_step + 1, max(len(plan.steps), 1))}"
                            if plan else ""),
        "hypothesis": (plan.title if plan else ""),
        "next_action": next_action or _next_step_text(plan),
        "evidence_refs": list(getattr(session, "evidence", []) or [])[-16:],
        "authority": getattr(session, "authority", "read"),
        "resident": resident or getattr(session, "resident", ""),
        "repo_commit": (plan.repo_commit if plan else ""),
        "note": note,
        "written_at": time.time(),
    }
    path = (Path(session.workspace) / ".hcli" / "chat"
            / f"{session.id}.checkpoint.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, record)
    session.last_checkpoint = str(path)
    return record


def _next_step_text(plan: Any) -> str:
    if plan is None or not getattr(plan, "steps", None):
        return ""
    index = min(plan.current_step, len(plan.steps) - 1)
    return plan.steps[index]


def resume(session: Any) -> Optional[Dict[str, Any]]:
    """The checkpoint for this session, verified against reality.

    A STALE NEXT ACTION IS NOT REPLAYED. If the repository moved since the
    checkpoint was written, the divergence is reported rather than acted on --
    S035 s30: do not blindly replay a stale next_action.
    """
    path = (Path(session.workspace) / ".hcli" / "chat"
            / f"{session.id}.checkpoint.json")
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    from .chat_state import _head_commit

    now = _head_commit(session.workspace)
    was = str(record.get("repo_commit") or "")
    record["repo_now"] = now
    record["diverged"] = bool(was and now and was != now)
    if record["diverged"]:
        record["divergence"] = (
            f"the plan was written against {was} and the repository is now at "
            f"{now}; re-inspect before continuing rather than replaying the "
            f"recorded next action")
    return record


def resume_block(record: Optional[Dict[str, Any]]) -> str:
    """What a resuming turn needs to see. Compact by construction."""
    if not record:
        return ""
    lines = ["Resuming this conversation from its checkpoint:"]
    for key in ("objective", "active_workunit", "next_action"):
        if record.get(key):
            lines.append(f"  {key.upper()} {record[key]}")
    if record.get("diverged"):
        lines.append(f"  DIVERGED {record['divergence']}")
    return "\n".join(lines)
