"""Goal text compiler for HCLI.

``WorkUnitDAG`` is compiler IR: a small in-memory dependency graph that
``GoalCompiler`` uses to turn natural-language goal text into implement /
validate WorkUnits. It is a thin wrapper over the canonical WorkUnit
readiness and status-transition logic in ``workunit.py`` (``identify_ready``,
``transition_status``). Persistence uses ``dag_store.py``'s atomic writer.
It is not a second DAG implementation, and it is not the mission-level
obligation ledger (that is ``ledger.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .context_budget import (
    CHARS_PER_TOKEN,
    PacketBudgetError,
    preflight_packet,
)
from .workunit import WorkUnit, identify_ready, transition_status


class InvalidTransitionError(ValueError):
    """Raised when a WorkUnitDAG status change is refused by transition_status."""

    def __init__(self, unit_id: str, from_status: str, to_status: str) -> None:
        self.unit_id = unit_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"invalid WorkUnit transition {from_status!r} -> {to_status!r} "
            f"for {unit_id}"
        )


# Compiler-IR callers (GoalCompiler / engine tests) skip an explicit
# running step: get_ready_units() then mark_completed(). Walk the legal
# transition_status path rather than writing .status directly.
_ADVANCE = {
    ("pending", "completed"): ("ready", "running", "completed"),
    ("pending", "failed"): ("ready", "running", "failed"),
    ("ready", "completed"): ("running", "completed"),
    ("ready", "failed"): ("running", "failed"),
    ("running", "completed"): ("completed",),
    ("running", "failed"): ("failed",),
    ("failed", "completed"): ("ready", "running", "completed"),
}


class WorkUnitDAG:
    """Thin compiler-IR wrapper over workunit.py's canonical DAG logic.

    Readiness is ``identify_ready`` (repair-aware, retry-budget-aware).
    Completions and failures go through ``transition_status`` and raise
    ``InvalidTransitionError`` when the canonical state machine refuses.
    Durable identity is ``WorkUnit.id`` — the same field the scheduler
    and ``dag_store.py`` persist. Goal obligations (G001, …) live in
    ``ledger.py`` and are a different identity space.
    """

    def __init__(self) -> None:
        self.units: Dict[str, WorkUnit] = {}

    def add_unit(
        self,
        unit_id: str,
        description: str,
        dependencies: Optional[Iterable[str]] = None,
        *,
        role: str = "implementation",
        verifier: Optional[str] = None,
        resource_class: Optional[str] = None,
        preferred_backend: Optional[str] = None,
    ) -> WorkUnit:
        if not unit_id:
            raise ValueError("WorkUnit id must not be empty")

        if unit_id in self.units:
            raise ValueError(f"Duplicate WorkUnit id: {unit_id}")

        deps = list(dependencies or [])

        if unit_id in deps:
            raise ValueError("WorkUnit cannot depend on itself")

        wu = WorkUnit(
            id=unit_id,
            role=role,
            description=description,
            dependencies=deps,
            verifier=verifier or None,
            preferred_backend=preferred_backend or None,
            **({"resource_class": resource_class} if resource_class else {}),
        )

        self.units[unit_id] = wu

        return wu

    def _advance_to(self, wu: WorkUnit, target: str) -> None:
        start = wu.status
        steps = _ADVANCE.get((start, target))
        if not steps:
            raise InvalidTransitionError(wu.id, start, target)
        for step in steps:
            if not transition_status(wu, step):
                raise InvalidTransitionError(wu.id, wu.status, step)

    def mark_completed(self, unit_id: str) -> None:
        wu = self.units[unit_id]
        self._advance_to(wu, "completed")
        wu.assigned_runtime = None

    def mark_failed(self, unit_id: str) -> None:
        wu = self.units[unit_id]
        self._advance_to(wu, "failed")
        wu.assigned_runtime = None

    def get_ready_units(self) -> List[WorkUnit]:
        return identify_ready(self.units)

    def to_dict(self) -> Dict[str, Any]:
        return {
            unit_id: wu.to_dict()
            for unit_id, wu in self.units.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkUnitDAG":
        dag = cls()
        if not isinstance(data, dict):
            raise TypeError("WorkUnitDAG.from_dict expects a dict")
        units_blob: Any = data
        if isinstance(data.get("units"), dict) and (
            "version" in data or "id" not in data
        ):
            units_blob = data["units"]
        for uid, payload in units_blob.items():
            if isinstance(payload, WorkUnit):
                wu = payload
            else:
                wu = WorkUnit.from_dict(payload)
            if not wu.id:
                wu.id = str(uid)
            dag.units[str(uid)] = wu
        return dag

    def save(self, workspace: Union[str, Path]) -> None:
        """Persist via dag_store.py's atomic writer. Disk is authority."""
        from .dag_store import DagStore

        DagStore(workspace).save(self.units)

    @classmethod
    def from_workspace(
        cls,
        workspace: Union[str, Path],
        recover_running: bool = True,
    ) -> "WorkUnitDAG":
        from .dag_store import DagStore

        dag = cls()
        dag.units = DagStore(workspace).load(recover_running=recover_running)
        return dag


# Names an obligation marks as code: `backticked` or written as a call.
_SYMBOL_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_.]*)`"
    r"|\b([A-Za-z_][A-Za-z0-9_.]*)\s*\("
)
_EXTENSIONS = frozenset(
    "py pyi js jsx ts tsx md json yaml yml toml rs go txt".split()
)
_CODE_EXTENSIONS = frozenset("py pyi rs go ts tsx js jsx".split())


class GoalCompiler:
    """Compile a durable natural-language goal into focused deterministic IR.

    This compiler intentionally performs inexpensive structural extraction.
    Semantic model planning can enrich the resulting DAG later, but workers
    should not need the full durable goal merely to receive one bounded
    WorkUnit.
    """

    # The leading `/?` matters: without it an absolute path in the goal text
    # was captured with its root stripped, so a compiled verifier became
    # `pytest Users/scammermike/...` -- a relative path that does not exist.
    # The unit then failed for a reason that had nothing to do with its claim,
    # and the repair machinery dutifully retried the same broken command.
    _FILE_RE = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"(/?(?:[A-Za-z0-9_.-]+/)*"
        r"[A-Za-z0-9_.-]+\."
        r"(?:py|pyi|js|jsx|ts|tsx|md|json|yaml|yml|toml|rs|go|txt))"
    )

    _INVARIANT_MARKERS = (
        "do not",
        "don't",
        "must",
        "must not",
        "never",
        "only",
        "without",
        "preserve",
        "required",
    )

    _ACCEPTANCE_MARKERS = (
        "test",
        "tests",
        "pass",
        "validate",
        "validation",
        "acceptance",
        "works",
        "working",
    )

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    _RESTATEMENT_ACCEPTANCE = (
        "Requested behavior is implemented and validated."
    )

    def _sentences(self, text: str) -> List[str]:
        pieces = re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip(),
        )

        return [
            piece.strip()
            for piece in pieces
            if piece.strip()
        ]

    def _referenced_files(self, goal_text: str) -> List[str]:
        return list(
            dict.fromkeys(
                match.rstrip(".,:;)")
                for match in self._FILE_RE.findall(goal_text)
            )
        )

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        in_fence = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                if current is not None:
                    current["body"].append(line)
                continue
            if not in_fence:
                match = self._HEADING_RE.match(line)
                if match:
                    if current is not None:
                        sections.append(current)
                    current = {
                        "level": len(match.group(1)),
                        "title": match.group(2).strip(),
                        "body": [],
                    }
                    continue
            if current is not None:
                current["body"].append(line)
        if current is not None:
            sections.append(current)
        return sections

    def _is_validation_text(self, text: str) -> bool:
        lower = (text or "").lower()
        return any(
            token in lower
            for token in ("verify", "validat", "acceptance")
        )

    def _falsifiable_acceptance(
        self,
        title: str,
        body: str,
        files: List[str],
        claims: List[str],
    ) -> str:
        """Derive a check that can fail. Never copy the obligation text."""
        blob = f"{title}\n{body}"
        named = list(dict.fromkeys(self._referenced_files(blob)))[:4]
        if not named:
            named = [
                path
                for path in (files or [])
                if path and path in blob
            ][:4]
        pieces: List[str] = []
        if named:
            pieces.append(
                "named files exist and a non-tautological check of the "
                "claim fails when unmet: " + ", ".join(named)
            )
        claim = next(
            (
                item
                for item in claims
                if item.strip() and item.strip() != title.strip()
            ),
            None,
        )
        if claim:
            pieces.append(
                "the claim can be falsified by a command that exits "
                "non-zero when it does not hold"
            )
        if re.search(
            r"\b(test|pytest|cargo test|validate)\b",
            blob,
            re.I,
        ):
            pieces.append(
                "the named tests collect at least one case and fail if "
                "the claimed behavior is absent"
            )
        label = title.strip() or "this obligation"
        pieces.append(
            f"observable evidence exists that {label!r} is discharged; "
            "a command that cannot fail does not satisfy this"
        )
        acceptance = "; ".join(pieces)
        if acceptance.strip() in {
            title.strip(),
            self._RESTATEMENT_ACCEPTANCE,
        }:
            acceptance = (
                "an independent check of this claim can fail; restating "
                "the obligation is not acceptance"
            )
        return acceptance

    def _verify_command(self, title: str, body: str, files: List[str]) -> str:
        """Emit a command that can fail, or nothing. Never a tautology."""
        named = self._referenced_files(f"{title}\n{body}") or list(files or [])
        test_files = [
            path
            for path in named
            if re.search(r"test", path, re.I) and path.endswith(".py")
        ]
        if test_files:
            return f"python3 -m pytest {test_files[0]}"
        return self._synthesised_verify_command(f"{title}\n{body}", named)

    # A code-shaped name: has an underscore, mixes case, or is an all-caps
    # constant. Prose words ("works", "pass") fail all three, so `works (see
    # below)` never becomes a symbol to grep for.
    @staticmethod
    def _is_code_shaped(token: str) -> bool:
        if not token or token.lower() in _EXTENSIONS:
            return False
        if "_" in token:
            return True
        if token.isupper() and len(token) >= 3:
            return True
        return token != token.lower() and token != token.upper()

    def _claim_symbols(self, blob: str) -> List[str]:
        """Names the obligation asserts into existence, in first-seen order."""
        out: List[str] = []
        for backticked, called in _SYMBOL_RE.findall(blob):
            token = (backticked or called).strip(".")
            if not self._is_code_shaped(token.replace(".", "_")):
                continue
            for part in token.split("."):
                if self._is_code_shaped(part) and part not in out:
                    out.append(part)
            if len(out) >= 3:
                break
        return out[:3]

    def _synthesised_verify_command(self, blob: str, named: List[str]) -> str:
        """Derive a check from the claim when no test file is named.

        Binds to a DEFINITION of each claimed name, never to a mention: the
        obligation's own prose gets committed too, so `git grep NAME` alone
        would go green on the goal text that asked for NAME.

        Returns "" when the claim yields nothing checkable -- no code-shaped
        name, or no source file to scope the search to. That is deliberate:
        `Ledger.run_verify` refuses an empty command, so the obligation stays
        unVERIFIED. Two false greens are refused here rather than emitted:
        a check that only asserts a named file EXISTS goes green on an empty
        file, and an unscoped repo-wide search goes green because some
        unrelated file among 15k already defines a same-named symbol -- which
        `goal_compile.check_disk_satisfaction` would read as "already done"
        and skip the work entirely.
        """
        symbols = self._claim_symbols(blob)
        # Only paths this obligation itself names. `named` falls back to the
        # WHOLE goal's file list, which is how 20 units ended up gating on
        # whichever filename appeared first anywhere in the text.
        scope = [
            p
            for p in named
            if p in blob and p.rsplit(".", 1)[-1].lower() in _CODE_EXTENSIONS
        ]
        if not symbols or not scope:
            return ""
        pathspec = " ".join(f"'{p}'" for p in scope[:4])
        # ponytail: git grep sees tracked files only, so an untracked new file
        # reads as absent. Fine here (HCLI lands its work as commits); switch to
        # a walk if untracked worktrees ever need to verify.
        return " && ".join(
            "git grep -qE -- "
            f"'(def|class)[[:space:]]+{sym}[^A-Za-z0-9_]"
            f"|^[[:space:]]*{sym}[[:space:]]*=[^=]' -- {pathspec}"
            for sym in symbols
        )

    def _obligation_from_section(
        self,
        section: Dict[str, Any],
        files: List[str],
        oid: str,
    ) -> Dict[str, Any]:
        title = str(section.get("title") or "").strip()
        body = "\n".join(section.get("body") or [])
        body_files = self._referenced_files(f"{title}\n{body}") or list(files)
        claims = [
            sentence
            for sentence in self._sentences(body)
            if any(
                marker in sentence.lower()
                for marker in self._ACCEPTANCE_MARKERS
                + self._INVARIANT_MARKERS
            )
        ]
        kind = (
            "phase"
            if re.match(r"phase\s+[a-z0-9]", title, re.I)
            else "section"
        )
        role = (
            "validation"
            if self._is_validation_text(f"{title}\n{body}")
            else "implementation"
        )
        return {
            "id": oid,
            "text": title,
            "acceptance": self._falsifiable_acceptance(
                title, body, body_files, claims
            ),
            "verify": self._verify_command(title, body, body_files),
            "role": role,
            "kind": kind,
        }

    def _obligation_from_sentence(
        self,
        sentence: str,
        files: List[str],
        oid: str,
    ) -> Dict[str, Any]:
        text = sentence.strip()
        named = self._referenced_files(text) or list(files)
        role = (
            "validation"
            if self._is_validation_text(text)
            else "implementation"
        )
        return {
            "id": oid,
            "text": text,
            "acceptance": self._falsifiable_acceptance(
                text, "", named, []
            ),
            "verify": self._verify_command(text, "", named),
            "role": role,
            "kind": "claim",
        }

    def _extract_obligations(
        self,
        text: str,
        sentences: List[str],
        files: List[str],
    ) -> List[Dict[str, Any]]:
        sections = self._extract_sections(text)
        heading_obs: List[Dict[str, Any]] = []
        skipped_title = False
        for section in sections:
            title = str(section.get("title") or "").strip()
            if not title:
                continue
            if section.get("level") == 1 and not skipped_title:
                skipped_title = True
                continue
            oid = f"G{len(heading_obs) + 1:03d}"
            heading_obs.append(
                self._obligation_from_section(section, files, oid)
            )
        if len(heading_obs) > 2:
            return heading_obs

        sentence_obs: List[Dict[str, Any]] = []
        seen = set()
        for sentence in sentences:
            lower = sentence.lower()
            if not any(
                marker in lower
                for marker in self._ACCEPTANCE_MARKERS
                + self._INVARIANT_MARKERS
            ):
                continue
            key = re.sub(r"\s+", " ", sentence.strip().lower())
            if len(key) < 8 or key in seen:
                continue
            seen.add(key)
            oid = f"G{len(sentence_obs) + 1:03d}"
            sentence_obs.append(
                self._obligation_from_sentence(sentence, files, oid)
            )
        if len(sentence_obs) > 2:
            return sentence_obs
        return heading_obs or sentence_obs

    def _cover_tag(self, obligations: List[Dict[str, Any]]) -> str:
        ids = [str(ob.get("id") or "") for ob in obligations if ob.get("id")]
        if not ids:
            ids = ["G001"]
        return "obligations=" + ",".join(ids)

    def _build_workunit_dag(
        self,
        obligations: List[Dict[str, Any]],
        primary: str,
        files: List[str],
    ) -> WorkUnitDAG:
        dag = WorkUnitDAG()
        if len(obligations) <= 2:
            description = primary
            if files:
                description += " Relevant files: " + ", ".join(files[:6])
            cover = self._cover_tag(obligations)
            dag.add_unit(
                "implement",
                f"{cover} {description}",
                [],
                role="implementation",
                # Cognition opens the 11 GB resident. LIGHT_CONTROL admits 128
                # of them; GPU_DECODE admits one, which is what "one resident"
                # means. Same defect as the obligation path, same fix.
                resource_class="GPU_DECODE",
            )
            verifies = [
                str(ob.get("verify") or "").strip()
                for ob in obligations
                if str(ob.get("verify") or "").strip()
            ]
            # Several obligations, one validation unit: it must run all of their
            # checks, and `&&` means any one failing fails the unit.
            combined = " && ".join(dict.fromkeys(verifies))
            dag.add_unit(
                "validate",
                f"{cover} Validate the implementation against the "
                "active acceptance criteria.",
                ["implement"],
                role="validation",
                verifier=combined or None,
                resource_class="TEST" if combined else None,
                preferred_backend="cpu" if combined else None,
            )
            return dag

        last_phase: Optional[str] = None
        for ob in obligations:
            oid = str(ob["id"])
            deps: List[str] = []
            if ob.get("kind") == "phase" and last_phase:
                deps = [last_phase]
            # The obligation already carries a real, failable verify command
            # derived from the goal text. Without copying it onto the unit that
            # discharges the obligation, the scheduler dispatches work it has no
            # way to verify -- the obligation knows how to check itself and the
            # WorkUnit does not, which breaks the chain at its last link.
            ob_verify = str(ob.get("verify") or "").strip()
            dag.add_unit(
                oid,
                f"obligation={oid} {ob['text']}",
                deps,
                role=str(ob.get("role") or "implementation"),
                verifier=ob_verify or None,
                # A unit with no verify command is a COGNITION unit: it will
                # open the resident. `WorkUnit.resource_class` defaults to
                # LIGHT_CONTROL, whose limit is 128, so a goal with eight
                # obligations dispatched eight cognition units at once and each
                # opened its own 11 GB model body -- 88 GB on a 96 GB host.
                # Reproduced twice: free RAM fell to 0.2 GB and the supervisor
                # correctly refused with WAITING_FOR_MEMORY. GPU_DECODE has a
                # limit of 1, which is what "one resident" actually means.
                resource_class="TEST" if ob_verify else "GPU_DECODE",
                # A unit whose acceptance IS a command belongs on the backend
                # that runs commands. Left to default cognition it was routed to
                # the model, which then tried to CREATE the very test file the
                # obligation names -- work nobody asked for, failing for a
                # reason unrelated to the claim.
                preferred_backend="cpu" if ob_verify else None,
            )
            if ob_verify:
                # An obligation that carries a verify command needs TWO units,
                # not one. Emitting only the TEST unit produced a mission that
                # could RUN its gates and never implement anything -- the exact
                # mirror of the cognition-only failure, and just as useless:
                # every unit ran pytest on a red gate, failed in seconds, and
                # repaired. `_default_dag` already has the right shape
                # (implement -> validate); the obligation path lacked it.
                #
                # The unit added above is the GATE (TEST, cpu backend). This is
                # the work that makes the gate green: cognition, GPU_DECODE so
                # the scheduler admits exactly one resident at a time, and the
                # gate depends on it.
                work_id = f"{oid}.work"
                dag.add_unit(
                    work_id,
                    f"obligation={oid} {ob['text']}",
                    deps,
                    role=str(ob.get("role") or "implementation"),
                    resource_class="GPU_DECODE",
                )
                unit = dag.units.get(oid)
                if unit is not None:
                    existing = [d for d in (unit.dependencies or []) if d != work_id]
                    unit.dependencies = existing + [work_id]
            if ob.get("kind") == "phase":
                last_phase = oid
        return dag

    def compile(self, goal_text: str) -> Dict[str, Any]:
        text = (goal_text or "").strip()

        if not text:
            raise ValueError("Goal text must not be empty")

        sentences = self._sentences(text)

        invariants = [
            sentence
            for sentence in sentences
            if any(
                marker in sentence.lower()
                for marker in self._INVARIANT_MARKERS
            )
        ]

        files = self._referenced_files(text)

        primary = None

        for sentence in sentences:
            lower = sentence.lower()

            if any(
                marker in lower
                for marker in self._INVARIANT_MARKERS
            ):
                continue

            primary = sentence
            break

        if primary is None:
            primary = sentences[0]

        if len(primary) > 240:
            primary = primary[:237].rstrip() + "..."

        obligations = self._extract_obligations(text, sentences, files)
        acceptance = [
            str(ob.get("acceptance") or "").strip()
            for ob in obligations
            if str(ob.get("acceptance") or "").strip()
        ]
        if not acceptance:
            acceptance = [
                "an independent check of the requested behavior can fail; "
                "restating the request is not acceptance"
            ]

        dag = self._build_workunit_dag(obligations, primary, files)

        return {
            "goal": text,
            "goal_summary": primary,
            "invariants": invariants,
            "acceptance_criteria": acceptance,
            "referenced_files": files,
            "obligations": obligations,
            "workunits": dag,
        }

    def build_focused_context(
        self,
        workunit: WorkUnit,
        compiled_goal: Dict[str, Any],
    ) -> str:
        sections = [
            f"WORKUNIT: {workunit.id}",
            f"OBJECTIVE: {workunit.description}",
        ]

        invariants = list(
            compiled_goal.get("invariants", [])
        )[:4]

        if invariants:
            sections.append(
                "INVARIANTS:\n- "
                + "\n- ".join(invariants)
            )

        acceptance = list(
            compiled_goal.get(
                "acceptance_criteria",
                [],
            )
        )[:4]

        if acceptance:
            sections.append(
                "ACCEPTANCE:\n- "
                + "\n- ".join(acceptance)
            )

        files = list(
            compiled_goal.get(
                "referenced_files",
                [],
            )
        )[:8]

        if files:
            sections.append(
                "FILES:\n- "
                + "\n- ".join(files)
            )

        # This is deliberately a focused slice, not the complete durable goal.
        focused = "\n\n".join(sections)

        # Keep deterministic focused context bounded even when an invariant
        # sentence itself is unexpectedly enormous.
        if len(focused) > 1200:
            focused = focused[:1197].rstrip() + "..."

        return focused


PACKET_CHAR_CAP = 4000
MAX_COMPILED_INVARIANTS = 4
MAX_COMPILED_ACCEPTANCE = 4
# ponytail: fixed ceiling, not "most relevant N" — good enough to stop a
# repeat of a same-role dead end without turning NEIGHBORHOOD into an archive.
MAX_SCAR_LINES = 3

_DOCTRINE_SNIPPETS = (
    "MODELS THINK.",
    "TOOLS KNOW.",
    "CONTEXT IS A CACHE.",
    "DISK STATE IS AUTHORITY.",
)

_COMPILED_IR_KEYS = (
    "goal_summary",
    "invariants",
    "acceptance_criteria",
    "referenced_files",
)


# Uppercase GOAL plus optional space before the colon. Case-sensitive on
# purpose: `/ultragoal:` must not match. The engine dump is `GOAL:\n`.
_GOAL_DUMP_RE = re.compile(r"GOAL\s*:")
_DUMP_BLOCK_RE = re.compile(
    r"GOAL\s*:.*?(?="
    r"\n(?:PHASE|WORKUNIT|ROLE|OBJECTIVE|INVARIANTS|ACCEPTANCE|"
    r"EVIDENCE_PATHS|NEIGHBORHOOD|STEERING|ROOT_REF|TRUNCATION|"
    r"FAILURE_CONTEXT|DETERMINISTIC EVIDENCE):"
    r"|\n\n"
    r"|\Z)",
    re.DOTALL,
)
_DUMP_JSON_RE = re.compile(r"GOAL\s*:.*?(?=\"[,}\]])")

ROOT_GOAL_OMITTED = "[ROOT_GOAL_OMITTED]"
GOAL_DUMP_REDACTED = "[GOAL_DUMP_REDACTED]"
# Short goals ARE the unit. Only excise a body long enough to be a dump.
MIN_ROOT_EXCISE = 80


def contains_goal_dump(text: str) -> bool:
    """True when the engine/Mission ``GOAL:`` dump pattern is present.

    ``GOAL :`` (space before the colon) is the variant the old
    ``"GOAL:" in prompt`` check accepted.
    """
    return bool(_GOAL_DUMP_RE.search(str(text or "")))


def _excise_root_goal(text: str, root_goal: str) -> str:
    raw = str(text or "")
    root = str(root_goal or "").strip()
    if not raw or not root or len(root) < MIN_ROOT_EXCISE:
        return raw
    if root in raw:
        return raw.replace(root, ROOT_GOAL_OMITTED)
    return raw


def _redact_goal_dump(text: str) -> str:
    """Drop ``GOAL:`` headers AND the payload they introduce.

    The old rewrite (``GOAL:`` → ``GOAL_``) left the root-goal body in the
    packet, which is the leak WorkerPacket's substring check could not see.
    """
    raw = str(text or "")
    if not contains_goal_dump(raw):
        return raw
    raw = _DUMP_JSON_RE.sub(GOAL_DUMP_REDACTED, raw)
    raw = _DUMP_BLOCK_RE.sub(GOAL_DUMP_REDACTED, raw)
    raw = _GOAL_DUMP_RE.sub(GOAL_DUMP_REDACTED, raw)
    return raw


def _sanitize_goal_header(text: str, root_goal: str = "") -> str:
    """Stop the ``GOAL:`` dump pattern from leaking into a packet."""
    raw = _excise_root_goal(str(text or ""), root_goal)
    if contains_goal_dump(raw):
        raw = _redact_goal_dump(raw)
    return raw


def refuse_goal_dump(prompt: str, root_goal: str = "") -> None:
    """Raise if a worker prompt still carries the dump header or the root body."""
    text = str(prompt or "")
    if contains_goal_dump(text) or "GOAL:" in text:
        raise ValueError("worker packet must not contain GOAL:")
    root = str(root_goal or "").strip()
    if root and len(root) >= MIN_ROOT_EXCISE and root in text:
        raise ValueError("worker packet must not contain the root goal")


@dataclass(frozen=True)
class EvidenceIdentity:
    """Snapshot of one evidence file at gather time."""

    path: str
    size: int
    mtime_ns: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": int(self.size),
            "mtime_ns": int(self.mtime_ns),
            "sha256": self.sha256,
        }


class StaleEvidenceError(ValueError):
    """Raised when a file changed between gather and use. Not a silent re-read."""


def identity_for_path(
    path: Union[str, Path], *, root: Optional[Union[str, Path]] = None
) -> EvidenceIdentity:
    path = Path(path)
    st = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if root is not None:
        try:
            rel = path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
    else:
        rel = path.as_posix()
    return EvidenceIdentity(
        path=rel,
        size=int(st.st_size),
        mtime_ns=int(st.st_mtime_ns),
        sha256=digest,
    )


def _coerce_identity(item: Any) -> Optional[EvidenceIdentity]:
    if isinstance(item, EvidenceIdentity):
        return item
    if isinstance(item, dict) and item.get("path"):
        try:
            return EvidenceIdentity(
                path=str(item["path"]),
                size=int(item["size"]),
                mtime_ns=int(item["mtime_ns"]),
                sha256=str(item["sha256"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def assert_evidence_fresh(
    items: Sequence[Any],
    workspace: Union[str, Path],
) -> List[EvidenceIdentity]:
    """Refuse stale evidence. Never silently re-read a mutated file."""
    root = Path(workspace)
    verified: List[EvidenceIdentity] = []
    for item in items or ():
        ident = _coerce_identity(item)
        if ident is None:
            raise StaleEvidenceError(
                f"stale evidence refused: missing identity on {item!r}"
            )
        path = Path(ident.path)
        if not path.is_absolute():
            path = root / ident.path
        if not path.is_file():
            raise StaleEvidenceError(
                f"stale evidence refused: {ident.path} vanished"
            )
        current = identity_for_path(path, root=root)
        if (
            current.size != ident.size
            or current.mtime_ns != ident.mtime_ns
            or current.sha256 != ident.sha256
        ):
            raise StaleEvidenceError(
                f"stale evidence refused: {ident.path} "
                f"gathered size={ident.size} mtime_ns={ident.mtime_ns} "
                f"sha256={ident.sha256}; "
                f"now size={current.size} mtime_ns={current.mtime_ns} "
                f"sha256={current.sha256}"
            )
        verified.append(current)
    return verified


def assert_packet_evidence_fresh(
    packet: "WorkerPacket",
    workspace: Union[str, Path],
) -> List[EvidenceIdentity]:
    return assert_evidence_fresh(packet.evidence, workspace)


def _without_doctrine(items: Iterable[str], root_goal: str = "") -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if any(snippet in text for snippet in _DOCTRINE_SNIPPETS):
            continue
        out.append(_sanitize_goal_header(text, root_goal))
    return out


def compiled_ir_to_jsonable(compiled: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Persist compiler IR without the full root goal text."""
    if not isinstance(compiled, dict):
        return {}
    out: Dict[str, Any] = {}
    summary = compiled.get("goal_summary")
    if isinstance(summary, str) and summary:
        out["goal_summary"] = summary
    for key in ("invariants", "acceptance_criteria", "referenced_files"):
        value = compiled.get(key)
        if isinstance(value, (list, tuple)):
            out[key] = [str(item) for item in value]
    return out


def compiled_ir_from_jsonable(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if not any(key in data for key in _COMPILED_IR_KEYS):
        return None
    return {
        "goal_summary": str(data.get("goal_summary") or ""),
        "invariants": [str(item) for item in (data.get("invariants") or [])],
        "acceptance_criteria": [
            str(item) for item in (data.get("acceptance_criteria") or [])
        ],
        "referenced_files": [
            str(item) for item in (data.get("referenced_files") or [])
        ],
    }


def _normalize_steering(steering: Optional[Iterable[Any]]) -> List[Tuple[str, str]]:
    kept: List[Tuple[str, str]] = []
    for item in steering or []:
        if hasattr(item, "kind") and hasattr(item, "text"):
            kind = str(getattr(item, "kind", "knowledge") or "knowledge")
            text = str(getattr(item, "text", "") or "")
            if kind not in ("constraint", "correction", "knowledge"):
                continue
        else:
            text = str(item or "")
            kind = "constraint"
            stripped = text.strip()
            if stripped.startswith("[constraint]"):
                kind = "constraint"
                text = stripped[len("[constraint]"):].strip()
            elif stripped.startswith("[correction]"):
                kind = "correction"
                text = stripped[len("[correction]"):].strip()
        text = _sanitize_goal_header(text.strip())
        if text:
            kept.append((kind, text))
    return kept


def _ledger_constraint_lines(ledger: Any) -> List[str]:
    if ledger is None:
        return []
    # Never read ledger._preamble — on /ultragoal it is the full parent text.
    getter = getattr(ledger, "obligations", None)
    if not callable(getter):
        return []
    lines: List[str] = []
    try:
        obligations = getter()
    except Exception:
        return []
    for ob in obligations or []:
        oid = str(getattr(ob, "id", "") or "")
        text = str(getattr(ob, "text", "") or "").strip()
        if not text:
            continue
        label = f"{oid}: {text}" if oid else text
        lines.append(_sanitize_goal_header(label))
    return lines


def _mentioned_and_known_files(
    wu: WorkUnit,
    compiled_goal: Dict[str, Any],
    failure_context: Any,
) -> Tuple[str, ...]:
    compiler = GoalCompiler()
    known = [str(path) for path in (compiled_goal.get("referenced_files") or [])]
    texts = [wu.description or ""]
    if failure_context:
        try:
            texts.append(json.dumps(failure_context, default=str, sort_keys=True))
        except TypeError:
            texts.append(str(failure_context))
    blob = "\n".join(texts)
    mentioned = compiler._referenced_files(blob)
    hits: List[str] = []
    if known:
        known_set = set(known)
        for path in mentioned:
            if path in known_set:
                hits.append(path)
        for path in known:
            name = path.rsplit("/", 1)[-1]
            if path in blob or (name and name in blob):
                hits.append(path)
        return tuple(dict.fromkeys(hits))
    return tuple(dict.fromkeys(mentioned))


def _acceptance_for_unit(
    wu: WorkUnit,
    compiled_goal: Dict[str, Any],
    ledger: Any,
) -> List[str]:
    criteria = _without_doctrine(compiled_goal.get("acceptance_criteria") or [])
    ident = (wu.id or "").lower()
    role = (wu.role or "").lower()
    matched = [
        item
        for item in criteria
        if (ident and ident in item.lower()) or (role and role in item.lower())
    ]
    if matched:
        chosen = matched[:MAX_COMPILED_ACCEPTANCE]
    elif criteria:
        chosen = criteria[:MAX_COMPILED_ACCEPTANCE]
    else:
        chosen = ["Requested behavior is implemented and validated."]
    if ledger is not None:
        getter = getattr(ledger, "obligations", None)
        if callable(getter):
            try:
                for ob in getter() or []:
                    oid = str(getattr(ob, "id", "") or "")
                    if oid and oid != wu.id and wu.id not in str(getattr(ob, "text", "")):
                        continue
                    acceptance = str(getattr(ob, "acceptance", "") or "").strip()
                    if acceptance:
                        chosen.append(_sanitize_goal_header(acceptance))
            except Exception:
                pass
    return list(dict.fromkeys(chosen))


def _failure_why(context: Any) -> str:
    """Best short causal string out of a WorkUnit's failure_context.

    Mirrors the fields `workunit.failure_signature()` hashes (reason, error,
    validation) — that triple is this codebase's established "what actually
    went wrong" record, so this surfaces the same signal as readable text
    instead of inventing a second causal schema.
    """
    if not isinstance(context, dict):
        return ""
    validation = context.get("validation")
    if isinstance(validation, dict):
        reason = validation.get("reason")
        if reason:
            return str(reason)
    reason = context.get("reason")
    if reason:
        return str(reason)
    error = context.get("error")
    if error:
        return str(error)
    return ""


def _neighborhood_lines(
    wu: WorkUnit,
    units: Optional[Dict[str, WorkUnit]],
) -> List[str]:
    lines: List[str] = []
    pool = units or {}
    dep_ids = set(wu.dependencies or [])
    for dep_id in wu.dependencies or []:
        dep = pool.get(dep_id)
        if dep is None:
            lines.append(f"{dep_id} status=missing receipt=(none)")
            continue
        receipt = "(none)"
        context = getattr(dep, "failure_context", None) or {}
        if isinstance(context, dict):
            for key in ("receipt", "receipt_id", "goal_id"):
                value = context.get(key)
                if value:
                    receipt = _sanitize_goal_header(str(value))
                    break
        status = getattr(dep, "status", "unknown") or "unknown"
        line = f"{dep_id} status={status} receipt={receipt}"
        if status == "failed":
            why = _failure_why(context)
            if why:
                line += f" why={_sanitize_goal_header(why[:120])}"
        lines.append(line)

    # Scars: same-role dead ends elsewhere in this mission that are NOT a
    # direct dependency edge, so a repeat of the same mistake is visible even
    # across DAG-sibling boundaries. Bounded (MAX_SCAR_LINES) on purpose —
    # this is a pointer to the worst-case recent failures of this kind, not
    # an archive of every failure the mission has ever produced.
    scar_count = 0
    for other_id, other in pool.items():
        if scar_count >= MAX_SCAR_LINES:
            break
        if other_id == wu.id or other_id in dep_ids:
            continue
        if getattr(other, "status", None) != "failed":
            continue
        if str(getattr(other, "role", "") or "") != str(wu.role or ""):
            continue
        context = getattr(other, "failure_context", None) or {}
        sig = context.get("failure_signature") if isinstance(context, dict) else None
        line = f"{other_id} status=failed"
        if sig:
            line += f" sig={sig}"
        why = _failure_why(context)
        if why:
            line += f" why={_sanitize_goal_header(why[:120])}"
        if not sig and not why:
            # Nothing causal to say beyond "it failed" — not worth a slot.
            continue
        lines.append(line)
        scar_count += 1
    return lines


def _bullet_section(title: str, lines: Sequence[str]) -> str:
    if not lines:
        return f"{title}:\n(none)"
    return title + ":\n" + "\n".join(f"- {line}" for line in lines)


def _workunit_block(
    wu: WorkUnit,
    description: str,
    failure_context: Any,
    root_goal: str = "",
) -> str:
    lines = [
        f"WORKUNIT: {wu.id}",
        f"ROLE: {wu.role}",
        f"OBJECTIVE: {_sanitize_goal_header(description, root_goal)}",
    ]
    if failure_context:
        try:
            blob = json.dumps(failure_context, default=str, sort_keys=True)
        except TypeError:
            blob = str(failure_context)
        blob = _sanitize_goal_header(blob, root_goal)
        if len(blob) > 400:
            blob = blob[:397].rstrip() + "..."
        lines.append(f"FAILURE_CONTEXT: {blob}")
    return "\n".join(lines)


def _assemble_prompt(
    *,
    phase: str,
    workunit_block: str,
    invariants: Sequence[str],
    acceptance: Sequence[str],
    evidence_paths: Sequence[str],
    neighborhood: Sequence[str],
    steering: Sequence[str],
    goal_ref: str,
    neighborhood_mode: str,
    omitted: Sequence[str] = (),
    root_goal: str = "",
) -> str:
    if neighborhood_mode == "full":
        neighborhood_section = _bullet_section("NEIGHBORHOOD", neighborhood)
    elif neighborhood_mode == "ids":
        ids = []
        for line in neighborhood:
            token = str(line).split()[0] if line else ""
            if token:
                ids.append(token)
        neighborhood_section = _bullet_section("NEIGHBORHOOD", ids)
    else:
        neighborhood_section = "NEIGHBORHOOD:\n(none)"
    parts = [
        f"PHASE: {phase or '(none)'}",
        workunit_block,
        _bullet_section("INVARIANTS", invariants),
        _bullet_section("ACCEPTANCE", acceptance),
        _bullet_section("EVIDENCE_PATHS", evidence_paths),
        neighborhood_section,
        _bullet_section("STEERING", steering),
    ]
    if omitted:
        parts.append(
            "TRUNCATION: omitted="
            + ",".join(dict.fromkeys(str(item) for item in omitted if item))
        )
    if goal_ref:
        parts.append(f"ROOT_REF: {_sanitize_goal_header(goal_ref, root_goal)}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class WorkerPacket:
    """The only user text a worker model is allowed to see.

    Does not store the full root goal. A worker that needs the parent can
    follow ROOT_REF on disk; it never receives the parent inlined.
    """

    unit_id: str
    phase: str
    workunit: str
    invariants: tuple
    acceptance: tuple
    neighborhood: tuple
    evidence_paths: tuple
    steering: tuple
    prompt: str
    truncated: bool = False
    omitted: tuple = ()
    evidence: tuple = ()
    budget_signal: str = ""

    def __post_init__(self) -> None:
        refuse_goal_dump(self.prompt)


def _stamp_packet_evidence(
    evidence_paths: Sequence[str],
    *,
    workspace: Optional[Union[str, Path]] = None,
    evidence: Optional[Sequence[Any]] = None,
) -> List[EvidenceIdentity]:
    identities: List[EvidenceIdentity] = []
    seen = set()
    for item in evidence or ():
        ident = _coerce_identity(item)
        if ident is None or ident.path in seen:
            continue
        identities.append(ident)
        seen.add(ident.path)
    kept_paths = {str(path) for path in evidence_paths}
    kept_names = {Path(path).name for path in kept_paths}
    identities = [
        ident
        for ident in identities
        if ident.path in kept_paths or Path(ident.path).name in kept_names
    ]
    seen = {ident.path for ident in identities}
    ws = Path(workspace) if workspace else None
    if ws is None:
        return identities
    for rel in evidence_paths:
        token = str(rel)
        if token in seen or Path(token).name in {Path(p).name for p in seen}:
            continue
        path = Path(token)
        if not path.is_absolute():
            path = ws / token
        if not path.is_file():
            continue
        ident = identity_for_path(path, root=ws)
        identities.append(ident)
        seen.add(ident.path)
    return identities


def compile_worker_context(
    wu: WorkUnit,
    compiled_goal: dict,
    *,
    phase: str,
    units: Dict[str, WorkUnit],
    steering: list,
    failure_context: Optional[dict] = None,
    ledger: Any = None,
    goal_ref: str = "",
    char_cap: int = PACKET_CHAR_CAP,
    root_goal: str = "",
    workspace: Optional[Union[str, Path]] = None,
    evidence: Optional[Sequence[Any]] = None,
    budget: Any = None,
) -> WorkerPacket:
    """Compile the six-field worker packet. Never inlines the root goal."""
    compiled = compiled_goal if isinstance(compiled_goal, dict) else {}
    fc = failure_context if failure_context is not None else wu.failure_context
    if not root_goal:
        root_goal = str(compiled.get("goal") or "")

    steer_pairs = [
        (kind, _sanitize_goal_header(text, root_goal))
        for kind, text in _normalize_steering(steering)
    ]
    constraint_texts = [text for kind, text in steer_pairs if kind == "constraint"]
    steer_lines = [f"[{kind}] {text}" for kind, text in steer_pairs]
    ledger_lines = [
        _sanitize_goal_header(line, root_goal)
        for line in _ledger_constraint_lines(ledger)
    ]
    compiled_invariants = _without_doctrine(
        compiled.get("invariants") or [], root_goal
    )

    pinned = list(dict.fromkeys(constraint_texts + ledger_lines))
    # Seed from build_focused_context: a handful of compiler invariants, never
    # the whole ultragoal. Pinned constraint/ledger lines always come first.
    seeded: List[str] = list(pinned)
    for item in compiled_invariants:
        if item in seeded:
            continue
        seeded.append(item)
        compiled_n = len(seeded) - len(pinned)
        if compiled_n >= MAX_COMPILED_INVARIANTS:
            break
    invariants = seeded
    acceptance = [
        _sanitize_goal_header(item, root_goal)
        for item in _acceptance_for_unit(wu, compiled, ledger)
    ]
    evidence_paths = list(_mentioned_and_known_files(wu, compiled, fc))
    for item in evidence or ():
        ident = _coerce_identity(item)
        if ident is not None and ident.path not in evidence_paths:
            evidence_paths.append(ident.path)
    neighborhood = [
        _sanitize_goal_header(line, root_goal)
        for line in _neighborhood_lines(wu, units)
    ]
    description = _sanitize_goal_header(wu.description or "", root_goal)

    cap = int(char_cap) if char_cap else PACKET_CHAR_CAP
    if cap < 64:
        cap = 64

    omitted: List[str] = []
    mode = "full"
    prompt = ""
    while True:
        workunit_block = _workunit_block(wu, description, fc, root_goal)
        prompt = _assemble_prompt(
            phase=phase,
            workunit_block=workunit_block,
            invariants=invariants,
            acceptance=acceptance,
            evidence_paths=evidence_paths,
            neighborhood=neighborhood,
            steering=steer_lines,
            goal_ref=goal_ref,
            neighborhood_mode=mode,
            omitted=omitted,
            root_goal=root_goal,
        )
        prompt = _sanitize_goal_header(prompt, root_goal)
        if len(prompt) <= cap:
            break
        if mode == "full" and neighborhood:
            mode = "ids"
            omitted.append("neighborhood_detail")
            continue
        if mode == "ids" and neighborhood:
            mode = "drop"
            omitted.append("neighborhood")
            continue
        if evidence_paths:
            evidence_paths = evidence_paths[:-1]
            if "evidence_paths" not in omitted:
                omitted.append("evidence_paths")
            continue
        if len(acceptance) > 1:
            acceptance = acceptance[:-1]
            if "acceptance" not in omitted:
                omitted.append("acceptance")
            continue
        if len(acceptance) == 1:
            acceptance = []
            if "acceptance" not in omitted:
                omitted.append("acceptance")
            continue
        extra = invariants[len(pinned):]
        if extra:
            invariants = pinned + extra[:-1]
            if "invariants" not in omitted:
                omitted.append("invariants")
            continue
        if mode != "drop":
            mode = "drop"
            if "neighborhood" not in omitted:
                omitted.append("neighborhood")
            continue
        if len(description) > 80:
            description = description[:77].rstrip() + "..."
            if "description_tail" not in omitted:
                omitted.append("description_tail")
            continue
        if len(steer_lines) > len(constraint_texts):
            steer_lines = [f"[constraint] {text}" for text in constraint_texts]
            if "steering_non_constraint" not in omitted:
                omitted.append("steering_non_constraint")
            continue
        break

    if len(prompt) > cap:
        # Last resort: keep pinned constraints, drop the rest. Visible.
        for field in (
            "acceptance",
            "evidence_paths",
            "neighborhood",
            "failure_context",
            "invariants",
        ):
            if field not in omitted:
                omitted.append(field)
        evidence_paths = []
        acceptance = []
        workunit_block = _workunit_block(wu, description, None, root_goal)
        prompt = _assemble_prompt(
            phase=phase,
            workunit_block=workunit_block,
            invariants=pinned,
            acceptance=[],
            evidence_paths=[],
            neighborhood=[],
            steering=[f"[constraint] {text}" for text in constraint_texts],
            goal_ref=goal_ref,
            neighborhood_mode="drop",
            omitted=omitted,
            root_goal=root_goal,
        )
        prompt = _sanitize_goal_header(prompt, root_goal)
        if len(prompt) > cap:
            raise PacketBudgetError(
                omitted=tuple(dict.fromkeys(omitted)),
                shortfall=len(prompt) - cap,
                prompt_len=len(prompt),
                cap=cap,
            )

    refuse_goal_dump(prompt, root_goal)
    omitted_t = tuple(dict.fromkeys(omitted))
    identities = _stamp_packet_evidence(
        evidence_paths, workspace=workspace, evidence=evidence
    )
    budget_signal = ""
    if omitted_t:
        budget_signal = "TRUNCATION: omitted=" + ",".join(omitted_t)
    if budget is not None:
        evidence_tokens = sum(
            max(0, int(item.size)) for item in identities
        ) // max(1, int(CHARS_PER_TOKEN))
        result = preflight_packet(
            budget,
            prompt,
            evidence_tokens=evidence_tokens,
            kind="worker",
        )
        if not result.ok:
            raise PacketBudgetError(
                result=result,
                omitted=omitted_t,
                prompt_len=len(prompt),
                cap=int(result.per_request_ctx),
            )
        budget_signal = (
            (budget_signal + "; " if budget_signal else "") + result.remedy
        )

    return WorkerPacket(
        unit_id=str(wu.id),
        phase=str(phase or ""),
        workunit=_sanitize_goal_header(
            f"{wu.id} {wu.role} {wu.description}", root_goal
        ),
        invariants=tuple(invariants),
        acceptance=tuple(acceptance),
        neighborhood=tuple(neighborhood),
        evidence_paths=tuple(evidence_paths),
        steering=tuple(steer_lines),
        prompt=prompt,
        truncated=bool(omitted_t),
        omitted=omitted_t,
        evidence=tuple(identities),
        budget_signal=budget_signal,
    )
