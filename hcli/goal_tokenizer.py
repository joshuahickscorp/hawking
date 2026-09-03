"""Goal tokenizer: raw directive text -> typed ``hcli.goal_ir.GoalNode`` atoms.

"Token" here means a TYPED UNIT OF INTENT (a ``GoalType``), never a
BPE/vocabulary token. Do not let that word leak into anything user-facing
with its usual LLM meaning.

``hcli.goal.GoalCompiler`` already turns goal text into WorkUnits, but it is
an OBLIGATION EXTRACTOR: one ``WorkUnit`` per sentence/heading, no provenance,
no OUTCOME/METHOD split. This module feeds ``hcli.goal_ir.GoalNode`` instead
-- see that module's docstring for why the split matters and what the
provenance rules enforce. This module owns none of that enforcement; it only
has to satisfy it.

THE SEPARATION THAT MATTERS MOST: a sentence shaped like "<outcome> by/via/
using <method>" (e.g. "Make Odyssey faster by caching models on SSD") mints
TWO nodes -- an ``OBJECTIVE`` for the outcome and a ``SUGGESTED_METHOD`` that
``dependencies``-links back to it. The method can fail its verifier and be
superseded without touching the objective; folding it into the objective's
statement would make that impossible.

RULES-ONLY BY DESIGN. Every classifier here is a regex or a keyword table --
no model call, no network, bounded per call. Detection is intentionally
conservative: a sentence that matches nothing is DROPPED, not forced into a
type, because over-decomposition (one atom per sentence) is the exact failure
mode this schema exists to avoid. The seam for a smarter classifier is
structural, not a parameter: swap or extend ``_classify`` (and the tables
above it) with a model-backed pass over sentences ``_classify`` returns
``None`` for, and it degrades to today's behavior with none of them wired.

PROVENANCE: every emitted node is ``Provenance.DERIVED`` with a real PASTE
``source_ref`` (via ``preserve_source``) spanning the exact bytes it came
from -- never ``EXPLICIT_USER`` (this module infers the type; it does not
receive a human's explicit per-atom assertion) and never ungrounded
``MODEL_INFERRED`` (everything here is read straight out of the input text).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .goal import GoalCompiler
from .goal_ir import (
    MAX_ID_CHARS,
    STATEMENT_MAX_CHARS,
    GoalNode,
    GoalType,
    Provenance,
    Status,
    make_stable_id,
    preserve_source,
)
from .paste_cache import PasteCache

# Reused, not reimplemented: GoalCompiler's file-reference regex is already
# tested and handles the leading-slash gotcha (see its comment). A parallel
# copy here would be exactly the kind of duplicate parser the module
# docstring in goal_ir.py warns against.
_compiler = GoalCompiler()

# ---------------------------------------------------------------------------
# Span-aware line/sentence splitting. GoalCompiler._sentences() throws
# offsets away (it strips and rejoins); this module needs the ORIGINAL
# char_start/char_end into the raw text for preserve_source(), so it walks
# lines itself and only borrows GoalCompiler for file-reference regex reuse.
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^(\s*)([-*]|\d+[.)])\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*```")


def _sentence_spans(line: str) -> List[Tuple[int, int]]:
    """(start, end) for each sentence in *line*, a terminator only ends one
    when followed by whitespace or end-of-line -- a bare ``[^.!?]+`` class
    split (or GoalCompiler's own ``.!?`` char-class regex) cuts a filename
    like ``hcli/goal.py`` in half at the dot, since nothing there requires
    whitespace after it."""
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(line)
    for i, ch in enumerate(line):
        if ch in ".!?":
            nxt = line[i + 1] if i + 1 < n else ""
            if not nxt or nxt.isspace():
                spans.append((start, i + 1))
                start = i + 1
    if start < n:
        spans.append((start, n))
    return spans


def _iter_units(text: str):
    """Yield dicts for each heading / bullet / sentence-like span in *text*.

    Each dict carries ``start``/``end`` (offsets into *text*, for
    ``preserve_source``), ``text``, ``is_heading``, ``heading_level``,
    ``is_bullet``, and ``section`` (the nearest preceding heading, as
    ``(level, lowercase_title)``, or ``None``) -- section context is what
    lets an "Acceptance Criteria:" heading turn its bullets into
    ``SUCCESS_CRITERION`` nodes without each bullet needing its own marker
    word. Text inside fenced code blocks is skipped: code is not a goal atom.
    """
    pos = 0
    section: Optional[Tuple[int, str]] = None
    in_fence = False
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped_line = line.rstrip("\n").rstrip("\r")
        if _FENCE_RE.match(stripped_line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        content = stripped_line.strip()
        if not content:
            continue

        heading = _HEADING_RE.match(stripped_line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            t_start = line_start + stripped_line.index(title)
            yield {
                "start": t_start,
                "end": t_start + len(title),
                "text": title,
                "is_heading": True,
                "heading_level": level,
                "is_bullet": False,
                "section": section,
            }
            section = (level, title.lower())
            continue

        bullet = _BULLET_RE.match(stripped_line)
        if bullet:
            body = bullet.group(3)
            marker_end = len(bullet.group(1)) + len(bullet.group(2))
            b_start = line_start + stripped_line.index(body, marker_end)
            body_stripped = body.strip()
            if body_stripped:
                yield {
                    "start": b_start + (len(body) - len(body.lstrip())),
                    "end": b_start + len(body.rstrip()),
                    "text": body_stripped,
                    "is_heading": False,
                    "heading_level": None,
                    "is_bullet": True,
                    "section": section,
                }
            continue

        for raw_start, raw_end in _sentence_spans(stripped_line):
            raw = stripped_line[raw_start:raw_end]
            lstripped = raw.lstrip()
            if not lstripped:
                continue
            sent = lstripped.rstrip()
            if not sent:
                continue
            s_off = len(raw) - len(lstripped)
            s_start = line_start + raw_start + s_off
            yield {
                "start": s_start,
                "end": s_start + len(sent),
                "text": sent,
                "is_heading": False,
                "heading_level": None,
                "is_bullet": False,
                "section": section,
            }


def _clean_statement(text: str) -> str:
    """A compact restatement -- see GoalNode's docstring on why not a copy."""
    cleaned = re.sub(r"\s+", " ", text).strip().rstrip(" .;:,")
    if not cleaned:
        cleaned = "(unlabeled)"
    if len(cleaned) > STATEMENT_MAX_CHARS:
        cleaned = cleaned[: STATEMENT_MAX_CHARS - 3].rstrip() + "..."
    return cleaned


# ---------------------------------------------------------------------------
# Stable ids. make_stable_id() only normalizes a caller-supplied slug; this
# module is the caller, so it has to pick the slug (first few content words,
# stopwords dropped) and handle two sentences slugging to the same id.
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "to", "of", "and", "or", "is", "are", "this",
        "that", "for", "on", "in", "with", "by", "be", "do", "not", "don",
        "via", "using", "it", "as", "we", "should", "must", "will", "can",
        "its", "so", "but", "if", "when", "then",
    }
)


def _slug_from(text: str, max_words: int = 6) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if w not in _STOPWORDS]
    if not words:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return "_".join(words[:max_words]) or "item"


def _stable_id(goal_type: GoalType, seed_text: str, used: Set[str]) -> str:
    candidate = None
    for max_words in (6, 4, 3, 2, 1):
        try:
            candidate = make_stable_id(goal_type, _slug_from(seed_text, max_words))
            break
        except ValueError:
            continue
    if candidate is None:
        candidate = make_stable_id(goal_type, "item")
    base, suffix = candidate, 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = (base[: MAX_ID_CHARS - len(tail)] if len(base) + len(tail) > MAX_ID_CHARS else base) + tail
        suffix += 1
    return candidate


# ---------------------------------------------------------------------------
# Classification. Ordered most-specific-first; the first match wins. A
# sentence matching nothing returns None and is dropped -- see module
# docstring on why that beats forcing a type.
# ---------------------------------------------------------------------------

# ponytail: lexical, not semantic -- "do not X" vs "instead of/remain/become"
# is a blunt but cheap way to separate a concrete action ban (HARD_CONSTRAINT,
# e.g. "do not delete source specimens") from a standing anti-pattern
# (ANTI_GOAL, e.g. "do not let Claude remain hot-loop orchestrator"). Upgrade
# path: a model classifier for sentences this misses.
_PROHIBITION_MARKERS = ("forbidden", "prohibited", "banned", "not allowed", "not permitted", "may not")
_NEGATION_MARKERS = ("do not ", "don't ", "never ", "must not ", "cannot ", "won't ")
_ANTI_GOAL_STRUCTURAL = ("instead of", "rather than", " remain ", " become ", " stay as ")
_POSITIVE_HARD_MARKERS = ("must always", "non-negotiable", "hard constraint", "invariant:", "under no circumstances")


def _detect_negation_family(padded_lower: str) -> Optional[GoalType]:
    if any(w in padded_lower for w in _PROHIBITION_MARKERS):
        return GoalType.PROHIBITION
    has_negation = any(w in padded_lower for w in _NEGATION_MARKERS)
    if has_negation and any(w in padded_lower for w in _ANTI_GOAL_STRUCTURAL):
        return GoalType.ANTI_GOAL
    if has_negation:
        return GoalType.HARD_CONSTRAINT
    if any(w in padded_lower for w in _POSITIVE_HARD_MARKERS):
        return GoalType.HARD_CONSTRAINT
    return None


# Ordered: research/evidence/authority/stop/continuation/dependency before the more
# generic priority/preference/hypothesis/example/future-option markers, so a
# sentence like "requires approval" lands on AUTHORITY_REQUIRED rather than
# the bare word "requires" being read as a DEPENDENCY.
_KEYWORD_RULES: Tuple[Tuple[GoalType, Tuple[str, ...]], ...] = (
    (GoalType.FAILURE_CRITERION, ("fails if", "failure:", "breaks if", "regresses if", "considered failed", "counts as a failure")),
    (GoalType.SUCCESS_CRITERION, ("success:", "succeeds when", "passes when", "done when", "acceptance criteri", "considered done", "considered complete")),
    (GoalType.EVIDENCE_REQUIREMENT, ("must show evidence", "must produce a receipt", "leave a receipt", "prove that", "verify with", "record evidence", "attach evidence")),
    (GoalType.AUTHORITY_REQUIRED, ("requires authority", "needs approval", "needs permission", "ask before", "confirm before proceeding", "requires sign-off")),
    (GoalType.AUTHORITY_GRANT, ("authorized to", "granted authority", "permission to", "free to act", "cleared to")),
    (GoalType.STOP_CONDITION, ("stop when", "stop if", "halt when", "abort when", "kill the loop when")),
    (GoalType.CONTINUATION_POLICY, ("continue until", "keep going until", "resume when", "retry until", "self-correct")),
    (GoalType.DEPENDENCY, ("depends on", "blocked by", "cannot start until", "must complete before", "waits for")),
    (GoalType.PRIORITY, ("top priority", "highest priority", "prioritize this", " p0:", " p0 ", " p1:", " p1 ")),
    (GoalType.SOFT_PREFERENCE, ("prefer ", "preferably", "ideally", "when possible", "if possible", "nice to have", "would be nice")),
    (GoalType.HYPOTHESIS, ("hypothesis:", "we believe", "might be because", "suspect that", "could be due to")),
    (GoalType.EXAMPLE, ("for example", "e.g.", "such as", "example:")),
    (GoalType.FUTURE_OPTION, ("in the future", "could eventually", "future option:", "not now, but", "someday", "later, consider")),
)

_COMMAND_RE = re.compile(r"`[^`]+`")
_QUOTED_EXAMPLE_RE = re.compile(
    r"^\s*(?:(?:\"[^\"\n]*\")|(?:“[^”\n]*”)|(?:‘[^’\n]*’)|(?:'[^'\n]*'))\s*[.!?]?\s*$"
)
_VERIFY_WORD_RE = re.compile(r"\b(test|tests|pytest|verify|validate|run)\b", re.I)
_RESOURCE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:GB|GiB|MB|TB|cores?|CPUs?|GPUs?)\b", re.I)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?|days?|weeks?)\b", re.I)

# Imperative-shaped outcome verbs, checked only near the SENTENCE START (see
# _detect_objective) to keep this from firing on a verb buried in an
# unrelated clause later in a long sentence.
_OBJECTIVE_VERBS = (
    "make", "reduce", "increase", "improve", "speed up", "fix", "ensure",
    "build", "ship", "optimize", "compress", "eliminate", "unify", "cut",
    "shrink", "stabilize", "harden", "unblock", "accelerate", "minimize",
    "maximize", "restore", "prevent",
)
_METHOD_SPLIT_RE = re.compile(r"\s+\b(?:by|via|using)\b\s+", re.I)
_METHOD_FIRST_RE = re.compile(
    r"^\s*use\s+(?P<methods>.+?)\s+to\s+"
    r"(?P<outcome>(?:get|achieve|reach|bring|keep|take)\s+.+?)\s*[.!?]?\s*$",
    re.I,
)
_FORGET_RE = re.compile(r"^\s*forget\b.+?\bfor\s+now\s*[.!?]?\s*$", re.I)
_MEASUREMENT_TARGET_RE = re.compile(r"\b(?:hdd|ssd|bottleneck|contention)\b", re.I)


def _split_method_list(methods: str) -> List[str]:
    """Split a method list without treating narrative as a new goal.

    This is intentionally only used after ``_METHOD_FIRST_RE`` has already
    established the sentence's outcome/method grammar. Commas and a final
    conjunction are enough for the compact imperative form we accept here;
    arbitrary prose remains subject to the ordinary conservative classifier.
    """
    return [
        item.strip(" .;:,")
        for item in re.split(r"\s*,\s*|\s+and\s+", methods, flags=re.I)
        if item.strip(" .;:,")
    ]


def _subunit(
    unit: Dict[str, Any], start: int, end: int, **metadata: Any
) -> Dict[str, Any]:
    """Return a span-preserving child of a sentence-like tokenizer unit."""
    text = unit["text"]
    child = text[start:end].strip(" \t,;")
    if not child:
        return unit
    left = start + len(text[start:end]) - len(text[start:end].lstrip(" \t,;"))
    right = left + len(child)
    return {
        **unit,
        "start": unit["start"] + left,
        "end": unit["start"] + right,
        "text": child,
        **metadata,
    }


def _expand_compound_update(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split the one update grammar that carries four independent intents.

    ``forget X for now; improve Y, especially Z, but don't abandon W`` is
    not one prohibition. It contains a parked item, an elevated objective,
    a measurement hypothesis, and a protective constraint. Splitting only
    this recognizable shape avoids the old failure (the first negation won)
    without turning arbitrary comma-heavy prose into one atom per clause.
    """
    if unit["is_heading"] or unit["is_bullet"]:
        return [unit]
    text = unit["text"]
    forget = re.search(r"\bforget\b.+?\bfor\s+now\b", text, re.I)
    protect = re.search(r"\bbut\s+don['’]t\s+abandon\b.+", text, re.I)
    if not forget or not protect or forget.end() > protect.start():
        return [unit]
    middle = text[forget.end():protect.start()]
    improve = re.search(r"\bimprove\b.+", middle, re.I)
    if not improve:
        return [unit]

    middle_text = middle[improve.start():]
    especially = re.search(r"\s*,\s*especially\s+", middle_text, re.I)
    if not especially:
        return [unit]

    children = [
        _subunit(unit, forget.start(), forget.end()),
        _subunit(
            unit,
            forget.end() + improve.start(),
            forget.end() + improve.start() + especially.start(),
            elevated=True,
        ),
        _subunit(
            unit,
            forget.end() + improve.start() + especially.end(),
            forget.end() + improve.start() + len(middle_text),
        ),
        _subunit(unit, protect.start(), len(text)),
    ]
    return [child for child in children if child["text"].strip()]

_ACCEPTANCE_SECTION_RE = re.compile(r"acceptance|success criteri", re.I)
_FAILURE_SECTION_RE = re.compile(r"failure criteri", re.I)


def _detect_objective(sentence: str) -> bool:
    head = sentence.lower()[:60]
    return any(re.search(rf"\b{re.escape(v)}\b", head) for v in _OBJECTIVE_VERBS)


def _classify(
    sentence: str,
    *,
    is_bullet: bool,
    section: Optional[Tuple[int, str]],
    elevated: bool = False,
) -> Optional[Tuple[GoalType, Dict[str, Any]]]:
    lower = " " + sentence.lower() + " "

    if is_bullet and section is not None:
        title_lower = section[1]
        if _ACCEPTANCE_SECTION_RE.search(title_lower):
            return GoalType.SUCCESS_CRITERION, {"confidence": 0.85}
        if _FAILURE_SECTION_RE.search(title_lower):
            return GoalType.FAILURE_CRITERION, {"confidence": 0.85}

    # A defer-now instruction is a live, revisitable work item, not a
    # destructive deletion and not a generic hard constraint. The caller
    # carries the explicit lifecycle override into GoalNode.
    if _FORGET_RE.search(sentence):
        return GoalType.SUBOBJECTIVE, {
            "confidence": 0.75,
            "status": Status.PARKED,
            "priority": 1,
        }

    # A fully quoted sentence is an example/counterexample when it appears
    # in directive prose (not a new imperative from the user). This matters
    # for constructions such as ``Do not interpret this as: "make X"``:
    # treating the inner quotation as an OBJECTIVE would schedule the exact
    # thing the outer directive prohibited. Keep it traceable as EXAMPLE,
    # but never let it become a frontier.
    if _QUOTED_EXAMPLE_RE.match(sentence):
        return GoalType.EXAMPLE, {"confidence": 0.9}

    negated = _detect_negation_family(lower)
    if negated is not None:
        return negated, {"confidence": 0.8}

    for goal_type, markers in _KEYWORD_RULES:
        if any(marker in lower for marker in markers):
            return goal_type, {"confidence": 0.75}

    # Method-first imperative: keep the requested outcome independent from
    # the proposed mechanisms. This must run before RESOURCE/TEMPORAL rules
    # so a target such as "under 24h" does not swallow the whole sentence.
    method_first = _METHOD_FIRST_RE.match(sentence)
    if method_first:
        methods = _split_method_list(method_first.group("methods"))
        outcome = method_first.group("outcome").strip(" .;:,")
        if outcome and methods:
            if re.match(r"^(?:get|achieve|reach|bring|keep|take)\s+", outcome, re.I):
                outcome = re.sub(
                    r"^(?:get|achieve|reach|bring|keep|take)\s+",
                    "",
                    outcome,
                    flags=re.I,
                )
            return GoalType.OBJECTIVE, {
                "confidence": 0.65,
                "statement": outcome,
                "method_texts": methods,
            }

    if _COMMAND_RE.search(sentence) and _VERIFY_WORD_RE.search(sentence):
        return GoalType.EVIDENCE_REQUIREMENT, {"confidence": 0.75}

    if _RESOURCE_RE.search(sentence):
        return GoalType.RESOURCE_REQUIREMENT, {"confidence": 0.7}

    if _DATE_RE.search(sentence) or _DURATION_RE.search(sentence):
        return GoalType.TEMPORAL_CONSTRAINT, {"confidence": 0.7}

    if sentence.rstrip().endswith("?"):
        return GoalType.OPEN_QUESTION, {"confidence": 0.55}

    if _detect_objective(sentence):
        goal_type = GoalType.SUBOBJECTIVE if is_bullet else GoalType.OBJECTIVE
        extra: Dict[str, Any] = {"confidence": 0.65}
        # THE LOAD-BEARING SPLIT: "<outcome> by/via/using <method>" mints an
        # OBJECTIVE for the outcome and a SUGGESTED_METHOD for the method,
        # never one node carrying both (see module docstring).
        split = _METHOD_SPLIT_RE.search(sentence)
        if split:
            outcome = sentence[: split.start()].strip()
            method = sentence[split.end():].strip()
            if len(outcome) >= 4 and len(method) >= 4:
                extra["statement"] = outcome
                extra["method_text"] = method
        if elevated or "especially" in lower:
            extra["priority"] = 1
        return goal_type, extra

    # In the compound update grammar, the "especially ..." clause is a
    # measurement target. It is a hypothesis until evidence establishes the
    # bottleneck; never promote the noun phrase to a fact. Keep this after
    # open-question and imperative detection so an objective mentioning SSD,
    # or a question about a bottleneck, keeps its primary type.
    if _MEASUREMENT_TARGET_RE.search(sentence):
        return GoalType.HYPOTHESIS, {"confidence": 0.6}

    return None


def tokenize(
    text: str,
    *,
    cache: PasteCache,
    mission: Optional[str] = None,
    parent_ultragoal: Optional[str] = None,
) -> List[GoalNode]:
    """Compile raw directive prose into typed, provenance-tagged GoalNodes.

    Deterministic and bounded: one linear pass over *text*, no model call.
    *cache* is required (not defaulted to a fresh ``PasteCache()``) so a
    caller cannot accidentally scatter pastes into the real repo's
    ``.hcli/pastes`` from a throwaway call -- tests pass a ``tmp_path``-
    rooted cache, real callers pass their working cache explicitly.

    *parent_ultragoal*, when given, is a steer-style call: atoms attach to an
    already-live goal and no new ``ULTRAGOAL`` is synthesized even if *text*
    opens with a top-level heading. Otherwise a leading ``# Heading`` becomes
    the root ``ULTRAGOAL`` and everything else hangs off it.
    """
    raw = str(text or "")
    if not raw.strip():
        return []

    nodes: List[GoalNode] = []
    used_ids: Set[str] = set()
    ultragoal_id = parent_ultragoal
    last_objective_id: Optional[str] = None

    def _mint(
        goal_type: GoalType,
        seed_text: str,
        statement: str,
        *,
        start: int,
        end: int,
        confidence: float,
        dependencies: Tuple[str, ...] = (),
        resources: Tuple[str, ...] = (),
        priority: int = 2,
        status: Status = Status.ACTIVE,
    ) -> GoalNode:
        node_id = _stable_id(goal_type, seed_text, used_ids)
        used_ids.add(node_id)
        sref = preserve_source(cache, raw, char_start=start, char_end=end, mission=mission)
        node = GoalNode(
            id=node_id,
            type=goal_type,
            statement=statement,
            provenance=Provenance.DERIVED,
            confidence=confidence,
            priority=priority,
            status=status,
            dependencies=dependencies,
            resources=resources,
            parent_ultragoal=(ultragoal_id if goal_type is not GoalType.ULTRAGOAL else None),
            source_refs=(sref,),
        )
        nodes.append(node)
        return node

    for raw_unit in _iter_units(raw):
        for unit in _expand_compound_update(raw_unit):
            u_text = unit["text"]
            if len(u_text) < 4:
                continue

            if unit["is_heading"]:
                if unit["heading_level"] == 1 and ultragoal_id is None and not nodes:
                    node = _mint(
                        GoalType.ULTRAGOAL,
                        u_text,
                        _clean_statement(u_text),
                        start=unit["start"],
                        end=unit["end"],
                        confidence=0.9,
                    )
                    ultragoal_id = node.id
                # A non-root heading is section-context only (see _iter_units);
                # it never mints an atom of its own.
                continue

            result = _classify(
                u_text,
                is_bullet=unit["is_bullet"],
                section=unit["section"],
                elevated=bool(unit.get("elevated")),
            )
            if result is None:
                continue
            goal_type, extra = result

            statement = _clean_statement(extra.get("statement", u_text))
            # GoalNode.resources is a generic Tuple[str, ...]; this module uses it
            # to carry any file paths the sentence names, same reuse pattern as
            # goal.py's own _mentioned_and_known_files().
            files = tuple(_compiler._referenced_files(u_text))

            deps: Tuple[str, ...] = ()
            if goal_type is GoalType.SUBOBJECTIVE:
                anchor = last_objective_id or ultragoal_id
                if anchor:
                    deps = (anchor,)

            node = _mint(
                goal_type,
                u_text,
                statement,
                start=unit["start"],
                end=unit["end"],
                confidence=extra["confidence"],
                dependencies=deps,
                resources=files,
                priority=extra.get("priority", 2),
                status=extra.get("status", Status.ACTIVE),
            )
            if goal_type is GoalType.OBJECTIVE:
                last_objective_id = node.id

            method_texts = extra.get("method_texts")
            if method_texts is None and extra.get("method_text"):
                method_texts = [extra["method_text"]]
            for method_text in method_texts or ():
                _mint(
                    GoalType.SUGGESTED_METHOD,
                    method_text,
                    _clean_statement(method_text),
                    start=unit["start"],
                    end=unit["end"],
                    confidence=0.6,
                    dependencies=(node.id,),
                    resources=tuple(_compiler._referenced_files(method_text)),
                )

    return nodes


if __name__ == "__main__":
    import tempfile

    _DEMO = """# Reduce Odyssey wall time

Make Odyssey faster by caching models on SSD.

Do not delete source specimens.
Prefer the local resident when possible.
Do not let Claude remain hot-loop orchestrator.

Success: the full suite passes with `pytest hcli/`.
Failure: any previously-passing test regresses.
"""
    with tempfile.TemporaryDirectory() as tmp:
        result = tokenize(_DEMO, cache=PasteCache(root=tmp))
        by_type: Dict[str, List[GoalNode]] = {}
        for n in result:
            by_type.setdefault(n.type.value, []).append(n)

        assert len(by_type.get("ULTRAGOAL", [])) == 1, by_type
        assert len(by_type.get("OBJECTIVE", [])) == 1, by_type
        assert len(by_type.get("SUGGESTED_METHOD", [])) == 1, by_type
        method = by_type["SUGGESTED_METHOD"][0]
        objective = by_type["OBJECTIVE"][0]
        assert method.dependencies == (objective.id,), (method, objective)
        assert "cach" in method.statement.lower(), method
        assert "faster" in objective.statement.lower(), objective
        assert "delete" not in objective.statement.lower(), objective

        assert len(by_type.get("HARD_CONSTRAINT", [])) == 1, by_type
        assert len(by_type.get("SOFT_PREFERENCE", [])) == 1, by_type
        assert len(by_type.get("ANTI_GOAL", [])) == 1, by_type
        assert len(by_type.get("SUCCESS_CRITERION", [])) == 1, by_type
        assert len(by_type.get("FAILURE_CRITERION", [])) == 1, by_type

        for n in result:
            assert n.provenance is Provenance.DERIVED, n
            assert n.source_refs, n

        print(f"OK: {len(result)} nodes across {len(by_type)} types")
