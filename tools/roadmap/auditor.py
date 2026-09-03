"""Adversarial status derivation for the roadmap capability graph.

Statuses are never hand-written. Rules, in order:

1. Hardware-requiring + device absent => BLOCKED_HARDWARE (never BUILT/PASS).
2. No git-tracked definition => ABSENT (era I / bounty) or DORMANT (later era).
3. Definition whose only callers are tests => SCAFFOLDED. Receipts do not upgrade this.
4. Definition + non-test call/subprocess of the implementing SYMBOL => wired.
   Importing a module is not calling a capability. Registration is not wiring.
   Importability is not existence. kind=import never justifies wired (at most
   SCAFFOLDED). A name-only match (constant, string, comment, except-handler)
   is weak_signal and never moves status. wired is not accepted and is not BUILT.
5. accepted is orthogonal: the gate's own acceptance criterion is demonstrably
   satisfied — a receipt or measurement that meets the stated bar, not merely a
   receipt that exists on the topic. A numeric bar (EBPW <= 1) is compared
   against the real value in the receipt; failing the comparison is not accepted.
6. BUILT requires wired AND accepted. wired and not accepted => WIRED.
   accepted without wired is still SCAFFOLDED. Never let wired alone produce BUILT.
7. After all local statuses: a defined gate whose every dependency is
   ABSENT/BLOCKED_* becomes UNREACHABLE unless it is itself BLOCKED_HARDWARE.
8. Named software/external blocker that actually holds => BLOCKED_EXTERNAL.

Every non-ABSENT verdict cites file:line, a command, or a receipt path.
Citations are bound to the emitting commit and re-resolved if a line is past EOF.
Evidence tier is STATIC: this auditor does not take hardware measurements.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.roadmap import EVIDENCE_TIER, GRAPH_REL, SCHEMA, VERSION
from tools.roadmap import catalog
from tools.roadmap import lineage
from tools.roadmap.gitfs import (
    REPO,
    SourceView,
    blob_text,
    head_commit,
    head_paths,
    prefetch_blobs,
)
from tools.roadmap.hardware import probe, probe_all
from tools.roadmap.parse import load_existing_state, parse_roadmap, span
from tools.roadmap import index_client
from tools.roadmap import reach

_BUILT_KINDS = reach.BUILT_KINDS

_LATER_ERAS = {"II", "III", "IV", "V"}


_CITATION_KEYS = (
    "evidence_refs",
    "code_refs",
    "runtime_caller",
    "import_sites",
    "weak_signals",
    "tests",
)

_NUMERIC_OPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


def _cite(file: str, line: int | None, *, kind: str, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"file": file, "line": line, "kind": kind}
    if note:
        out["note"] = note
    return out


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _compare_numeric(measured: float, op: str, threshold: float) -> bool:
    fn = _NUMERIC_OPS.get(op)
    if fn is None:
        raise ValueError(f"unknown numeric acceptance op {op!r}")
    return bool(fn(measured, threshold))


def _load_json_receipt(rel: str, view: SourceView) -> Any | None:
    text = view.read(rel) if rel else ""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _wired_fact(look: dict[str, Any]) -> dict[str, Any]:
    invocations = _invocation_sites(look)
    if invocations:
        evidence = [
            _cite(s["file"], s["line"], kind=s.get("kind") or "call", note=s.get("symbol"))
            for s in invocations
        ]
        return {"value": True, "evidence": evidence}
    defined_file = look["defined_refs"][0]["file"] if look.get("defined_refs") else None
    defined_line = look["defined_refs"][0]["line"] if look.get("defined_refs") else None
    if look.get("defined"):
        note = (
            "definition exists; no non-test call/subprocess of the implementing "
            "symbol. A module import is not a call. not wired."
        )
    else:
        note = "no git-tracked definition; not wired"
    return {
        "value": False,
        "evidence": [
            {
                "kind": "no_production_caller",
                "file": defined_file or ((look.get("missing_paths") or [None])[0]),
                "line": defined_line,
                "note": note,
            }
        ],
    }


def _numeric_acceptance(spec: dict[str, Any], view: SourceView) -> dict[str, Any]:
    receipt = spec.get("receipt")
    field = spec.get("field")
    op = spec.get("op")
    threshold = spec.get("threshold")
    base = {
        "kind": "numeric_acceptance",
        "file": receipt,
        "line": None,
        "field": field,
        "op": op,
        "threshold": threshold,
    }
    if not receipt or not field or op not in _NUMERIC_OPS or threshold is None:
        base["note"] = "numeric acceptance spec is incomplete; not accepted"
        return {"value": False, "evidence": [base]}
    data = _load_json_receipt(str(receipt), view)
    if data is None:
        base["note"] = f"receipt {receipt} missing or not JSON; not accepted"
        return {"value": False, "evidence": [base]}
    raw = _dig(data, str(field))
    if raw is None:
        base["measured"] = None
        base["note"] = f"{field} missing in {receipt}; not accepted"
        return {"value": False, "evidence": [base]}
    try:
        measured = float(raw)
    except (TypeError, ValueError):
        base["measured"] = raw
        base["note"] = f"measured {field}={raw!r} is not numeric; not accepted"
        return {"value": False, "evidence": [base]}
    try:
        bar = float(threshold)
    except (TypeError, ValueError):
        base["measured"] = measured
        base["note"] = f"threshold {threshold!r} is not numeric; not accepted"
        return {"value": False, "evidence": [base]}
    ok = _compare_numeric(measured, str(op), bar)
    # Format so a reader (and the FLASH test) sees the real value against the bar.
    bar_s = str(int(bar)) if float(bar) == int(bar) else str(bar)
    base["measured"] = measured
    base["note"] = f"measured {measured} against required {op} {bar_s}"
    return {"value": ok, "evidence": [base]}


_CRITERION_PLACEHOLDERS = (
    "not readable",
    "roadmap missing",
    "h-roadmap.md missing",
)


def _criterion_text(doc: dict[str, Any]) -> str:
    """The quoted criterion, across the three shapes the receipt corpus uses.

    Receipts write `criterion_quoted`, `criterion` (str or {quoted|quote}), or
    `quote`. Assuming one shape silently mis-reads two thirds of the corpus.
    """
    for key in ("criterion_quoted", "criterion", "quote"):
        val = doc.get(key)
        if isinstance(val, dict):
            val = val.get("quoted") or val.get("quote")
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _criterion_is_real(doc: dict[str, Any]) -> str:
    """"" when the receipt quotes a real criterion, else why it does not.

    criterion_altered=false is a claim ABOUT the criterion, so it is worth
    nothing when the criterion itself is absent or is a "roadmap not readable"
    placeholder. Four acceptance harnesses used to substitute exactly such a
    string when the external roadmap was missing, which would let a gate be
    ACCEPTED against no obligation at all.
    """
    text = _criterion_text(doc).strip()
    if not text:
        return "the receipt quotes no criterion, so criterion_altered proves nothing"
    low = text.lower()
    if any(mark in low for mark in _CRITERION_PLACEHOLDERS):
        return (
            "the receipt quotes a roadmap-unreadable placeholder instead of the "
            f"criterion: {text[:80]!r}"
        )
    return ""


def _criterion_span(doc: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    """(start, end, pointer) across the shapes the corpus uses."""
    src = doc.get("criterion_source")
    if not isinstance(src, dict):
        crit = doc.get("criterion")
        src = crit if isinstance(crit, dict) else {}
    start, end = src.get("start_line"), src.get("end_line")
    return (start if isinstance(start, int) else None,
            end if isinstance(end, int) else None,
            src.get("pointer"))


def criterion_matches_its_source(doc: dict[str, Any]) -> tuple[str, str]:
    """Is the stored criterion still what its cited source says? (verdict, why).

    criterion_altered was a HARDCODED FALSE LITERAL at fourteen write sites across
    all six acceptance lanes and was computed at none of them, so half the BUILT
    law -- "verdict ACCEPTED and criterion_altered false" -- was satisfied by a
    constant. A receipt cannot be trusted to report that it did not move its own
    goalposts; the claim has to be recomputed from the source it cites.

    MATCHES      the span re-quotes into the stored text
    ALTERED      it does not, so the criterion changed after the receipt was written
    UNVERIFIABLE the receipt cites no span, so the claim cannot be checked at all
    """
    if doc.get("criterion_altered") or doc.get("criterion_weakened"):
        return "ALTERED", "the receipt says so itself"
    quoted = _criterion_text(doc)
    if not quoted.strip():
        return "UNVERIFIABLE", "the receipt quotes no criterion"
    start, end, pointer = _criterion_span(doc)
    if pointer:
        # Authored in-repo rather than quoted from the roadmap; the supplement is
        # the source, and it is committed, so compare against it.
        try:
            sup = json.loads((REPO / "civilization" / "GATE_CRITERIA_SUPPLEMENT.json").read_text())
            gate = str(pointer).split(".")[1] if "." in str(pointer) else ""
            want = ((sup.get("gates") or {}).get(gate) or {}).get("criterion") or ""
        except Exception:
            return "UNVERIFIABLE", "the criteria supplement is unreadable"
        if not want:
            return "UNVERIFIABLE", f"the supplement declares no criterion for {pointer!r}"
        return ("MATCHES", "") if want.strip() in quoted else (
            "ALTERED", "the stored quote no longer matches the criteria supplement")
    if start is None or end is None:
        return "UNVERIFIABLE", "the receipt cites no line span to re-quote"
    try:
        lines = lineage.roadmap_lines()
    except OSError:
        return "UNVERIFIABLE", "the canonical roadmap is unreadable"
    span = "\n".join(lines[start - 1:end]).strip()
    if not span:
        return "ALTERED", f"span {start}-{end} is empty in the roadmap it cites"
    return ("MATCHES", "") if span in quoted else (
        "ALTERED", f"span {start}-{end} does not re-quote into the stored criterion")


def _accepted_fact(probe: dict[str, Any], look: dict[str, Any], view: SourceView,
                   gate_id: str | None = None) -> dict[str, Any]:
    """The gate's own acceptance criterion, not 'a receipt on this topic exists'."""
    spec = probe.get("acceptance")
    if isinstance(spec, dict) and spec.get("kind") == "numeric":
        return _numeric_acceptance(spec, view)
    # An acceptance lane may demonstrate the criterion directly. That evidence
    # lives at receipts/acceptance/<GATE>.json and is consumed ADVERSARIALLY:
    # a receipt only counts when it says ACCEPTED, swears the criterion was not
    # altered, and names the command it ran. A receipt that merely exists, or
    # that reports BLOCKED, never produces accepted -- BLOCKED is a truthful
    # result, not a pass.
    if gate_id:
        rel = f"receipts/acceptance/{gate_id}.json"
        raw = view.read(rel)
        if raw:
            try:
                doc = json.loads(raw)
            except Exception:
                doc = None
            if isinstance(doc, dict):
                verdict = str(doc.get("verdict") or "").strip().upper()
                command = doc.get("command")
                unreal = _criterion_is_real(doc) if verdict == "ACCEPTED" else ""
                match, why = (criterion_matches_its_source(doc)
                              if verdict == "ACCEPTED" else ("", ""))
                altered = match == "ALTERED"
                if verdict == "ACCEPTED" and not altered and command and not unreal:
                    return {
                        "value": True,
                        "evidence": [
                            {
                                "kind": "acceptance_demonstrated",
                                "file": rel,
                                "line": None,
                                "command": command,
                                "evidence_tier": doc.get("evidence_tier"),
                                # RECOMPUTED, not read off the receipt.
                                "criterion_source_check": match,
                                "note": (
                                    "the gate's own criterion was demonstrated by a real run"
                                    if match == "MATCHES" else
                                    f"accepted, but the criterion could not be re-checked: {why}"
                                ),
                            }
                        ],
                    }
                if verdict:
                    return {
                        "value": False,
                        "evidence": [
                            {
                                "kind": "acceptance_refused",
                                "file": rel,
                                "line": None,
                                "note": (
                                    f"acceptance verdict {verdict}"
                                    + (f" (criterion ALTERED, refused: {why})" if altered else "")
                                    + (f" (criterion is not real: {unreal})" if unreal else "")
                                    + (f": {doc.get('blocker')}" if doc.get("blocker") else "")
                                ),
                            }
                        ],
                    }
    topic = (look.get("receipts") or [None])[0]
    note = (
        "wired is not accepted: no receipt or measurement demonstrates the "
        "gate's own acceptance criterion. A receipt on the topic is not the bar."
    )
    return {
        "value": False,
        "evidence": [
            {
                "kind": "acceptance_undemonstrated",
                "file": topic,
                "line": None,
                "note": note,
            }
        ],
    }


def _iter_citation_refs(entry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in _CITATION_KEYS:
        for ref in entry.get(key) or []:
            if isinstance(ref, dict):
                yield ref
    for fact_key in ("wired", "accepted"):
        fact = entry.get(fact_key)
        if isinstance(fact, dict):
            for ref in fact.get("evidence") or []:
                if isinstance(ref, dict):
                    yield ref


# Bounds-checking every citation used to spawn one `git cat-file` PER CITATION:
# 2447 subprocesses, 31s of a 61s audit, for a few dozen distinct files. The
# blob for a (path, commit) pair is immutable, so the count is memoizable and
# the cache is trivially correct.
_LINE_COUNT_CACHE: dict[tuple[str, str], int | None] = {}


def _line_count_at_commit(rel: str, commit: str, view: SourceView) -> int | None:
    if not rel or str(rel).startswith("/"):
        return None
    # Overlay content is mutable within a run, so it is never cached.
    if rel not in view.overlay:
        key = (str(rel), str(commit))
        if key in _LINE_COUNT_CACHE:
            return _LINE_COUNT_CACHE[key]
        value = _line_count_at_commit_uncached(rel, commit, view)
        _LINE_COUNT_CACHE[key] = value
        return value
    return _line_count_at_commit_uncached(rel, commit, view)


def _line_count_at_commit_uncached(rel: str, commit: str, view: SourceView) -> int | None:
    if not rel or str(rel).startswith("/"):
        return None
    if rel in view.overlay:
        text = view.read(rel)
        return len(text.splitlines()) if text else 0
    text = blob_text(commit, rel)
    if text is None:
        # Overlay-or-disk view of a path git does not know (should be rare).
        scanned = view.read(rel)
        if scanned:
            return len(scanned.splitlines())
        return None
    return len(text.splitlines())


def _find_symbol_line(text: str, needle: str | None, hint: int) -> int | None:
    if not text or not needle:
        return None
    hits = [i for i, line in enumerate(text.splitlines(), start=1) if needle in line]
    if not hits:
        return None
    return min(hits, key=lambda i: abs(i - hint))


def _bind_entry_citations(entry: dict[str, Any], view: SourceView, commit: str) -> list[str]:
    """Stamp commit on repo citations; re-resolve lines that sit past EOF.

    Returns remaining out-of-bounds descriptions (empty means clean).
    """
    violations: list[str] = []
    gid = entry.get("id")
    for ref in _iter_citation_refs(entry):
        rel = ref.get("file")
        line = ref.get("line")
        if not rel or str(rel).startswith("/"):
            continue
        ref["commit"] = commit
        if not isinstance(line, int):
            continue
        n = _line_count_at_commit(str(rel), commit, view)
        if n is None:
            continue
        if 1 <= line <= n:
            continue
        needle = ref.get("symbol") or ref.get("note")
        if rel in view.overlay:
            text = view.read(str(rel))
        else:
            text = blob_text(commit, str(rel)) or view.read(str(rel))
        new_line = _find_symbol_line(text or "", str(needle) if needle else None, line)
        if new_line is not None and 1 <= new_line <= n:
            ref["line"] = new_line
            ref["re_resolved"] = True
            continue
        violations.append(
            f"{gid} {rel}:{line} exceeds {n} lines at {commit}"
        )
    return violations


def citation_bound_violations(doc: dict[str, Any], view: SourceView | None = None) -> list[str]:
    """Return descriptions of citations whose line is past EOF at the emitting commit."""
    view = view or SourceView()
    commit = doc.get("generated_from_commit") or head_commit()
    out: list[str] = []
    for table in (doc.get("gates") or {}, doc.get("genes") or {}):
        for entry in table.values():
            for ref in _iter_citation_refs(entry):
                rel = ref.get("file")
                line = ref.get("line")
                if not rel or not isinstance(line, int) or str(rel).startswith("/"):
                    continue
                n = _line_count_at_commit(str(rel), ref.get("commit") or commit, view)
                if n is None:
                    continue
                if line < 1 or line > n:
                    out.append(
                        f"{entry.get('id')} {rel}:{line} exceeds {n} lines at "
                        f"{ref.get('commit') or commit}"
                    )
    return out


_LS_FILES_CACHE: dict[str, list[str]] = {}


def _git_ls_files(pattern: str) -> list[str]:
    cached = _LS_FILES_CACHE.get(pattern)
    if cached is not None:
        return cached
    import subprocess

    cp = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = [line for line in cp.stdout.splitlines() if line]
    _LS_FILES_CACHE[pattern] = rows
    return rows


def _receipt_hits(globs: list[str], view: SourceView) -> list[str]:
    hits: list[str] = []
    from glob import glob as _glob

    for pattern in globs:
        # glob against the working tree AND git ls-files via Python pathlib on REPO
        # plus git ls-files. Missing-from-disk receipts still count as citations,
        # never as BUILT evidence.
        disk = [p for p in _glob(str(REPO / pattern), recursive=True)]
        for p in disk:
            rel = str(Path(p).resolve().relative_to(REPO)) if Path(p).exists() else p
            hits.append(Path(rel).as_posix() if Path(rel).is_absolute() is False else str(Path(p).relative_to(REPO)))
        for line in _git_ls_files(pattern):
            if line not in hits:
                hits.append(line)
    return sorted(set(hits))


def _look_up(
    view: SourceView,
    probe: dict[str, Any],
    *,
    unique_paths: set[str],
) -> dict[str, Any]:
    """Collect definition / caller / test / receipt citations for one probe.

    runtime_caller is invocations only (call/subprocess of this probe's
    symbol). Module imports live in import_sites and cannot justify BUILT.
    """
    look = reach.scan_probe(view, probe, unique_paths=unique_paths)
    look["receipts"] = _receipt_hits(list(probe.get("receipt_globs") or []), view)
    return look


def _invocation_sites(look: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s
        for s in (look.get("runtime_caller") or [])
        if s.get("kind") in _BUILT_KINDS
    ]


def _next_upgrade(status: str, *, wake: str | None, callers_needed: str | None, deps: list[str]) -> str:
    if status == "BLOCKED_HARDWARE":
        return f"wake_condition {wake} becomes true"
    if status == "BLOCKED_EXTERNAL":
        return f"external blocker {wake} clears"
    if status == "SCAFFOLDED":
        return (
            f"wire a non-test call (or CLI subprocess) of "
            f"{callers_needed or 'the implementing symbol'} — an import is not a call"
        )
    if status == "WIRED":
        return (
            "demonstrate the gate's own acceptance criterion — a non-test caller "
            "is wired, not accepted, and does not produce BUILT"
        )
    if status == "ABSENT":
        return (
            f"implement {callers_needed or 'the named symbol'} and wire a non-test call of it"
        )
    if status == "DORMANT":
        return "Era I is sovereign; do not promote this until its era is admitted"
    if status == "UNREACHABLE":
        return "satisfy dependencies: " + ", ".join(deps or ["(unnamed)"])
    if status == "BUILT":
        return "promotion receipt / next gate in the dependency chain"
    return "re-run the auditor"


def _local_status(
    *,
    era: str,
    look: dict[str, Any],
    hw_id: str | None,
    hw_probe: dict[str, Any] | None,
    ext: str | None,
    wired: bool = False,
    accepted: bool = False,
    wired_evidence: list[dict[str, Any]] | None = None,
    accepted_evidence: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], str | None, str | None]:
    """Return (status, evidence_refs, hardware_blocker, software_blocker).

    wired and accepted are orthogonal. BUILT requires both. wired alone is WIRED.
    """
    evidence: list[dict[str, Any]] = []
    wired_evidence = list(wired_evidence or [])
    accepted_evidence = list(accepted_evidence or [])
    if hw_id and hw_probe and not hw_probe.get("present"):
        evidence.append(
            {
                "kind": "hardware_probe",
                "command": hw_id,
                "file": None,
                "line": None,
                "note": hw_probe.get("evidence"),
            }
        )
        if look["defined"]:
            evidence.extend(look["defined_refs"])
        if wired_evidence:
            evidence.extend(wired_evidence[:8])
        else:
            invocations = _invocation_sites(look)
            if invocations:
                evidence.extend(
                    _cite(s["file"], s["line"], kind=s.get("kind") or "call", note=s.get("symbol"))
                    for s in invocations[:8]
                )
        return "BLOCKED_HARDWARE", evidence, hw_id, None

    if not look["defined"]:
        evidence.append(
            {
                "kind": "git_cat_file",
                "command": "git cat-file -e HEAD:<path>",
                "file": (look["missing_paths"] or [None])[0],
                "line": None,
                "note": "no git-tracked definition for catalogued code_paths",
            }
        )
        # A NAMED external blocker outranks ABSENT. "nothing exists" and
        # "nothing exists, here is exactly what holds it and what would wake
        # it" are different facts, and the roadmap tracks them as different
        # states. Without this, the Theia model ladder read ABSENT even though
        # its blocker and wake conditions are recorded.
        if ext:
            return "BLOCKED_EXTERNAL", evidence, None, ext
        if era in _LATER_ERAS:
            return "DORMANT", evidence, None, None
        return "ABSENT", evidence, None, None

    evidence.extend(look["defined_refs"])
    # Load-bearing split: a non-test caller is wired, not BUILT. accepted is
    # a separate fact. Mutating the next branch to return BUILT on wired
    # alone must fail test_no_gate_is_built_on_wired_alone.
    if wired:
        evidence.extend(wired_evidence or [
            _cite(s["file"], s["line"], kind=s.get("kind") or "call", note=s.get("symbol"))
            for s in _invocation_sites(look)
        ])
        evidence.extend(accepted_evidence)
        if accepted:
            return "BUILT", evidence, None, None
        return "WIRED", evidence, None, None

    evidence.append(
        {
            "kind": "no_production_caller",
            "file": look["defined_refs"][0]["file"] if look["defined_refs"] else None,
            "line": look["defined_refs"][0]["line"] if look["defined_refs"] else None,
            "note": (
                "definition exists; no non-test call/subprocess of the implementing "
                "symbol. A module import is not a call. Receipts do not upgrade this."
            ),
        }
    )
    for s in (look.get("import_sites") or [])[:8]:
        evidence.append(
            _cite(
                s["file"],
                s["line"],
                kind="import",
                note="import supports SCAFFOLDED only; never BUILT",
            )
        )
    for s in (look.get("weak_signals") or [])[:4]:
        evidence.append(
            _cite(
                s.get("file"),
                s.get("line"),
                kind="weak_signal",
                note=s.get("note") or s.get("symbol") or "name-only match; does not move status",
            )
        )
    if look["tests"]:
        evidence.extend(_cite(s["file"], s["line"], kind="test") for s in look["tests"][:8])
    if look["receipts"]:
        evidence.append(
            {
                "kind": "receipt_citation_only",
                "file": look["receipts"][0],
                "line": None,
                "note": "receipt exists but producer+caller were not both found; not BUILT evidence",
            }
        )
    if accepted_evidence:
        evidence.extend(accepted_evidence)
    if ext:
        return "BLOCKED_EXTERNAL", evidence, None, ext
    return "SCAFFOLDED", evidence, None, None


def _fill_entry(
    *,
    entry_id: str | None = None,
    skeleton: dict[str, Any],
    probe: dict[str, Any],
    look: dict[str, Any],
    hw_cache: dict[str, dict[str, Any]],
    roadmap_file: str,
    view: SourceView,
) -> dict[str, Any]:
    era = probe.get("era") or skeleton.get("era") or "I"
    hw_id = probe.get("hardware_wake")
    hw_probe = hw_cache.get(hw_id) if hw_id else None
    wired = _wired_fact(look)
    accepted = _accepted_fact(probe, look, view, gate_id=entry_id)
    status, evidence, hw_block, sw_block = _local_status(
        era=era,
        look=look,
        hw_id=hw_id,
        hw_probe=hw_probe,
        ext=probe.get("software_blocker"),
        wired=wired["value"],
        accepted=accepted["value"],
        wired_evidence=wired["evidence"],
        accepted_evidence=accepted["evidence"],
    )
    acc = probe.get("acceptance_span") or {}
    entry = dict(skeleton)
    entry["era"] = era
    entry["gene"] = probe.get("gene")
    entry["dependencies"] = list(probe.get("dependencies") or [])
    entry["acceptance_span"] = span(
        roadmap_file, int(acc.get("start_line") or 0), int(acc.get("end_line") or 0),
        note="proof-obligation section (not copied)",
    )
    entry["status"] = status
    entry["wired"] = wired
    entry["accepted"] = accepted
    entry["evidence_tier"] = EVIDENCE_TIER
    entry["evidence_refs"] = evidence
    entry["code_refs"] = look["defined_refs"]
    entry["runtime_caller"] = _invocation_sites(look)
    entry["import_sites"] = list(look.get("import_sites") or [])
    entry["weak_signals"] = list(look.get("weak_signals") or [])
    entry["tests"] = look["tests"]
    entry["receipts_cited"] = look["receipts"]
    entry["hardware_blocker"] = hw_block
    entry["software_blocker"] = sw_block
    entry["wake_condition"] = hw_id if status == "BLOCKED_HARDWARE" else None
    if status == "BLOCKED_HARDWARE" and hw_id:
        entry["wake_condition_detail"] = hw_probe
    symbols = [s.get("symbol") for s in (probe.get("symbols") or []) if s.get("symbol")]
    needed = (
        ", ".join(symbols)
        or (look.get("symbols_scanned") or [None])[0]
        or (probe.get("modules") or [None])[0]
        or (probe.get("code_paths") or [None])[0]
    )
    entry["next_admissible_upgrade"] = _next_upgrade(
        status, wake=hw_id or sw_block, callers_needed=needed, deps=entry["dependencies"]
    )
    return entry


def _apply_unreachable(entries: dict[str, dict[str, Any]]) -> None:
    blocking = {"ABSENT", "BLOCKED_HARDWARE", "BLOCKED_EXTERNAL", "UNREACHABLE", "DORMANT"}
    # Multiple passes so chains collapse.
    for _ in range(8):
        changed = False
        for entry in entries.values():
            if entry["status"] in {"BLOCKED_HARDWARE", "ABSENT", "DORMANT", "UNREACHABLE", "BLOCKED_EXTERNAL"}:
                continue
            deps = entry.get("dependencies") or []
            if not deps:
                continue
            dep_rows = [entries[d] for d in deps if d in entries]
            if not dep_rows:
                continue
            if all(d["status"] in blocking for d in dep_rows):
                entry["status"] = "UNREACHABLE"
                entry["evidence_refs"] = list(entry.get("evidence_refs") or []) + [
                    {
                        "kind": "dependency",
                        "file": None,
                        "line": None,
                        "note": "all dependencies are ABSENT/BLOCKED/DORMANT/UNREACHABLE",
                    }
                ]
                entry["next_admissible_upgrade"] = _next_upgrade(
                    "UNREACHABLE", wake=None, callers_needed=None, deps=deps
                )
                entry["wake_condition"] = None
                changed = True
        if not changed:
            break


def _credit_disk_truth(view: SourceView) -> list[dict[str, Any]]:
    rows = []
    for rel in catalog.DISK_TRUTH_MODULES:
        rows.append(
            {
                "path": rel,
                "present_in_git": view.exists(rel),
                "evidence": f"git cat-file -e HEAD:{rel}",
            }
        )
    return rows


def _verify_absent_claims(view: SourceView) -> list[dict[str, Any]]:
    # SORTED, because head_paths() is a frozenset and list() of one varies with
    # PYTHONHASHSEED. Two audits of the SAME commit produced graphs differing in
    # 24 hawking_paths entries -- so the generated authority was not reproducible
    # at a fixed HEAD, which defeats diffing it, content-addressing it, and any
    # check of the form "did the graph change?".
    ls = sorted(head_paths())
    out = []
    for name, meaning in catalog.ABSENT_CLAIMS:
        hits = [
            p
            for p in ls
            if name in Path(p).parts
            or Path(p).stem == name
            or Path(p).name == name
            # A compound module name still IS the concept: semantic_transport.py
            # is a transport edge compiler. Matching only the bare stem understated
            # `transport` as ABSENT while tools/accelerator/semantic_transport.py existed.
            or name in Path(p).stem.split("_")
        ]
        # Filter known false friends for transport/placement.
        if name == "transport":
            hawking = [p for p in hits if p.startswith("tools/") or p.startswith("hcli/")]
            out.append(
                {
                    "claim": name,
                    "meaning": meaning,
                    "hawking_paths": hawking,
                    "unrelated_hits": [p for p in hits if p not in hawking],
                    "verdict": "ABSENT" if not hawking else "PRESENT",
                }
            )
        elif name == "placement":
            hawking = [
                p
                for p in hits
                if p.startswith("tools/") or p.startswith("hcli/")
            ]
            out.append(
                {
                    "claim": name,
                    "meaning": meaning,
                    "hawking_paths": hawking,
                    "unrelated_hits": [p for p in hits if p not in hawking],
                    "verdict": "ABSENT" if not hawking else "PRESENT",
                }
            )
        else:
            out.append(
                {
                    "claim": name,
                    "meaning": meaning,
                    "hawking_paths": hits,
                    "unrelated_hits": [],
                    "verdict": "ABSENT" if not hits else "PRESENT",
                }
            )
    return out


_AUDIT_MEMO: dict[tuple[Any, ...], dict[str, Any]] = {}


def _audit_cache_key(include_assemble: bool, roadmap: Path | None) -> tuple[Any, ...]:
    return (
        head_commit(),
        include_assemble,
        str(roadmap) if roadmap else "",
        index_client.code_digest(),
    )


def audit(
    *,
    view: SourceView | None = None,
    include_assemble: bool = False,
    roadmap: Path | None = None,
) -> dict[str, Any]:
    view = view or SourceView()
    if not view.overlay:
        key = _audit_cache_key(include_assemble, roadmap)
        memo = _AUDIT_MEMO.get(key)
        if memo is not None:
            return memo
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        disk = index_client.artifact_session_dir() / f"graph-{digest}.json"
        lockp = disk.with_suffix(".lock")
        with open(lockp, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            memo = _AUDIT_MEMO.get(key)
            if memo is None and disk.is_file():
                memo = json.loads(disk.read_text())
                _AUDIT_MEMO[key] = memo
            if memo is not None:
                return memo
            doc = _audit_uncached(
                view=view, include_assemble=include_assemble, roadmap=roadmap
            )
            tmp = disk.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc))
            tmp.replace(disk)
            _AUDIT_MEMO[key] = doc
            return doc
    return _audit_uncached(view=view, include_assemble=include_assemble, roadmap=roadmap)


def _audit_uncached(
    *,
    view: SourceView,
    include_assemble: bool,
    roadmap: Path | None,
) -> dict[str, Any]:
    parsed = parse_roadmap(roadmap)
    road_file = parsed["roadmap_path"]
    hw_cache = probe_all()
    existing = load_existing_state()
    index_meta = None
    prefetch_rels: list[str] = list(catalog.DISK_TRUTH_MODULES)
    for table in (catalog.GATES, catalog.GENES):
        for probe in table.values():
            prefetch_rels.extend(probe.get("code_paths") or [])
    view.prefetch(prefetch_rels)
    try:
        dump = index_client.warmup(view)
        if dump:
            index_meta = {
                "schema": dump.get("schema"),
                "backend": "hawking-index-query python-facts",
                "files_indexed": dump.get("file_count"),
                "bin": dump.get("bin"),
                "commit": dump.get("commit"),
            }
    except FileNotFoundError:
        index_meta = {"backend": "ast", "reason": "hawking-index-query binary not built"}
    if not index_meta or index_meta.get("backend") == "ast":
        reach.prefetch_catalog(view, [catalog.GATES, catalog.GENES])

    gate_unique = reach.unique_code_paths(catalog.GATES)
    gene_unique = reach.unique_code_paths(catalog.GENES)

    gates: dict[str, dict[str, Any]] = {}
    for name, skeleton in parsed["gates"].items():
        probe_row = catalog.GATES.get(name)
        if probe_row is None:
            raise KeyError(f"APPENDIX O gate {name} has no catalog probe (auditor cannot invent a look-up)")
        look = _look_up(view, probe_row, unique_paths=gate_unique)
        gates[name] = _fill_entry(
            entry_id=name,
            skeleton=skeleton,
            probe=probe_row,
            look=look,
            hw_cache=hw_cache,
            roadmap_file=road_file,
            view=view,
        )
    _apply_unreachable(gates)

    genes: dict[str, dict[str, Any]] = {}
    for gid, skeleton in parsed["genes"].items():
        probe_row = catalog.GENES.get(gid)
        if probe_row is None:
            raise KeyError(f"gene {gid} has no catalog probe")
        look = _look_up(view, probe_row, unique_paths=gene_unique)
        genes[gid] = _fill_entry(
            skeleton=skeleton,
            probe=probe_row,
            look=look,
            hw_cache=hw_cache,
            roadmap_file=road_file,
            view=view,
        )

    emit_commit = (index_meta or {}).get("commit") or head_commit()
    cite_rels: list[str] = []
    for entry in list(gates.values()) + list(genes.values()):
        for ref in _iter_citation_refs(entry):
            rel = ref.get("file")
            if rel and not str(rel).startswith("/"):
                cite_rels.append(str(rel))
    prefetch_blobs(emit_commit, cite_rels)
    bound_violations: list[str] = []
    for entry in list(gates.values()) + list(genes.values()):
        bound_violations.extend(_bind_entry_citations(entry, view, emit_commit))
    if bound_violations:
        raise AssertionError(
            "citation(s) out of bounds at emit:\n" + "\n".join(bound_violations)
        )

    status_counts = Counter(g["status"] for g in gates.values())
    gene_counts = Counter(g["status"] for g in genes.values())

    reachability = None
    if include_assemble:
        reachability = reach.assemble_snapshot(view)

    built = [
        {
            "id": g["id"],
            "runtime_caller": g["runtime_caller"][:3],
        }
        for g in gates.values()
        if g["status"] == "BUILT"
    ]

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Machine-readable capability graph for H-ROADMAP.md. Status is derived "
            "from callers/tests/receipts/hardware probes, never self-report. "
            "wired and accepted are orthogonal facts; BUILT requires both."
        ),
        "law": (
            "A capability nothing calls does not exist. Grep for call sites of the "
            "implementing symbol, not module imports, not definitions. Importing a "
            "module is not calling a capability. A non-test caller is wired, not "
            "accepted. BUILT requires wired AND accepted. wired alone is WIRED."
        ),
        "evidence_tier": EVIDENCE_TIER,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "tools/roadmap/auditor.py",
        "generated_from_commit": emit_commit,
        "roadmap": {
            "path": parsed["roadmap_path"],
            "hash": parsed["roadmap_hash"],
            "line_count": parsed["roadmap_line_count"],
            "existing_state": "civilization/ROADMAP_STATE.json" if existing else None,
            "existing_program_count": len((existing or {}).get("program_statuses") or {}),
        },
        "counts": {
            "gates": len(gates),
            "genes": len(genes),
            "gates_by_status": dict(status_counts),
            "genes_by_status": dict(gene_counts),
            "built_gates": len(built),
            "wired_gates": sum(1 for g in gates.values() if (g.get("wired") or {}).get("value")),
            "accepted_gates": sum(1 for g in gates.values() if (g.get("accepted") or {}).get("value")),
        },
        "gates": gates,
        "genes": genes,
        "disk_truth_modules": _credit_disk_truth(view),
        "verified_absent": _verify_absent_claims(view),
        "hardware_probes": hw_cache,
        "built_gates": built,
        "reachability_snapshot": reachability,
        "index": index_meta,
        "method": (
            "STATIC source analysis. Definitions via git cat-file; callers via "
            "hawking-index python-facts (schema hawking.index.python_facts.v1) "
            "built once from HEAD blobs (sparse-checkout safe), falling back to "
            "tools.future.capability_reachability AST Call/subprocess helpers when "
            "the index binary is absent. kind=import never justifies "
            "wired (SCAFFOLDED at most). Name-only matches are weak_signal and never "
            "move status. Subprocess counts only an exact CLI path, not a suffix of "
            "another tree. Receipts are citations only unless a numeric acceptance "
            "spec compares the measured value against the stated bar. Hardware "
            "presence is an inventory probe, not a performance measurement. "
            "BUILT requires wired AND accepted; wired alone is WIRED. Citations "
            "are bound to the emitting commit and re-resolved if a line is past EOF."
        ),
    }
    return doc


def write_graph(doc: dict[str, Any], dest: Path | None = None) -> Path:
    dest = dest or (REPO / GRAPH_REL)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2) + "\n")
    return dest
