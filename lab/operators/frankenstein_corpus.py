#!/usr/bin/env python3.12
"""PROTO_FRANKENSTEIN_V0 bounded L0/L1 real-problem corpus + disjoint memberships.

Assembles TRAIN / CALIBRATION / PUBLIC_TEST / HIDDEN_TEST / RETENTION from local
verified sources only (ramanujan Lean corpora, thesis coding, support-halo,
expert-iteration fixture).  NO synthetic Gaussian activations.

L0 = 32 (pipeline smoke) ⊂ L1 = 128 (first full train ladder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.operators.frankenstein_trace_format import (
    MEMBERSHIP_SPLITS,
    MembershipManager,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
CORPUS_DIR = EVIDENCE_ROOT / "corpus"
DEFAULT_L0_PATH = CORPUS_DIR / "PROTO_FRANKENSTEIN_V0_L0_CORPUS.jsonl"
DEFAULT_L1_PATH = CORPUS_DIR / "PROTO_FRANKENSTEIN_V0_L1_CORPUS.jsonl"
DEFAULT_MEMBERSHIP_PATH = CORPUS_DIR / "PROTO_FRANKENSTEIN_V0_MEMBERSHIP.json"
DEFAULT_INDEX_PATH = CORPUS_DIR / "PROTO_FRANKENSTEIN_V0_CORPUS_INDEX.json"

CORPUS_SCHEMA = "hawking.frankenstein.proto_v0_corpus.v1"
MEMBERSHIP_FREEZE_SCHEMA = "hawking.frankenstein.proto_v0_membership.v1"
INDEX_SCHEMA = "hawking.frankenstein.proto_v0_corpus_index.v1"

L0_SIZE = 32
L1_SIZE = 128

# Capability families required by PROTO_FRANKENSTEIN_V0 steer.
CAPABILITY_FAMILIES: tuple[str, ...] = (
    "math_method",
    "multi_step",
    "formalization",
    "proof_repair",
    "counterexample",
    "symbolic",
    "coding",
    "repo",
    "tools",
    "agent",
    "long_ctx",
    "general",
)

# Retention split is base-capability preservation (non-math-primary).
RETENTION_FAMILIES: frozenset[str] = frozenset(
    {"coding", "repo", "tools", "agent", "long_ctx", "general"}
)

# Local real sources (paths relative to REPO_ROOT).
RAMANUJAN_CORPORA = REPO_ROOT / "ramanujan" / "scaffold" / "data" / "corpora"
EXPERT_MATH_FIXTURE = (
    REPO_ROOT / "ramanujan" / "scaffold" / "fixtures" / "expert_iteration_math.json"
)
THESIS_SMOKE = REPO_ROOT / "tools" / "eval" / "thesis_smoke_corpus_v0.jsonl"
THESIS_RUST = REPO_ROOT / "tools" / "eval" / "thesis_rust_corpus_v0.jsonl"
SUPPORT_HALO = (
    REPO_ROOT
    / "workspace"
    / "campaign"
    / "governance"
    / "odyssey"
    / "program"
    / "evaluation"
    / "support_halo_corpus_v0.jsonl"
)

# Deterministic seed for ladder selection (content-hash based, not RNG draws).
SELECTION_SEED = "proto-frankenstein-v0-corpus/2026-08-05"

# Target per-family floors for L1 (sum >= 128; overflow distributed).
L1_FAMILY_TARGETS: dict[str, int] = {
    "math_method": 12,
    "multi_step": 12,
    "formalization": 14,
    "proof_repair": 12,
    "counterexample": 10,
    "symbolic": 8,
    "coding": 14,
    "repo": 8,
    "tools": 10,
    "agent": 8,
    "long_ctx": 6,
    "general": 14,
}

# L0: at least 2 per family where available (32 total).
L0_FAMILY_TARGETS: dict[str, int] = {
    "math_method": 3,
    "multi_step": 3,
    "formalization": 3,
    "proof_repair": 3,
    "counterexample": 2,
    "symbolic": 2,
    "coding": 4,
    "repo": 2,
    "tools": 3,
    "agent": 2,
    "long_ctx": 2,
    "general": 3,
}

# Split mix for non-retention items (fractions of non-retention budget).
# Retention is filled separately from RETENTION_FAMILIES pool.
SPLIT_MIX: tuple[tuple[str, float], ...] = (
    ("train", 0.55),
    ("calibration", 0.15),
    ("public_test", 0.15),
    ("hidden_test", 0.15),
)


class CorpusError(RuntimeError):
    """Corpus assembly failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
            raise CorpusError(f"not a regular file: {path}")
        if path.read_bytes() == encoded:
            return
        # Corpus freezes are rewritable during scaffold development; overwrite
        # only when content changes (not a sealed immutable evidence gate).
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_text(path, raw)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise CorpusError(f"missing source: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"bad jsonl {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusError(f"non-object row {path}:{line_no}")
            yield row


def _rank_key(example_id: str, family: str) -> str:
    """Stable selection rank (lower = preferred within family)."""

    return _sha256_text(f"{SELECTION_SEED}|{family}|{example_id}")


def _make_problem(
    *,
    example_id: str,
    family: str,
    surface_text: str,
    answer: str | None,
    source: str,
    source_id: str,
    verified: bool,
    verification_kind: str,
    meta: Mapping[str, Any] | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    if family not in CAPABILITY_FAMILIES:
        raise CorpusError(f"unknown family {family!r}")
    if not surface_text or not surface_text.strip():
        raise CorpusError(f"empty surface for {example_id}")
    surface = surface_text.strip()
    ch = content_hash or _sha256_text(surface)
    return {
        "example_id": example_id,
        "family": family,
        "surface_text": surface,
        "byte_length": len(surface.encode("utf-8")),
        "answer": answer,
        "source": source,
        "source_id": source_id,
        "content_hash": ch,
        "verified": bool(verified),
        "verification_kind": verification_kind,
        "synthetic_gaussian": False,
        "meta": dict(meta or {}),
    }


# ---------------------------------------------------------------------------
# Source loaders (real local only)
# ---------------------------------------------------------------------------


def load_ramanujan_d1(limit: int = 400) -> list[dict[str, Any]]:
    """Formalization / multi-step from Mathlib theorem+proof pairs."""

    out: list[dict[str, Any]] = []
    path = RAMANUJAN_CORPORA / "d1_proof_traces.jsonl"
    for row in _read_jsonl(path):
        if not row.get("admitted", True):
            continue
        name = row.get("name") or row.get("id")
        statement = row.get("statement") or row.get("signature") or ""
        proof = row.get("proof") or ""
        surface = (
            f"Formalize and prove the following Lean 4 statement.\n\n"
            f"Name: {name}\n"
            f"Statement:\n{statement}\n\n"
            f"Provide a valid proof term or tactic script."
        )
        family = "formalization" if (len(out) % 2 == 0) else "multi_step"
        out.append(
            _make_problem(
                example_id=f"pfv0:{row['id']}",
                family=family,
                surface_text=surface,
                answer=str(proof) if proof else None,
                source="ramanujan.d1_proof_traces",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="mathlib_source_theorem_proof_pair",
                content_hash=row.get("content_hash"),
                meta={
                    "module": row.get("module"),
                    "kind": row.get("kind"),
                    "file": row.get("file"),
                },
            )
        )
        if len(out) >= limit:
            break
    return out


def load_ramanujan_d2(limit: int = 200) -> list[dict[str, Any]]:
    """Multi-step tactic transitions."""

    out: list[dict[str, Any]] = []
    path = RAMANUJAN_CORPORA / "d2_state_transitions.jsonl"
    for row in _read_jsonl(path):
        if not row.get("admitted", True):
            continue
        before = row.get("state_before") or {}
        after = row.get("state_after") or {}
        tactic = row.get("tactic") or ""
        goal = before.get("goal") or ""
        surface = (
            f"Given the Lean proof state goal:\n{goal}\n\n"
            f"What single tactic advances the proof? "
            f"State the next tactic and the resulting goal sketch."
        )
        out.append(
            _make_problem(
                example_id=f"pfv0:{row['id']}",
                family="multi_step",
                surface_text=surface,
                answer=str(tactic),
                source="ramanujan.d2_state_transitions",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="mathlib_source_tactic_sequence_transitions",
                content_hash=row.get("content_hash"),
                meta={
                    "theorem": row.get("theorem"),
                    "closed_after": after.get("closed"),
                },
            )
        )
        if len(out) >= limit:
            break
    return out


def load_ramanujan_d3(limit: int = 200) -> list[dict[str, Any]]:
    """Method/premise selection for formal goals."""

    out: list[dict[str, Any]] = []
    path = RAMANUJAN_CORPORA / "d3_premise_pairs.jsonl"
    for row in _read_jsonl(path):
        if not row.get("admitted", True):
            continue
        pos = list(row.get("positive_premises") or [])
        neg = list(row.get("negative_premises") or [])
        goal = row.get("goal") or row.get("text") or ""
        surface = (
            f"Select useful premises for proving the goal (method selection).\n\n"
            f"Goal:\n{goal}\n\n"
            f"Candidate premises (mixed useful/distractor):\n"
            + "\n".join(f"- {p}" for p in (pos + neg)[:12])
            + "\n\nReturn the useful premises only."
        )
        out.append(
            _make_problem(
                example_id=f"pfv0:{row['id']}",
                family="math_method",
                surface_text=surface,
                answer=", ".join(pos) if pos else None,
                source="ramanujan.d3_premise_pairs",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="mathlib_premise_pos_neg_pairs",
                content_hash=row.get("content_hash"),
                meta={"positive_premises": pos, "negative_premises": neg},
            )
        )
        if len(out) >= limit:
            break
    return out


def load_ramanujan_d4(limit: int = 200) -> list[dict[str, Any]]:
    """Proof repair from real Lean errors."""

    out: list[dict[str, Any]] = []
    path = RAMANUJAN_CORPORA / "d4_repair_pairs.jsonl"
    for row in _read_jsonl(path):
        if not row.get("admitted", True):
            continue
        broken = row.get("broken_proof") or ""
        fixed = row.get("fixed_proof") or ""
        error = row.get("error") or ""
        sig = row.get("signature") or ""
        surface = (
            f"Repair the broken Lean proof.\n\n"
            f"Signature:\n{sig}\n\n"
            f"Broken proof:\n{broken}\n\n"
            f"Lean error:\n{error}\n\n"
            f"Provide a corrected proof."
        )
        out.append(
            _make_problem(
                example_id=f"pfv0:{row['id']}",
                family="proof_repair",
                surface_text=surface,
                answer=str(fixed),
                source="ramanujan.d4_repair_pairs",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="perturb_valid_proof_capture_real_lean_error",
                content_hash=row.get("content_hash"),
                meta={"lean_returncode": row.get("lean_returncode")},
            )
        )
        if len(out) >= limit:
            break
    return out


def load_ramanujan_d6(limit: int = 120) -> list[dict[str, Any]]:
    """Counterexamples / refutations."""

    out: list[dict[str, Any]] = []
    path = RAMANUJAN_CORPORA / "d6_counterexamples.jsonl"
    for row in _read_jsonl(path):
        if not row.get("admitted", True):
            continue
        false_stmt = row.get("false_statement") or row.get("text") or ""
        refutation = row.get("refutation_sketch") or row.get("witness") or ""
        surface = (
            f"The following claim is false or a known failed conjecture. "
            f"Provide a counterexample or refutation sketch.\n\n"
            f"Claim:\n{false_stmt}"
        )
        out.append(
            _make_problem(
                example_id=f"pfv0:{row['id']}",
                family="counterexample",
                surface_text=surface,
                answer=str(refutation)[:2000] if refutation else None,
                source="ramanujan.d6_counterexamples",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="mathlib_counterexamples_plus_enumerative_witnesses",
                content_hash=row.get("content_hash"),
                meta={"kind": row.get("kind"), "module": row.get("module")},
            )
        )
        if len(out) >= limit:
            break
    return out


def load_ramanujan_d7(limit: int = 86) -> list[dict[str, Any]]:
    """Tool / formal action traces from search harness."""

    out: list[dict[str, Any]] = []
    path = RAMANUJAN_CORPORA / "d7_tool_use_traces.jsonl"
    for row in _read_jsonl(path):
        if not row.get("admitted", True):
            continue
        tool = row.get("tool_name") or row.get("tool") or "tactic"
        action = row.get("action") or ""
        problem = row.get("problem") or row.get("id")
        text = row.get("text") or ""
        surface = (
            f"You are solving a formal proof search problem `{problem}`.\n"
            f"Emit the next tool action.\n\n"
            f"Context:\n{text}\n\n"
            f"Preferred tool family: {tool}."
        )
        # Alternate tools / agent framing for coverage.
        family = "tools" if (len(out) % 3 != 0) else "agent"
        out.append(
            _make_problem(
                example_id=f"pfv0:{row['id']}",
                family=family,
                surface_text=surface,
                answer=str(action),
                source="ramanujan.d7_tool_use_traces",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="ramanujan_search_harness_tool_trace",
                content_hash=row.get("content_hash"),
                meta={
                    "tool_name": tool,
                    "search_found": row.get("search_found"),
                },
            )
        )
        if len(out) >= limit:
            break
    return out


def load_expert_math() -> list[dict[str, Any]]:
    """Exact-numeric symbolic / multi-step from verifier fixture."""

    if not EXPERT_MATH_FIXTURE.is_file():
        raise CorpusError(f"missing expert math fixture: {EXPERT_MATH_FIXTURE}")
    doc = json.loads(EXPERT_MATH_FIXTURE.read_text(encoding="utf-8"))
    problems = doc.get("problems") or []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(problems):
        statement = row.get("statement") or ""
        expression = row.get("expression") or ""
        correct = row.get("correct_repair") or row.get("answer")
        surface = (
            f"Solve exactly (symbolic / numeric).\n\n"
            f"Problem: {statement}\n"
            f"Expression: {expression}\n"
            f"Return the exact answer only."
        )
        family = "symbolic" if i % 2 == 0 else "multi_step"
        out.append(
            _make_problem(
                example_id=f"pfv0:expert:{row['id']}",
                family=family,
                surface_text=surface,
                answer=str(correct) if correct is not None else None,
                source="ramanujan.expert_iteration_math",
                source_id=str(row["id"]),
                verified=True,
                verification_kind="exact_numeric_verifier_fixture",
                meta={"kind": row.get("kind"), "expression": expression},
            )
        )
    return out


def load_thesis_coding() -> list[dict[str, Any]]:
    """Python + Rust coding problems with executable tests."""

    out: list[dict[str, Any]] = []
    for path, family, source in (
        (THESIS_SMOKE, "coding", "tools.eval.thesis_smoke_corpus_v0"),
        (THESIS_RUST, "repo", "tools.eval.thesis_rust_corpus_v0"),
    ):
        for row in _read_jsonl(path):
            prompt = row.get("prompt") or ""
            test = row.get("test") or ""
            surface = (
                f"{prompt}\n\n"
                f"Tests (must pass):\n```\n{test}\n```"
            )
            # Rust items are repo-flavored; python stays coding.
            fam = family
            if row.get("lang") == "python" and family == "repo":
                fam = "coding"
            out.append(
                _make_problem(
                    example_id=f"pfv0:thesis:{row['id']}",
                    family=fam,
                    surface_text=surface,
                    answer=None,
                    source=source,
                    source_id=str(row["id"]),
                    verified=True,
                    verification_kind="executable_unit_tests",
                    meta={
                        "lang": row.get("lang"),
                        "entry": row.get("entry"),
                        "test": test,
                    },
                )
            )
    return out


def _render_long_ctx(row: Mapping[str, Any]) -> str:
    """Materialize support-halo long-context needle prompts locally."""

    template = str(row.get("prompt_template") or "")
    haystack_chars = int(row.get("haystack_chars") or 3500)
    needle = str(row.get("needle") or "")
    frac = float(row.get("needle_offset_frac") or 0.5)
    # Deterministic filler (not Gaussian activations — text only).
    unit = "lorem ipsum dolor sit amet consectetur adipiscing elit. "
    repeats = (haystack_chars // len(unit)) + 2
    filler = (unit * repeats)[:haystack_chars]
    insert_at = max(0, min(len(filler), int(len(filler) * frac)))
    haystack = filler[:insert_at] + f"\n{needle}\n" + filler[insert_at:]
    if "{haystack}" in template:
        return template.replace("{haystack}", haystack)
    return f"{template}\n\n{haystack}"


def load_support_halo() -> list[dict[str, Any]]:
    """General / tools / agent / long_ctx / coding from support-halo v0."""

    dim_to_family = {
        "technical_language": "general",
        "general_reasoning": "general",
        "coding": "coding",
        "retrieval": "general",
        "tools": "tools",
        "long_context": "long_ctx",
        "self_correction": "agent",
    }
    out: list[dict[str, Any]] = []
    for row in _read_jsonl(SUPPORT_HALO):
        dim = str(row.get("dimension") or "general")
        family = dim_to_family.get(dim, "general")
        if dim == "long_context":
            surface = _render_long_ctx(row)
            answer = row.get("needle_answer") or row.get("exact")
        else:
            surface = str(row.get("prompt") or "")
            answer = row.get("exact")
            if answer is None and row.get("expect"):
                answer = " | ".join(str(x) for x in row["expect"])
        if not surface.strip():
            continue
        # Mark tools-with-args as tools; self_correction as agent already.
        if row.get("tool_name"):
            family = "tools"
        out.append(
            _make_problem(
                example_id=f"pfv0:halo:{row['id']}",
                family=family,
                surface_text=surface,
                answer=str(answer) if answer is not None else None,
                source="odyssey.support_halo_corpus_v0",
                source_id=str(row["id"]),
                verified=True,
                verification_kind=f"support_halo_oracle:{row.get('oracle', 'text')}",
                meta={
                    "dimension": dim,
                    "oracle": row.get("oracle"),
                    "lang": row.get("lang"),
                },
            )
        )
    return out


def load_all_candidates() -> list[dict[str, Any]]:
    """Union of local sources with de-duplication by content_hash."""

    buckets: list[list[dict[str, Any]]] = [
        load_ramanujan_d1(),
        load_ramanujan_d2(),
        load_ramanujan_d3(),
        load_ramanujan_d4(),
        load_ramanujan_d6(),
        load_ramanujan_d7(),
        load_expert_math(),
        load_thesis_coding(),
        load_support_halo(),
    ]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for group in buckets:
        for item in group:
            ch = item["content_hash"]
            if ch in seen:
                continue
            seen.add(ch)
            if item.get("synthetic_gaussian"):
                raise CorpusError("refusing synthetic Gaussian problem")
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Ladder selection + split freeze
# ---------------------------------------------------------------------------


def _select_by_family(
    candidates: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
    *,
    total: int,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Greedy rank selection with per-family floors, then fill remainder."""

    exclude = exclude_ids or set()
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        eid = str(c["example_id"])
        if eid in exclude:
            continue
        by_family[str(c["family"])].append(dict(c))
    for fam, rows in by_family.items():
        rows.sort(key=lambda r: _rank_key(r["example_id"], r["family"]))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def _take(fam: str, n: int) -> None:
        pool = by_family.get(fam) or []
        taken = 0
        for row in pool:
            if taken >= n:
                break
            eid = row["example_id"]
            if eid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(eid)
            taken += 1

    # Floors
    for fam, n in targets.items():
        _take(fam, n)

    # Fill to total preferring under-represented families then rank.
    if len(selected) < total:
        counts = Counter(r["family"] for r in selected)
        remainder = [
            r
            for fam_rows in by_family.values()
            for r in fam_rows
            if r["example_id"] not in selected_ids
        ]
        remainder.sort(
            key=lambda r: (
                counts[r["family"]],
                _rank_key(r["example_id"], r["family"]),
            )
        )
        for row in remainder:
            if len(selected) >= total:
                break
            selected.append(row)
            selected_ids.add(row["example_id"])
            counts[row["family"]] += 1

    if len(selected) < total:
        raise CorpusError(
            f"insufficient real candidates: need {total}, got {len(selected)} "
            f"(families available: { {f: len(v) for f, v in by_family.items()} })"
        )
    return selected[:total]


def _assign_splits(items: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Disjoint split assignment with stratified math + retention.

    - Math-primary families are distributed across train/calib/public/hidden
      (so held-out math is real, not train-only).
    - Retention holds a base-capability slice (coding/repo/tools/agent/long_ctx/
      general) and never receives math-primary items.
    - Leftover retention-family items join the non-retention mix for mixed batches.
    """

    family_of: dict[str, str] = {}
    retention_pool: list[str] = []
    math_pool: list[str] = []
    for item in items:
        eid = str(item["example_id"])
        fam = str(item["family"])
        family_of[eid] = fam
        if fam in RETENTION_FAMILIES:
            retention_pool.append(eid)
        else:
            math_pool.append(eid)

    n = len(items)
    retention_target = max(4, min(len(retention_pool), int(round(n * 0.125))))
    retention_pool.sort(key=lambda eid: _rank_key(eid, family_of[eid]))
    math_pool.sort(key=lambda eid: _rank_key(eid, family_of[eid]))
    retention_ids = set(retention_pool[:retention_target])

    # Non-retention: all math + leftover retention-family items.
    remaining = [eid for eid in math_pool] + [
        eid for eid in retention_pool if eid not in retention_ids
    ]
    # Stratify: round-robin by family so each split sees math + secondary.
    by_fam: dict[str, list[str]] = defaultdict(list)
    for eid in remaining:
        by_fam[family_of[eid]].append(eid)
    for fam in by_fam:
        by_fam[fam].sort(key=lambda eid: _rank_key(eid, fam))

    # Build interleaved order: cycle families, pop head each time.
    fam_order = sorted(by_fam.keys(), key=lambda f: _rank_key(f, "family-order"))
    interleaved: list[str] = []
    while any(by_fam[f] for f in fam_order):
        for f in fam_order:
            if by_fam[f]:
                interleaved.append(by_fam[f].pop(0))

    n_rem = len(interleaved)
    quotas: dict[str, int] = {}
    allocated = 0
    for i, (split, frac) in enumerate(SPLIT_MIX):
        if i == len(SPLIT_MIX) - 1:
            quotas[split] = n_rem - allocated
        else:
            q = int(round(n_rem * frac))
            quotas[split] = q
            allocated += q
    drift = n_rem - sum(quotas.values())
    quotas["train"] = quotas.get("train", 0) + drift

    assignments: dict[str, str] = {eid: "retention" for eid in retention_ids}
    cursor = 0
    for split, _frac in SPLIT_MIX:
        q = quotas.get(split, 0)
        for eid in interleaved[cursor : cursor + q]:
            assignments[eid] = split
        cursor += q

    if len(assignments) != n:
        raise CorpusError("split assignment count mismatch")
    if set(assignments.values()) - set(MEMBERSHIP_SPLITS):
        raise CorpusError("unknown split emitted")
    # Integrity: retention must not hold math-primary families.
    for eid, split in assignments.items():
        if split == "retention" and family_of[eid] not in RETENTION_FAMILIES:
            raise CorpusError(
                f"math-primary {family_of[eid]!r} assigned to retention ({eid})"
            )
    return assignments


def build_ladder(
    *,
    l0_size: int = L0_SIZE,
    l1_size: int = L1_SIZE,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build L0 ⊂ L1 ladder with frozen disjoint memberships."""

    if l0_size > l1_size:
        raise CorpusError("L0 must be <= L1")
    pool = list(candidates) if candidates is not None else load_all_candidates()
    if any(c.get("synthetic_gaussian") for c in pool):
        raise CorpusError("NO synthetic Gaussian allowed")

    l1 = _select_by_family(pool, L1_FAMILY_TARGETS, total=l1_size)
    l1_ids = {r["example_id"] for r in l1}
    # L0 is a subset of L1 with its own family floors.
    l0 = _select_by_family(l1, L0_FAMILY_TARGETS, total=l0_size)
    l0_ids = {r["example_id"] for r in l0}
    if not l0_ids.issubset(l1_ids):
        raise CorpusError("L0 must be subset of L1")

    assignments = _assign_splits(l1)
    mgr = MembershipManager()
    for eid, split in assignments.items():
        mgr.assign(eid, split)

    # Attach membership + ladder tags.
    l1_rows: list[dict[str, Any]] = []
    for row in l1:
        r = dict(row)
        r["membership"] = assignments[row["example_id"]]
        r["ladder"] = ["L1"] + (["L0"] if row["example_id"] in l0_ids else [])
        r["schema"] = CORPUS_SCHEMA
        l1_rows.append(r)
    l0_rows = [r for r in l1_rows if r["example_id"] in l0_ids]

    membership_doc = mgr.seal_document()
    # Upgrade membership freeze document with V0-specific fields.
    membership_freeze = seal(
        {
            "schema": MEMBERSHIP_FREEZE_SCHEMA,
            "recorded_at": _utc_now(),
            "membership_schema": membership_doc["schema"],
            "membership_seal_sha256": membership_doc["seal_sha256"],
            "splits": list(MEMBERSHIP_SPLITS),
            "counts": membership_doc["counts"],
            "assignments": membership_doc["assignments"],
            "disjoint": True,
            "ladder": {
                "L0": sorted(l0_ids),
                "L1": sorted(l1_ids),
                "L0_size": len(l0_ids),
                "L1_size": len(l1_ids),
                "L0_subset_of_L1": True,
            },
            "capability_families": list(CAPABILITY_FAMILIES),
            "retention_families": sorted(RETENTION_FAMILIES),
            "policy": {
                "no_synthetic_gaussian": True,
                "real_local_sources_only": True,
                "disjoint_memberships": list(MEMBERSHIP_SPLITS),
            },
            "fabricated": False,
        }
    )

    family_counts_l0 = Counter(r["family"] for r in l0_rows)
    family_counts_l1 = Counter(r["family"] for r in l1_rows)
    source_counts_l1 = Counter(r["source"] for r in l1_rows)
    split_by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in l1_rows:
        split_by_family[r["membership"]][r["family"]] += 1

    index = seal(
        {
            "schema": INDEX_SCHEMA,
            "recorded_at": _utc_now(),
            "L0_size": len(l0_rows),
            "L1_size": len(l1_rows),
            "L0_subset_of_L1": True,
            "family_counts_L0": dict(sorted(family_counts_l0.items())),
            "family_counts_L1": dict(sorted(family_counts_l1.items())),
            "source_counts_L1": dict(sorted(source_counts_l1.items())),
            "split_counts": membership_doc["counts"],
            "split_by_family": {
                s: dict(sorted(v.items())) for s, v in sorted(split_by_family.items())
            },
            "capability_families_covered_L0": sorted(family_counts_l0),
            "capability_families_covered_L1": sorted(family_counts_l1),
            "all_families_required": list(CAPABILITY_FAMILIES),
            "missing_families_L1": [
                f for f in CAPABILITY_FAMILIES if family_counts_l1.get(f, 0) == 0
            ],
            "synthetic_gaussian": False,
            "verified_only": all(r.get("verified") for r in l1_rows),
            "membership_freeze_seal_sha256": membership_freeze["seal_sha256"],
            "fabricated": False,
            "claim_boundary": {
                "activations_captured": False,
                "correspondence_numbers": False,
                "transfer_trained": False,
                "corpus_only": True,
            },
        }
    )

    return {
        "L0": l0_rows,
        "L1": l1_rows,
        "membership": membership_freeze,
        "index": index,
        "membership_manager": mgr,
    }


def write_corpus_artifacts(
    ladder: Mapping[str, Any],
    *,
    out_dir: Path | None = None,
) -> dict[str, str]:
    """Write L0/L1 jsonl + membership + index under evidence/corpus/."""

    base = Path(out_dir) if out_dir is not None else CORPUS_DIR
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "L0": base / DEFAULT_L0_PATH.name,
        "L1": base / DEFAULT_L1_PATH.name,
        "membership": base / DEFAULT_MEMBERSHIP_PATH.name,
        "index": base / DEFAULT_INDEX_PATH.name,
    }

    def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
        lines = [
            json.dumps(dict(r), sort_keys=True, ensure_ascii=False, allow_nan=False)
            for r in rows
        ]
        text = "\n".join(lines) + ("\n" if lines else "")
        _atomic_write_text(path, text)
        return _sha256_bytes(text.encode("utf-8"))

    def _rel_or_abs(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p.resolve())

    l0_hash = _write_jsonl(paths["L0"], ladder["L0"])
    l1_hash = _write_jsonl(paths["L1"], ladder["L1"])
    _atomic_write_json(paths["membership"], ladder["membership"])
    index = dict(ladder["index"])
    index["artifacts"] = {
        "L0_jsonl": _rel_or_abs(paths["L0"]),
        "L1_jsonl": _rel_or_abs(paths["L1"]),
        "membership_json": _rel_or_abs(paths["membership"]),
        "L0_sha256": l0_hash,
        "L1_sha256": l1_hash,
    }
    # Re-seal index with artifact digests.
    index.pop("seal_sha256", None)
    index = seal(index)
    verify(index, label="corpus index")
    _atomic_write_json(paths["index"], index)

    return {k: str(v) for k, v in paths.items()}


def assemble_and_write(
    *,
    out_dir: Path | None = None,
    l0_size: int = L0_SIZE,
    l1_size: int = L1_SIZE,
) -> dict[str, Any]:
    ladder = build_ladder(l0_size=l0_size, l1_size=l1_size)
    paths = write_corpus_artifacts(ladder, out_dir=out_dir)
    return {
        "status": "OK",
        "L0_size": len(ladder["L0"]),
        "L1_size": len(ladder["L1"]),
        "split_counts": ladder["membership"]["counts"],
        "family_counts_L1": ladder["index"]["family_counts_L1"],
        "source_counts_L1": ladder["index"]["source_counts_L1"],
        "missing_families_L1": ladder["index"]["missing_families_L1"],
        "membership_seal_sha256": ladder["membership"]["seal_sha256"],
        "index_seal_sha256": ladder["index"]["seal_sha256"],
        "paths": paths,
        "synthetic_gaussian": False,
        "fabricated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble PROTO_FRANKENSTEIN_V0 L0/L1 corpus")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--l0", type=int, default=L0_SIZE)
    p.add_argument("--l1", type=int, default=L1_SIZE)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = assemble_and_write(out_dir=args.out_dir, l0_size=args.l0, l1_size=args.l1)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
