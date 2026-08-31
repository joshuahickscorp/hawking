"""TEACHER_CORPUS_EXPANSION — rank the next Flash captures by diversity gain, not size.

One real corpus already exists: 256 unique layer-4 mlp_input rows, route union
117, every hash distinct. The screen that consumed it named the remaining
surfaces and asked for broader traces. This module plans those captures so the
expensive ones are the informative ones, validates a captured receipt against
the diversity contract, and reports overlap with the rows already in hand.

It does not run a capture, take a GPU lease, or invent a surface name. Row
admission (fabrication vs honest-thin) stays in teacher_corpus.validate_corpus;
this module extends the PLAN. The capture binary itself refuses fewer than 256
unique rows — plans do not design around that refusal.

    python3 tools/future/teacher_corpus_expansion.py --build
    python3 -m pytest tools/future/test_teacher_corpus_expansion.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, load_json, write_receipt
from tools.future.meta_funnel import GATES
from tools.future.teacher_corpus import BOUNDED_TARGET_ROWS, CAPABILITY_DOMAINS, FLASH_SPECIMEN

RECEIPT = "TEACHER_CORPUS_EXPANSION.json"
SCHEMA = "hawking.future.teacher_corpus_expansion.v1"
RECORDED_BY = "tools/future/teacher_corpus_expansion.py"

REAL_CORPUS_REL = "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL256.json"
SCREEN_REL = "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL256.json"

# flash_meta_teacher_trace.rs — recovered, not invented.
FLASH_MIN_ROWS = 256  # MIN_CORPUS_ROWS; equals teacher_corpus.BOUNDED_TARGET_ROWS
FLASH_MAX_ROWS = 4096
FLASH_MIN_ROUTE_UNION = 2
FLASH_MIN_TOPK_SETS = 2
FLASH_LAYER_COUNT_CITED = 48  # layer-3 teacher receipt "persistent 48-layer session"
EXISTING_TENSOR = "model.language_model.layers.4.mlp_input"
EXISTING_LAYER = 4
EXISTING_ORGAN = "layer_4.routed_experts.gate_up_proj"
PINNED_REVISION = FLASH_SPECIMEN["pinned_revision"]

# Calibration cited from the one real capture. Scaled figures are ESTIMATE.
CALIBRATION_ROWS = 256
CALIBRATION_LAYER = 4
CALIBRATION_ELAPSED = "~25 min"
CALIBRATION_NOTE = (
    "ESTIMATE scaled from the one real measurement: 256 rows at layer 4 took "
    "~25 min of dense source-BF16 execution (FLASH_META_TEACHER_L4_REAL256.json). "
    "This sidecar did not re-measure it."
)

# Funnel required_input -> a name the screen already used. Never the reverse
# (a funnel input is not a license to mint a surface the screen did not name).
FUNNEL_INPUT_TO_SURFACE = {
    "teacher_corpus": "hidden",
    "held_out_numerical": "hidden",
    "route_traces": "router",
    "logit_token": "terminal-logit",
}

_DISTILL_SLASH = re.compile(
    r"(?:distill|add)\s+([a-z0-9][a-z0-9\-]+(?:/[a-z0-9][a-z0-9\-]+)+)\s+surfaces",
    re.I,
)

# mlp_input is the screen's "hidden-state surface". Other tensors are not
# silently credited as coverage of a named distillation surface.
TENSOR_TO_CANONICAL = {
    EXISTING_TENSOR: "hidden",
}


class ExpansionRefused(ValueError):
    """Loud refusal: absent input, unknown surface, redundant capture, or a
    corpus that fails a diversity axis the contract requires.
    """

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result
        self.codes = list(result.get("refusals") or [])


class RedundantCapture(ExpansionRefused):
    """A plan that would recapture a surface the existing corpus already covers."""


class UnknownSurface(ExpansionRefused):
    """A surface name that is not in the screen's next_gate list."""


class EvidenceAbsent(ExpansionRefused):
    """A required receipt was not visible. Absence is a refusal, not a default."""


# ---------------------------------------------------------------------------
# Loaders. Sparse checkout is not absence; git HEAD is authority.
# ---------------------------------------------------------------------------


def load_rel(rel: str) -> dict[str, Any]:
    """Load a repo-relative JSON receipt. Fail closed if it is not visible."""
    path = REPO / rel
    if path.is_file():
        return load_json(path)
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        raise EvidenceAbsent(
            f"REFUSED: evidence not visible ({rel}); sparse checkout is not absence "
            "and a missing file is not a default corpus",
            {"accepted": False, "refusals": ["EVIDENCE_ABSENT"], "rel": rel},
        )
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvidenceAbsent(
            f"REFUSED: evidence at {rel} is not JSON ({exc})",
            {"accepted": False, "refusals": ["EVIDENCE_UNPARSEABLE"], "rel": rel},
        ) from exc
    if not isinstance(doc, dict):
        raise EvidenceAbsent(
            f"REFUSED: evidence at {rel} is not an object",
            {"accepted": False, "refusals": ["EVIDENCE_NOT_OBJECT"], "rel": rel},
        )
    return doc


def load_real_corpus() -> dict[str, Any]:
    return load_rel(REAL_CORPUS_REL)


def load_screen() -> dict[str, Any]:
    return load_rel(SCREEN_REL)


def try_load_rel(rel: str) -> dict[str, Any] | None:
    try:
        return load_rel(rel)
    except EvidenceAbsent:
        return None


# ---------------------------------------------------------------------------
# Surface names. Parsed from the screen, never invented.
# ---------------------------------------------------------------------------


def parse_slash_surfaces(text: str) -> tuple[str, ...]:
    """Recover the slash-separated distillation list the screen actually wrote."""
    if not text:
        return ()
    m = _DISTILL_SLASH.search(text)
    if not m:
        return ()
    parts = tuple(p.strip().lower() for p in m.group(1).split("/") if p.strip())
    # Dedup preserving order. A screen that repeats a name is still one surface.
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def declared_surfaces(screen: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The only surface names this module will plan. Empty is a refusal later."""
    doc = screen if screen is not None else try_load_rel(SCREEN_REL)
    if not isinstance(doc, dict):
        return ()
    names = parse_slash_surfaces(str(doc.get("next_gate") or ""))
    if names:
        return names
    # The capture receipt names the same list with "add … surfaces".
    return ()


def canonical_surface_of(corpus: Mapping[str, Any]) -> str | None:
    """Map a captured tensor name onto a screen surface. Unmapped is None."""
    trace = corpus.get("teacher_trace") if isinstance(corpus.get("teacher_trace"), dict) else {}
    tensor = str(trace.get("surface") or corpus.get("surface") or "")
    if tensor in TENSOR_TO_CANONICAL:
        return TENSOR_TO_CANONICAL[tensor]
    lowered = tensor.strip().lower()
    declared = declared_surfaces()
    if lowered in declared:
        return lowered
    return None


def _token_ids_of(corpus: Mapping[str, Any]) -> list[int]:
    raw = corpus.get("token_ids")
    if isinstance(raw, list) and raw:
        try:
            return [int(x) for x in raw]
        except (TypeError, ValueError):
            return []
    rows = corpus.get("rows")
    if isinstance(rows, list):
        out: list[int] = []
        for row in rows:
            if isinstance(row, dict) and "token_id" in row:
                try:
                    out.append(int(row["token_id"]))
                except (TypeError, ValueError):
                    return []
        return out
    return []


def _rows_of(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = corpus.get("rows")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def _row_hash(row: Mapping[str, Any]) -> str | None:
    for key in (
        "layer4_mlp_input_sha256",
        "surface_sha256",
        "row_sha256",
        "content_sha256",
    ):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _route_ids_of(row: Mapping[str, Any]) -> list[int]:
    raw = row.get("route_ids")
    if not isinstance(raw, list):
        return []
    try:
        return [int(x) for x in raw]
    except (TypeError, ValueError):
        return []


def _route_union_of(corpus: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[int]:
    audit = corpus.get("route_audit") if isinstance(corpus.get("route_audit"), dict) else {}
    declared = audit.get("route_union")
    computed: set[int] = set()
    for row in rows:
        computed.update(_route_ids_of(row))
    if isinstance(declared, list) and declared:
        try:
            declared_ids = [int(x) for x in declared]
        except (TypeError, ValueError):
            declared_ids = []
        if declared_ids and computed and set(declared_ids) != computed:
            return []  # disagreement is a later refusal, not a guessed union
        if declared_ids:
            return sorted(set(declared_ids))
    return sorted(computed)


def coverage_of(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """What a captured receipt actually covers. Labels are not coverage."""
    rows = _rows_of(corpus)
    tokens = _token_ids_of(corpus)
    hashes = [_row_hash(r) for r in rows]
    present_hashes = [h for h in hashes if h]
    union = _route_union_of(corpus, rows)
    topk = {tuple(_route_ids_of(r)) for r in rows}
    topk.discard(())
    trace = corpus.get("teacher_trace") if isinstance(corpus.get("teacher_trace"), dict) else {}
    return {
        "schema": corpus.get("schema"),
        "canonical_surface": canonical_surface_of(corpus),
        "tensor": trace.get("surface") or corpus.get("surface"),
        "layer": trace.get("layer"),
        "organ": trace.get("organ"),
        "pinned_revision": corpus.get("pinned_revision"),
        "n_rows": len(rows) if rows else (len(tokens) or None),
        "n_token_ids": len(tokens),
        "n_unique_token_ids": len(set(tokens)),
        "n_unique_row_hashes": len(set(present_hashes)),
        "route_union_size": len(union),
        "unique_ordered_topk_sets": len(topk),
        "token_ids": tokens,
        "route_union": union,
    }


# ---------------------------------------------------------------------------
# surfaces_needed
# ---------------------------------------------------------------------------


def _funnel_bindings(screen_names: Sequence[str]) -> list[dict[str, Any]]:
    names = set(screen_names)
    rows: list[dict[str, Any]] = []
    for gate in GATES:
        mapped = FUNNEL_INPUT_TO_SURFACE.get(gate.required_input)
        if mapped is None:
            continue
        rows.append(
            {
                "gate_id": gate.id,
                "gate_name": gate.name,
                "required_input": gate.required_input,
                "surface": mapped if mapped in names else None,
                "bound": mapped in names,
                "note": (
                    None
                    if mapped in names
                    else (
                        f"funnel input {gate.required_input!r} maps to {mapped!r}, "
                        "which the screen's next_gate did not name; not added"
                    )
                ),
            }
        )
    return rows


def surfaces_needed(
    corpus: Mapping[str, Any] | None = None,
    screen: Mapping[str, Any] | None = None,
    *,
    raise_on_refuse: bool = False,
) -> dict[str, Any]:
    """Surfaces the screen named that currently have zero teacher coverage.

    Names come from the screen's next_gate (and the capture receipt's matching
    list) plus funnel required_inputs mapped onto those same names. A funnel
    input is not a new surface.
    """
    refusals: list[str] = []
    screen_doc: Mapping[str, Any] | None = screen
    corpus_doc: Mapping[str, Any] | None = corpus
    if screen_doc is None:
        try:
            screen_doc = load_screen()
        except EvidenceAbsent as exc:
            refusals.append("SCREEN_ABSENT")
            if raise_on_refuse:
                raise
            screen_doc = exc.result
    if corpus_doc is None:
        try:
            corpus_doc = load_real_corpus()
        except EvidenceAbsent as exc:
            refusals.append("CORPUS_ABSENT")
            if raise_on_refuse:
                raise
            corpus_doc = None

    next_gate = ""
    if isinstance(screen_doc, dict):
        next_gate = str(screen_doc.get("next_gate") or "")
    names = parse_slash_surfaces(next_gate)
    if not names and isinstance(corpus_doc, dict):
        names = parse_slash_surfaces(str(corpus_doc.get("next_gate") or ""))
    if not names:
        refusals.append("SURFACE_NAMES_UNPARSEABLE")
        result = {
            "accepted": False,
            "declared_from_screen": [],
            "needed": [],
            "needed_names": [],
            "covered": [],
            "refusals": refusals,
            "claim_boundary": (
                "STATIC_ONLY. No surface name was recovered from the screen; "
                "this module will not invent one."
            ),
        }
        if raise_on_refuse:
            raise ExpansionRefused(
                "REFUSED: could not recover surface names from the screen next_gate",
                result,
            )
        return result

    covered_list: list[dict[str, Any]] = []
    covered_names: set[str] = set()
    if isinstance(corpus_doc, dict) and corpus_doc.get("schema"):
        cov = coverage_of(corpus_doc)
        name = cov.get("canonical_surface")
        # Full coverage of a named surface: the binary's own admission bar,
        # plus a bound tensor. Token-window completeness is not whole-model
        # completeness; recapture of that window is still redundant.
        full = (
            name in names
            and int(cov.get("n_unique_row_hashes") or 0) >= FLASH_MIN_ROWS
            and int(cov.get("n_unique_token_ids") or 0) >= FLASH_MIN_ROWS
            and int(cov.get("route_union_size") or 0) >= FLASH_MIN_ROUTE_UNION
            and cov.get("pinned_revision") == PINNED_REVISION
        )
        if name and full:
            covered_names.add(name)
            covered_list.append(
                {
                    "surface": name,
                    "tensor": cov.get("tensor"),
                    "layer": cov.get("layer"),
                    "n_unique_row_hashes": cov.get("n_unique_row_hashes"),
                    "n_unique_token_ids": cov.get("n_unique_token_ids"),
                    "route_union_size": cov.get("route_union_size"),
                    "unique_ordered_topk_sets": cov.get("unique_ordered_topk_sets"),
                    "pinned_revision": cov.get("pinned_revision"),
                    "full_coverage_of_captured_window": True,
                    "whole_model_coverage": False,
                }
            )

    needed_names = [n for n in names if n not in covered_names]
    funnel = _funnel_bindings(names)
    needed = []
    for n in needed_names:
        gates = [
            f["gate_name"]
            for f in funnel
            if f.get("surface") == n and f.get("bound")
        ]
        needed.append(
            {
                "surface": n,
                "zero_teacher_coverage": True,
                "funnel_gates": gates,
                "why": (
                    f"screen next_gate named {n!r}; no captured corpus on this "
                    "worktree covers it"
                ),
            }
        )

    return {
        "accepted": not refusals,
        "declared_from_screen": list(names),
        "next_gate": next_gate,
        "funnel_bindings": funnel,
        "covered": covered_list,
        "needed": needed,
        "needed_names": needed_names,
        "refusals": refusals,
        "rule": (
            "Zero coverage is a missing named surface, not a small row count. "
            "hidden at layer-4 mlp_input is covered by the real 256-row corpus; "
            "recapturing it is redundant. Broader traces are a different plan."
        ),
        "claim_boundary": (
            "STATIC_ONLY planner. Naming a surface is not a capture. Coverage "
            "is read from the real corpus receipt, not from a tool list."
        ),
    }


# ---------------------------------------------------------------------------
# Wall-time ESTIMATE. Never a measurement. Never a hardware field name.
# ---------------------------------------------------------------------------


def wall_time_estimate(
    *,
    n_tokens: int,
    layer: Any,
    surface: str,
) -> dict[str, Any]:
    """Scale the one real ~25 min / 256-row / layer-4 capture. Always ESTIMATE."""
    calibration = {
        "kind": "ESTIMATE",
        "label": "ESTIMATE",
        "rows": CALIBRATION_ROWS,
        "layer": CALIBRATION_LAYER,
        "elapsed": CALIBRATION_ELAPSED,
        "source": REAL_CORPUS_REL,
        "note": CALIBRATION_NOTE,
    }
    if n_tokens < FLASH_MIN_ROWS:
        return {
            "kind": "ESTIMATE",
            "label": "ESTIMATE",
            "text": (
                "ESTIMATE refused: the binary refuses fewer than 256 unique rows; "
                "no wall-time is offered for a capture it will not run"
            ),
            "refused": True,
            "n_tokens": n_tokens,
            "layer": layer,
            "surface": surface,
            "calibration": calibration,
        }
    if n_tokens > FLASH_MAX_ROWS:
        return {
            "kind": "ESTIMATE",
            "label": "ESTIMATE",
            "text": (
                f"ESTIMATE refused: the binary refuses --count above {FLASH_MAX_ROWS}"
            ),
            "refused": True,
            "n_tokens": n_tokens,
            "layer": layer,
            "surface": surface,
            "calibration": calibration,
        }
    # Linear in token count at the measured depth. Not a new measurement.
    scaled = (25 * n_tokens) / CALIBRATION_ROWS
    same_depth = layer == EXISTING_LAYER
    if same_depth:
        text = (
            f"ESTIMATE: ~{scaled:g} min for {n_tokens} unique tokens at layer-4 "
            f"depth, scaled linearly from the one real measurement "
            f"({CALIBRATION_ELAPSED} for {CALIBRATION_ROWS} rows at layer 4)"
        )
    else:
        text = (
            f"ESTIMATE: strictly more than ~{scaled:g} min (the layer-4 linear "
            f"scale for {n_tokens} tokens) because {surface} needs depth beyond "
            "layer 4; the extra cost is NOT_MEASURED and is not claimed as a multiple"
        )
    return {
        "kind": "ESTIMATE",
        "label": "ESTIMATE",
        "text": text,
        "refused": False,
        "n_tokens": n_tokens,
        "layer": layer,
        "surface": surface,
        "calibration": calibration,
        "scale_rule": (
            "ESTIMATE: linear in unique token count at layer-4 depth; deeper "
            "surfaces cost strictly more and stay ESTIMATE; co-capture savings "
            "are NOT_MEASURED"
        ),
    }


# ---------------------------------------------------------------------------
# Diversity contract a capture must satisfy (the binary's refusals, plus axes)
# ---------------------------------------------------------------------------


def diversity_contract(surface: str) -> dict[str, Any]:
    return {
        "surface": surface,
        "min_unique_rows": FLASH_MIN_ROWS,
        "min_unique_token_ids": FLASH_MIN_ROWS,
        "max_rows": FLASH_MAX_ROWS,
        "duplicate_token_ids": "refuse",
        "min_route_union": FLASH_MIN_ROUTE_UNION,
        "min_unique_ordered_topk_sets": FLASH_MIN_TOPK_SETS,
        "min_unique_row_hashes": FLASH_MIN_ROWS,
        "pinned_revision": PINNED_REVISION,
        "capability_domain": (
            "record when the capture path has it; ABSENT is reported, never "
            f"filled in. Declared domains: {list(CAPABILITY_DOMAINS)}"
        ),
        "do_not_fabricate": (
            "The binary refuses unique mlp_input rows below 256 and duplicate "
            "token IDs. A plan that pads, copies, or synthesises rows to close "
            "that bar is waste and is refused here too. Admission of "
            "teacher_corpus-schema fabrication stays in "
            "teacher_corpus.validate_corpus."
        ),
        "row_count_is_the_least_interesting_axis": (
            "A corpus of 4096 rows that all route to the same twelve experts "
            "teaches less than 256 rows spanning 117. Rank by route coverage, "
            "token-position spread, and new surfaces."
        ),
        "validator": "tools.future.teacher_corpus_expansion.validate",
        "binary_refusals_honored": [
            "fewer than 256 unique token rows",
            "duplicate token IDs",
            "unique mlp_input rows below 256",
            "route union < 2 or unique ordered top-k sets < 2",
            f"--count above {FLASH_MAX_ROWS}",
        ],
    }


# ---------------------------------------------------------------------------
# plan(surface)
# ---------------------------------------------------------------------------


def _prior_or_real(
    prior: Mapping[str, Any] | None,
    *,
    raise_on_refuse: bool,
) -> Mapping[str, Any] | None:
    if prior is not None:
        return prior
    try:
        return load_real_corpus()
    except EvidenceAbsent:
        if raise_on_refuse:
            raise
        return None


def _redundant_result(surface: str, prior_cov: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "refused": True,
        "accepted": False,
        "reason": "REDUNDANT_SURFACE",
        "surface": surface,
        "refusals": ["REDUNDANT_SURFACE"],
        "prior_coverage": {
            "surface": prior_cov.get("canonical_surface"),
            "tensor": prior_cov.get("tensor"),
            "layer": prior_cov.get("layer"),
            "n_unique_row_hashes": prior_cov.get("n_unique_row_hashes"),
            "n_unique_token_ids": prior_cov.get("n_unique_token_ids"),
            "route_union_size": prior_cov.get("route_union_size"),
        },
        "why": (
            f"{surface} already has a captured window that meets the binary's "
            "admission bar (unique rows, unique tokens, non-degenerate routes, "
            "pinned revision). Recapture would re-derive those rows. Broader "
            "traces, if wanted, are plan_broader_traces() — a disjoint token "
            "window, not a redo."
        ),
        "wall_time_estimate": {
            "kind": "ESTIMATE",
            "label": "ESTIMATE",
            "text": "ESTIMATE not applicable: capture refused as redundant",
            "refused": True,
        },
    }


def plan(
    surface: str,
    *,
    prior: Mapping[str, Any] | None = None,
    raise_on_refuse: bool = True,
    screen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Concrete capture for one screen-named surface. Refuses redundant ones."""
    names = declared_surfaces(screen)
    if not names:
        # Still need names if the caller passed no screen and load failed.
        try:
            names = declared_surfaces(load_screen()) if screen is None else ()
        except EvidenceAbsent as exc:
            if raise_on_refuse:
                raise
            return exc.result
    if not names:
        result = {
            "refused": True,
            "accepted": False,
            "reason": "SURFACE_NAMES_UNPARSEABLE",
            "surface": surface,
            "refusals": ["SURFACE_NAMES_UNPARSEABLE"],
            "wall_time_estimate": {
                "kind": "ESTIMATE",
                "label": "ESTIMATE",
                "text": "ESTIMATE not applicable: surface list unparseable",
                "refused": True,
            },
        }
        if raise_on_refuse:
            raise ExpansionRefused(
                "REFUSED: cannot plan; screen next_gate named no surfaces",
                result,
            )
        return result

    if surface not in names:
        result = {
            "refused": True,
            "accepted": False,
            "reason": "UNKNOWN_SURFACE",
            "surface": surface,
            "refusals": ["UNKNOWN_SURFACE"],
            "declared_from_screen": list(names),
            "why": (
                f"{surface!r} is not in the screen's next_gate list {list(names)}. "
                "This module does not invent surfaces."
            ),
            "wall_time_estimate": {
                "kind": "ESTIMATE",
                "label": "ESTIMATE",
                "text": "ESTIMATE not applicable: unknown surface refused",
                "refused": True,
            },
        }
        if raise_on_refuse:
            raise UnknownSurface(
                f"REFUSED: unknown surface {surface!r}; declared={list(names)}",
                result,
            )
        return result

    prior_doc = _prior_or_real(prior, raise_on_refuse=raise_on_refuse)
    if prior_doc is None:
        result = {
            "refused": True,
            "accepted": False,
            "reason": "ABSENT_PRIOR",
            "surface": surface,
            "refusals": ["ABSENT_PRIOR"],
            "why": (
                "No prior corpus is visible, so a plan cannot know whether the "
                "surface is already covered or which token ids to join. Absence "
                "is a refusal, not a default of tokens 0..255."
            ),
            "wall_time_estimate": {
                "kind": "ESTIMATE",
                "label": "ESTIMATE",
                "text": "ESTIMATE not applicable: prior corpus absent",
                "refused": True,
            },
        }
        if raise_on_refuse:
            raise EvidenceAbsent(
                "REFUSED: plan() has no prior corpus to join or to test for redundancy",
                result,
            )
        return result

    cov = coverage_of(prior_doc)
    needed = surfaces_needed(prior_doc, screen if screen is not None else try_load_rel(SCREEN_REL))
    if surface not in needed.get("needed_names", []) and surface in {
        c.get("surface") for c in needed.get("covered") or []
    }:
        result = _redundant_result(surface, cov)
        if raise_on_refuse:
            raise RedundantCapture(
                f"REFUSED: {surface} already has full coverage of the captured window",
                result,
            )
        return result

    prior_tokens = list(cov.get("token_ids") or [])
    if len(set(prior_tokens)) < FLASH_MIN_ROWS:
        result = {
            "refused": True,
            "accepted": False,
            "reason": "PRIOR_TOO_THIN_TO_JOIN",
            "surface": surface,
            "refusals": ["PRIOR_TOO_THIN_TO_JOIN"],
            "why": (
                "The prior corpus has fewer than 256 unique tokens; joining a new "
                "surface onto it would plan a capture the binary will refuse. "
                "This module does not pad tokens to close the bar."
            ),
            "wall_time_estimate": wall_time_estimate(
                n_tokens=len(set(prior_tokens)),
                layer=EXISTING_LAYER,
                surface=surface,
            ),
        }
        if raise_on_refuse:
            raise ExpansionRefused(
                "REFUSED: prior unique tokens below the binary's 256-row bar",
                result,
            )
        return result

    # Same token ids so the new surface joins the hidden rows. New tokens
    # belong in plan_broader_traces, not in a zero-coverage surface capture.
    token_ids = prior_tokens
    n = len(token_ids)
    if surface == "terminal-logit":
        layer: Any = "lm_head"
        layers = {
            "kind": "full_prefix_to_lm_head",
            "flash_layer_count_cited": FLASH_LAYER_COUNT_CITED,
            "citation": (
                "FLASH_META_TEACHER_L4_LAYER3_REAL256.json next: persistent "
                "48-layer session; tools/future/frontiers.py FT.STATE 48-layer item"
            ),
        }
        token_strategy = (
            "Reuse the admitted unique token ids so terminal-logit rows join the "
            "existing hidden corpus on token_id. The binary requires unique ids "
            "and refuses a small probe."
        )
        diversity_gain = (
            "Zero coverage today. Funnel gate 5 (logit_token_validation) cannot "
            "run without this surface. A new tensor, not more rows of mlp_input."
        )
        current_binary = (
            "flash_meta_teacher_trace.rs emits layer-4 mlp_input, per-row "
            "route_ids, and layer-4 output hashes. It does not emit terminal "
            "logits. This plan is a contract for a capture that binary does "
            "not yet perform."
        )
    elif surface == "router":
        layer = EXISTING_LAYER
        layers = {"kind": "same_layer_as_hidden", "layer": EXISTING_LAYER, "organ": "router"}
        token_strategy = (
            "Reuse the admitted unique token ids so router logits join the "
            "existing hidden rows on token_id. Route identity is already on "
            "those rows; the missing tensor is the router logits themselves."
        )
        diversity_gain = (
            "Zero coverage of router logits (route_ids are decisions, not the "
            "router surface). Funnel gate 4 (route_stability) consumes this. "
            "The real corpus already showed 256 distinct ordered top-k sets "
            "and a 117-expert union on these tokens — the informative axis is "
            "the logit surface, not more of the same twelve experts."
        )
        current_binary = (
            "The current binary records route_ids / top-k membership. That is "
            "not router-logit coverage. Declared route_ids != executed router capture."
        )
    else:
        # routed-output (and any other zero-coverage screen name)
        layer = EXISTING_LAYER
        layers = {
            "kind": "same_layer_as_hidden",
            "layer": EXISTING_LAYER,
            "organ": EXISTING_ORGAN,
        }
        token_strategy = (
            "Reuse the admitted unique token ids so routed-expert outputs join "
            "the hidden rows. layer_4.final_state hashes are the combined layer "
            "output, not routed-output, and do not count as coverage."
        )
        diversity_gain = (
            "Zero coverage. The L4 screen failed to reconstruct gate_up_proj "
            "from the hidden corpus; the actual routed expert outputs are the "
            "surface that family was trying to predict. 117 experts already "
            "touched by the hidden rows — capturing their outputs is the gain, "
            "not 4096 more hidden rows."
        )
        current_binary = (
            "The current binary writes mlp_input plus a layer-output hash. "
            "It does not write per-expert routed outputs."
        )

    estimate = wall_time_estimate(n_tokens=n, layer=layer, surface=surface)
    return {
        "refused": False,
        "accepted": True,
        "kind": "capture_plan",
        "surface": surface,
        "n_tokens": n,
        "token_ids": token_ids,
        "token_selection_strategy": token_strategy,
        "layers": layers,
        "layer": layer,
        "expected_wall_time_estimate": estimate,
        "wall_time_estimate": estimate,
        "diversity_contract": diversity_contract(surface),
        "expected_diversity_gain": diversity_gain,
        "pinned_revision": PINNED_REVISION,
        "specimen": {
            "model": FLASH_SPECIMEN["model"],
            "pinned_revision": PINNED_REVISION,
        },
        "current_binary_emits_this_surface": False,
        "current_binary_note": current_binary,
        "cli_contract": {
            "example": "flash_meta_teacher_trace",
            "flags": ["--root", "--tokens|--token-start --count", "--out", "--state-out"],
            "min_rows": FLASH_MIN_ROWS,
            "max_rows": FLASH_MAX_ROWS,
            "refuses": [
                "fewer than 256 unique token rows",
                "duplicate token IDs",
                "unique mlp_input rows below 256",
                "degenerate route union",
                "candidate accelerator env vars enabled",
                "no Metal-capable GPU (boundary receipt, 0 rows, not a corpus)",
            ],
        },
        "executed": False,
        "gpu_authority": False,
        "claim_boundary": (
            "STATIC_ONLY plan. Not a capture. Wall time is an ESTIMATE scaled "
            "from the one real layer-4 measurement. The current binary does not "
            "emit this surface."
        ),
    }


def plan_broader_traces(
    *,
    prior: Mapping[str, Any] | None = None,
    raise_on_refuse: bool = True,
) -> dict[str, Any]:
    """The screen's 'collect broader teacher traces' — a disjoint token window.

    Not plan('hidden'): that recapture is redundant. This is a new window on
    the same tensor, aimed at route-union growth beyond 117, not at size.
    """
    prior_doc = _prior_or_real(prior, raise_on_refuse=raise_on_refuse)
    if prior_doc is None:
        result = {
            "refused": True,
            "accepted": False,
            "reason": "ABSENT_PRIOR",
            "refusals": ["ABSENT_PRIOR"],
            "wall_time_estimate": {
                "kind": "ESTIMATE",
                "label": "ESTIMATE",
                "text": "ESTIMATE not applicable: prior corpus absent",
                "refused": True,
            },
        }
        if raise_on_refuse:
            raise EvidenceAbsent("REFUSED: broader traces need the prior token window", result)
        return result
    cov = coverage_of(prior_doc)
    prior_tokens = list(cov.get("token_ids") or [])
    if not prior_tokens:
        result = {
            "refused": True,
            "accepted": False,
            "reason": "PRIOR_HAS_NO_TOKEN_IDS",
            "refusals": ["PRIOR_HAS_NO_TOKEN_IDS"],
            "wall_time_estimate": {
                "kind": "ESTIMATE",
                "label": "ESTIMATE",
                "text": "ESTIMATE not applicable: prior has no token ids",
                "refused": True,
            },
        }
        if raise_on_refuse:
            raise ExpansionRefused("REFUSED: prior corpus has no token ids to avoid", result)
        return result
    start = max(prior_tokens) + 1
    token_ids = list(range(start, start + FLASH_MIN_ROWS))
    estimate = wall_time_estimate(
        n_tokens=len(token_ids), layer=EXISTING_LAYER, surface="hidden"
    )
    return {
        "refused": False,
        "accepted": True,
        "kind": "capture_plan",
        "surface": "hidden",
        "broader_traces": True,
        "n_tokens": len(token_ids),
        "token_ids": token_ids,
        "token_selection_strategy": (
            f"Disjoint window token_id {start}..{start + FLASH_MIN_ROWS - 1}, "
            f"256 unique ids, none of {len(set(prior_tokens))} already captured. "
            "The binary only accepts token ids, not prompt-domain tags; "
            "capability_domain remains ABSENT until the capture path records it. "
            "This is route-union growth, not a 4096-row recapture of the same window."
        ),
        "layers": {"kind": "same_tensor", "layer": EXISTING_LAYER, "tensor": EXISTING_TENSOR},
        "layer": EXISTING_LAYER,
        "expected_wall_time_estimate": estimate,
        "wall_time_estimate": estimate,
        "diversity_contract": diversity_contract("hidden"),
        "expected_diversity_gain": (
            f"The real window touched {cov.get('route_union_size')} of 512 experts. "
            "A disjoint 256-token window can grow that union. It cannot be claimed "
            "in advance that it will; the gain is expected, not measured. Recapturing "
            "tokens already in hand has expected gain 0."
        ),
        "pinned_revision": PINNED_REVISION,
        "current_binary_emits_this_surface": True,
        "current_binary_note": (
            "flash_meta_teacher_trace.rs already captures this tensor. A disjoint "
            "--token-start/--count run is the broader-trace capture. Do not pass "
            "the overlapping range."
        ),
        "executed": False,
        "gpu_authority": False,
        "claim_boundary": (
            "STATIC_ONLY plan. Wall time is an ESTIMATE. Expected route-union "
            "growth is a hypothesis until a capture runs."
        ),
    }


# ---------------------------------------------------------------------------
# validate(corpus)
# ---------------------------------------------------------------------------


def _axis(name: str, passed: bool, *, value: Any, required: bool, detail: str) -> dict[str, Any]:
    return {
        "axis": name,
        "pass": passed,
        "required": required,
        "value": value,
        "detail": detail,
    }


def validate(
    corpus: Mapping[str, Any] | None,
    *,
    raise_on_refuse: bool = False,
) -> dict[str, Any]:
    """Per-axis diversity check of a captured FLASH teacher receipt.

    Required axes are the ones the real corpus must (and does) pass: unique
    token ids, unique row hashes, binary min rows, non-degenerate route union,
    unique ordered top-k sets, pinned revision, bound surface. capability_domain
    is reported; ABSENT is not a pass and not a required fail — the real corpus
    does not label domains, and this module will not pretend it does.
    """
    refusals: list[str] = []
    axes: list[dict[str, Any]] = []

    if not isinstance(corpus, dict) or not corpus:
        result = {
            "accepted": False,
            "refusals": ["ABSENT_CORPUS"],
            "axes": [
                _axis("present", False, value=None, required=True, detail="corpus is absent")
            ],
            "n_rows": 0,
            "claim_boundary": "STATIC_ONLY validator. Absence is a refusal, not a pass.",
        }
        if raise_on_refuse:
            raise ExpansionRefused("REFUSED: validate() was given no corpus", result)
        return result

    rows = _rows_of(corpus)
    tokens = _token_ids_of(corpus)
    hashes = [_row_hash(r) for r in rows]
    missing_hash = sum(1 for h in hashes if not h)
    unique_hashes = {h for h in hashes if h}
    unique_tokens = set(tokens)
    union = _route_union_of(corpus, rows)
    topk = {tuple(_route_ids_of(r)) for r in rows}
    topk.discard(())
    trace = corpus.get("teacher_trace") if isinstance(corpus.get("teacher_trace"), dict) else {}
    tensor = trace.get("surface") or corpus.get("surface")
    pinned = corpus.get("pinned_revision")
    n_rows = len(rows) if rows else len(tokens)

    # Disagreement between token_ids and per-row token_id is a refusal.
    row_tokens: list[int] = []
    for row in rows:
        if "token_id" in row:
            try:
                row_tokens.append(int(row["token_id"]))
            except (TypeError, ValueError):
                row_tokens.append(-1)
    token_agree = (not row_tokens) or (row_tokens == tokens) or (not tokens)

    audit = corpus.get("route_audit") if isinstance(corpus.get("route_audit"), dict) else {}
    declared_union = audit.get("route_union")
    union_agree = True
    if isinstance(declared_union, list) and declared_union and rows:
        try:
            declared_ids = set(int(x) for x in declared_union)
        except (TypeError, ValueError):
            declared_ids = set()
            union_agree = False
        computed = set()
        for row in rows:
            computed.update(_route_ids_of(row))
        if declared_ids and computed and declared_ids != computed:
            union_agree = False
            union = []

    dup_tokens = bool(tokens) and (
        len(tokens) != len(unique_tokens) or (n_rows and len(tokens) != n_rows)
    )
    degenerate = len(union) < FLASH_MIN_ROUTE_UNION or len(topk) < FLASH_MIN_TOPK_SETS

    axes.append(
        _axis(
            "unique_token_ids",
            bool(tokens) and (not dup_tokens) and token_agree,
            value={
                "n": len(tokens),
                "n_unique": len(unique_tokens),
                "token_ids_agree_with_rows": token_agree,
            },
            required=True,
            detail=(
                "token ids absent"
                if not tokens
                else (
                    "duplicate token ids"
                    if dup_tokens
                    else (
                        "token_ids disagree with rows"
                        if not token_agree
                        else "all token ids unique"
                    )
                )
            ),
        )
    )
    if not tokens:
        refusals.append("TOKEN_IDS_ABSENT")
    elif dup_tokens:
        refusals.append("DUPLICATE_TOKEN_IDS")
    if tokens and not token_agree:
        refusals.append("TOKEN_ID_MISMATCH")

    axes.append(
        _axis(
            "unique_row_hashes",
            missing_hash == 0 and len(unique_hashes) == n_rows and n_rows > 0,
            value={"n_rows": n_rows, "n_unique": len(unique_hashes), "missing_hash": missing_hash},
            required=True,
            detail=(
                "row hashes missing"
                if missing_hash
                else (
                    "duplicate row hashes"
                    if n_rows and len(unique_hashes) < n_rows
                    else "all row hashes distinct"
                )
            ),
        )
    )
    if missing_hash:
        refusals.append("ROW_HASH_ABSENT")
    elif n_rows and len(unique_hashes) < n_rows:
        refusals.append("DUPLICATE_ROW_HASHES")

    axes.append(
        _axis(
            "min_unique_rows",
            len(unique_hashes) >= FLASH_MIN_ROWS,
            value={"n_unique": len(unique_hashes), "min": FLASH_MIN_ROWS},
            required=True,
            detail=(
                "below the binary's 256 unique-row refusal"
                if len(unique_hashes) < FLASH_MIN_ROWS
                else "meets binary min unique rows"
            ),
        )
    )
    if len(unique_hashes) < FLASH_MIN_ROWS:
        refusals.append("BINARY_MIN_ROWS_REFUSAL")

    axes.append(
        _axis(
            "route_union_nondegenerate",
            (not degenerate) and union_agree,
            value={
                "route_union_size": len(union),
                "unique_ordered_topk_sets": len(topk),
                "audit_agrees_with_rows": union_agree,
                "floor": FLASH_MIN_ROUTE_UNION,
            },
            required=True,
            detail=(
                "route_audit.route_union disagrees with rows"
                if not union_agree
                else (
                    "degenerate route union"
                    if degenerate
                    else "route union non-degenerate"
                )
            ),
        )
    )
    if not union_agree:
        refusals.append("ROUTE_UNION_DISAGREES")
    elif degenerate:
        refusals.append("DEGENERATE_ROUTE_UNION")

    axes.append(
        _axis(
            "token_position_spread",
            len(unique_tokens) >= FLASH_MIN_ROWS,
            value={"n_unique_token_ids": len(unique_tokens)},
            required=True,
            detail=(
                "token-position spread below 256 unique ids"
                if len(unique_tokens) < FLASH_MIN_ROWS
                else "token-position spread meets the binary bar"
            ),
        )
    )
    if len(unique_tokens) < FLASH_MIN_ROWS and "DUPLICATE_TOKEN_IDS" not in refusals:
        refusals.append("TOKEN_POSITION_SPREAD_BELOW_MIN")

    axes.append(
        _axis(
            "pinned_revision",
            pinned == PINNED_REVISION,
            value=pinned,
            required=True,
            detail=(
                "pinned revision matches Flash specimen"
                if pinned == PINNED_REVISION
                else "pinned revision missing or other specimen"
            ),
        )
    )
    if pinned != PINNED_REVISION:
        refusals.append("PINNED_REVISION_MISMATCH")

    axes.append(
        _axis(
            "surface_bound",
            bool(tensor),
            value=tensor,
            required=True,
            detail="teacher_trace.surface bound" if tensor else "surface unbound",
        )
    )
    if not tensor:
        refusals.append("SURFACE_UNBOUND")

    domains: list[str] = []
    for row in rows:
        d = row.get("capability_domain")
        if isinstance(d, str) and d:
            domains.append(d)
    if domains:
        n_dom = len(set(domains))
        axes.append(
            _axis(
                "capability_domain",
                n_dom >= 2,
                value={"n_unique": n_dom, "declared": list(CAPABILITY_DOMAINS)},
                required=False,
                detail="labelled capability domains present",
            )
        )
    else:
        axes.append(
            _axis(
                "capability_domain",
                False,
                value="ABSENT",
                required=False,
                detail=(
                    "ABSENT on this receipt; the Flash capture path records token "
                    "ids, not prompt-domain tags. Reported, not filled in, not a "
                    "required fail."
                ),
            )
        )

    # Dedup refusals, preserve order. Non-required axes never enter this list.
    seen: set[str] = set()
    refusals_u: list[str] = []
    for c in refusals:
        if c not in seen:
            seen.add(c)
            refusals_u.append(c)

    required_failed = [a for a in axes if a["required"] and not a["pass"]]
    accepted = not required_failed

    result = {
        "accepted": accepted,
        "refusals": refusals_u,
        "axes": axes,
        "n_rows": n_rows,
        "n_unique_token_ids": len(unique_tokens),
        "n_unique_row_hashes": len(unique_hashes),
        "route_union_size": len(union),
        "unique_ordered_topk_sets": len(topk),
        "canonical_surface": canonical_surface_of(corpus),
        "tensor": tensor,
        "pinned_revision": pinned,
        "claim_boundary": (
            "STATIC_ONLY validator of a captured receipt. No GPU. Refusal is a "
            "structural property of token ids, hashes, and routes."
        ),
    }
    if not result["accepted"] and raise_on_refuse:
        raise ExpansionRefused(
            f"REFUSED: teacher corpus expansion validator (codes={result['refusals']})",
            result,
        )
    return result


# ---------------------------------------------------------------------------
# dedupe_against
# ---------------------------------------------------------------------------


def _candidate_tokens_and_surface(
    candidate: Mapping[str, Any],
) -> tuple[list[int], str | None]:
    if candidate.get("kind") == "capture_plan":
        tokens = [int(x) for x in (candidate.get("token_ids") or [])]
        return tokens, str(candidate.get("surface") or "") or None
    cov = coverage_of(candidate)
    return list(cov.get("token_ids") or []), cov.get("canonical_surface")


def dedupe_against(
    prior: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Overlap between a new capture and the rows already in hand.

    One-argument form (candidate is None) is the waste case: a recapture of
    the prior's own surface and token ids. That must report WASTE.
    """
    if not isinstance(prior, dict) or not prior:
        return {
            "accepted": False,
            "refusals": ["ABSENT_PRIOR"],
            "verdict": "REFUSED",
            "why": "No prior corpus; overlap cannot be computed and is not defaulted to zero.",
        }
    prior_cov = coverage_of(prior)
    prior_tokens = set(prior_cov.get("token_ids") or [])
    prior_surface = prior_cov.get("canonical_surface")
    prior_hashes = {_row_hash(r) for r in _rows_of(prior)}
    prior_hashes.discard(None)

    if candidate is None:
        cand_tokens = list(prior_cov.get("token_ids") or [])
        cand_surface = prior_surface
        cand_hashes = set(prior_hashes)
        implicit = True
    else:
        cand_tokens, cand_surface = _candidate_tokens_and_surface(candidate)
        cand_hashes = {_row_hash(r) for r in _rows_of(candidate)}
        cand_hashes.discard(None)
        implicit = False

    tok_overlap = sorted(set(cand_tokens) & prior_tokens)
    hash_overlap = sorted(h for h in (cand_hashes & prior_hashes) if h)
    n_cand = len(set(cand_tokens)) or len(cand_hashes)
    token_frac = (len(tok_overlap) / len(set(cand_tokens))) if cand_tokens else 0.0
    same_surface = bool(cand_surface and cand_surface == prior_surface)

    if implicit or (same_surface and tok_overlap and token_frac == 1.0):
        verdict = "WASTE"
        why = (
            "Same surface and the same token ids as the existing corpus. "
            "The capture would re-derive rows already in hand."
        )
    elif same_surface and tok_overlap:
        verdict = "PARTIAL_OVERLAP"
        why = (
            f"{len(tok_overlap)} token ids already captured on {prior_surface}. "
            "Drop the overlap; only the disjoint remainder is new."
        )
    elif tok_overlap and not same_surface:
        verdict = "SAME_TOKENS_NEW_SURFACE"
        why = (
            "Token ids overlap but the surface is new. Joinable on token_id, "
            "not a re-derivation of the hidden rows."
        )
    else:
        verdict = "NO_OVERLAP"
        why = "No shared token ids with the existing corpus on this comparison."

    return {
        "accepted": True,
        "verdict": verdict,
        "why": why,
        "implicit_recapture": implicit,
        "prior_surface": prior_surface,
        "candidate_surface": cand_surface,
        "prior_n_unique_tokens": len(prior_tokens),
        "candidate_n_unique_tokens": len(set(cand_tokens)),
        "token_overlap_n": len(tok_overlap),
        "token_overlap_fraction_of_candidate": token_frac,
        "hash_overlap_n": len(hash_overlap),
        "same_surface": same_surface,
        "n_candidate": n_cand,
        "claim_boundary": (
            "STATIC_ONLY overlap report. Not a capture and not a hardware number."
        ),
    }


# ---------------------------------------------------------------------------
# Rank by diversity gain. Size is the least interesting axis.
# ---------------------------------------------------------------------------


def rank_plans(
    *,
    prior: Mapping[str, Any] | None = None,
    screen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Order capture plans by expected diversity gain, not by row count."""
    needed = surfaces_needed(prior, screen, raise_on_refuse=False)
    prior_doc = prior if isinstance(prior, dict) else try_load_rel(REAL_CORPUS_REL)
    ranked: list[dict[str, Any]] = []
    for name in needed.get("needed_names") or []:
        try:
            p = plan(name, prior=prior_doc, screen=screen, raise_on_refuse=True)
        except ExpansionRefused as exc:
            ranked.append(
                {
                    "surface": name,
                    "refused": True,
                    "reason": (exc.result or {}).get("reason"),
                    "expected_diversity_gain": None,
                    "n_tokens": None,
                }
            )
            continue
        ranked.append(
            {
                "surface": p["surface"],
                "refused": False,
                "n_tokens": p["n_tokens"],
                "expected_diversity_gain": p["expected_diversity_gain"],
                "wall_time_estimate": p["wall_time_estimate"],
                "current_binary_emits_this_surface": p["current_binary_emits_this_surface"],
            }
        )
    broader = None
    if prior_doc is not None:
        try:
            broader = plan_broader_traces(prior=prior_doc, raise_on_refuse=True)
        except ExpansionRefused:
            broader = None

    # Diversity rank is the needed-list order from the screen (new surfaces
    # first). Broader hidden traces come after, because they extend a surface
    # that already falsified a family rather than opening a zero-coverage axis.
    # A 4096-row recapture of the covered window is not on this list.
    return {
        "rank_rule": (
            "Expected diversity gain, not size. The real 256-row corpus spanning "
            "117 experts taught the screen enough to fail the family. A 4096-row "
            "recapture of the same window would teach less. New surfaces "
            "(router, routed-output, terminal-logit) outrank more hidden rows; "
            "a disjoint hidden window outranks a redundant recapture, which is "
            "refused rather than ranked."
        ),
        "needed_in_diversity_order": needed.get("needed_names") or [],
        "plans": ranked,
        "broader_traces": None
        if broader is None
        else {
            "surface": broader["surface"],
            "broader_traces": True,
            "n_tokens": broader["n_tokens"],
            "token_selection_strategy": broader["token_selection_strategy"],
            "expected_diversity_gain": broader["expected_diversity_gain"],
            "wall_time_estimate": broader["wall_time_estimate"],
        },
        "worked_example_real_corpus": {
            "n_rows": 256,
            "route_union": 117,
            "unique_token_ids": 256,
            "unique_ordered_topk_sets": 256,
            "unique_row_hashes": 256,
            "tensor": EXISTING_TENSOR,
            "layer": EXISTING_LAYER,
            "lesson": (
                "256 rows spanning 117 experts and 256 distinct ordered top-k "
                "sets were enough for the screen to fail the family. Row count "
                "is the least interesting axis."
            ),
        },
        "covered": needed.get("covered") or [],
        "refusals": needed.get("refusals") or [],
    }


# ---------------------------------------------------------------------------
# Fixtures for negative controls. Not captures.
# ---------------------------------------------------------------------------


def _fixture_hash(tag: str, i: int) -> str:
    return hashlib.sha256(f"fixture|{tag}|{i}".encode()).hexdigest()


def make_flash_corpus(
    *,
    n: int = FLASH_MIN_ROWS,
    duplicate_token_ids: bool = False,
    degenerate_routes: bool = False,
    surface: str = EXISTING_TENSOR,
    token_start: int = 0,
    pinned_revision: str = PINNED_REVISION,
    route_union_size: int = 117,
) -> dict[str, Any]:
    """FLASH-schema fixture. Never a GPU capture. No hardware fields."""
    rows: list[dict[str, Any]] = []
    token_ids: list[int] = []
    union: set[int] = set()
    n_routes = max(1, int(route_union_size))
    for i in range(n):
        if duplicate_token_ids:
            # Two rows share token 0; the last id is dropped so n is unchanged.
            tid = 0 if i == 1 else (token_start + i)
            if i == 1:
                tid = token_start
        else:
            tid = token_start + i
        if degenerate_routes:
            routes = [7]
        else:
            # Ordered top-k of length 10, heavy-tailed across n_routes.
            base = i % n_routes
            routes = [((base + k * 3) % n_routes) + 3 for k in range(10)]
        union.update(routes)
        token_ids.append(tid)
        rows.append(
            {
                "row": i,
                "token_id": tid,
                "layer4_mlp_input_sha256": _fixture_hash(surface, i),
                "route_ids": routes,
            }
        )
    topk = {tuple(r["route_ids"]) for r in rows}
    return {
        "schema": "hawking.flash.meta_teacher_trace.v1",
        "status": "FIXTURE_NOT_A_CAPTURE",
        "model": FLASH_SPECIMEN["model"],
        "pinned_revision": pinned_revision,
        "teacher_trace": {
            "layer": EXISTING_LAYER,
            "surface": surface,
            "organ": EXISTING_ORGAN,
            "rows": n,
            "raw_rows": n,
            "unique_rows": n,
        },
        "route_audit": {
            "rows": n,
            "unique_ordered_topk_sets": len(topk),
            "route_union": sorted(union),
        },
        "rows": rows,
        "token_ids": token_ids,
        "promotion_allowed": False,
        "claim_boundary": "Deterministic fixture; not a GPU capture and not a promotion.",
    }


# ---------------------------------------------------------------------------
# Selftest + receipt
# ---------------------------------------------------------------------------


def selftest() -> dict[str, Any]:
    real = load_real_corpus()
    screen = load_screen()
    real_v = validate(real, raise_on_refuse=False)
    if not real_v["accepted"]:
        raise SystemExit(f"selftest: real corpus must pass, got {real_v['refusals']}")

    dup = make_flash_corpus(duplicate_token_ids=True)
    dup_v = validate(dup, raise_on_refuse=False)
    if dup_v["accepted"] or "DUPLICATE_TOKEN_IDS" not in dup_v["refusals"]:
        raise SystemExit(f"selftest: duplicate token ids must fail, got {dup_v}")

    deg = make_flash_corpus(degenerate_routes=True)
    deg_v = validate(deg, raise_on_refuse=False)
    if deg_v["accepted"] or "DEGENERATE_ROUTE_UNION" not in deg_v["refusals"]:
        raise SystemExit(f"selftest: degenerate route union must fail, got {deg_v}")

    needed = surfaces_needed(real, screen)
    hidden_refused = False
    hidden_codes: list[str] = []
    try:
        plan("hidden", prior=real, screen=screen, raise_on_refuse=True)
    except RedundantCapture as exc:
        hidden_refused = True
        hidden_codes = list(exc.codes)
    if not hidden_refused:
        raise SystemExit("selftest: plan('hidden') against the real corpus must be redundant")

    unknown_refused = False
    try:
        plan("combine", prior=real, screen=screen, raise_on_refuse=True)
    except UnknownSurface:
        unknown_refused = True
    if not unknown_refused:
        raise SystemExit("selftest: plan('combine') must refuse an uninvented surface")

    router_plan = plan("router", prior=real, screen=screen, raise_on_refuse=True)
    if "ESTIMATE" not in json.dumps(router_plan["wall_time_estimate"]):
        raise SystemExit("selftest: wall time must be labelled ESTIMATE")

    waste = dedupe_against(real)
    join = dedupe_against(real, router_plan)
    broader = plan_broader_traces(prior=real)
    disjoint = dedupe_against(real, broader)

    absent = validate(None, raise_on_refuse=False)
    if absent["accepted"]:
        raise SystemExit("selftest: absent corpus must not pass")

    thin = make_flash_corpus(n=16)
    thin_v = validate(thin, raise_on_refuse=False)
    if thin_v["accepted"] or "BINARY_MIN_ROWS_REFUSAL" not in thin_v["refusals"]:
        raise SystemExit("selftest: a 16-row fixture must not design around the 256-row bar")

    return {
        "real_corpus_accepted": True,
        "real_route_union": real_v["route_union_size"],
        "real_unique_hashes": real_v["n_unique_row_hashes"],
        "real_unique_tokens": real_v["n_unique_token_ids"],
        "duplicate_token_ids_refused": True,
        "duplicate_token_ids_codes": dup_v["refusals"],
        "degenerate_route_union_refused": True,
        "degenerate_route_union_codes": deg_v["refusals"],
        "hidden_plan_redundant": hidden_refused,
        "hidden_plan_codes": hidden_codes,
        "unknown_surface_refused": unknown_refused,
        "needed_names": needed.get("needed_names"),
        "router_plan_accepted": router_plan.get("accepted"),
        "router_n_tokens": router_plan.get("n_tokens"),
        "wall_time_labelled_estimate": True,
        "dedupe_recapture_verdict": waste["verdict"],
        "dedupe_router_verdict": join["verdict"],
        "dedupe_broader_verdict": disjoint["verdict"],
        "absent_corpus_accepted": absent["accepted"],
        "thin_corpus_refused": (not thin_v["accepted"]),
        "bounded_target_rows_agrees": BOUNDED_TARGET_ROWS == FLASH_MIN_ROWS,
    }


def _strip_hardware(node: Any) -> Any:
    """Drop any hardware-named numeric fields before sealing. Belt, not a workaround."""
    from tools.future._common import HARDWARE_FIELDS

    if isinstance(node, dict):
        return {
            k: _strip_hardware(v)
            for k, v in node.items()
            if not (k in HARDWARE_FIELDS and isinstance(v, (int, float)))
        }
    if isinstance(node, list):
        return [_strip_hardware(v) for v in node]
    return node


def build() -> Any:
    test = selftest()
    real = load_real_corpus()
    screen = load_screen()
    needed = surfaces_needed(real, screen)
    ranked = rank_plans(prior=real, screen=screen)
    real_v = validate(real)
    waste = dedupe_against(real)
    plans = {}
    for name in needed.get("needed_names") or []:
        plans[name] = plan(name, prior=real, screen=screen, raise_on_refuse=True)
    broader = plan_broader_traces(prior=real)

    extra_neg = [
        (
            "flash_meta_teacher_trace.rs currently emits layer-4 mlp_input plus "
            "per-row route_ids and layer-output hashes. router / routed-output / "
            "terminal-logit plans are contracts for captures that binary does not "
            "yet perform. Declared next_gate surfaces != executed captures."
        ),
        (
            "The real corpus has no capability_domain labels. validate() reports "
            "that axis as ABSENT rather than filling it in."
        ),
        (
            "Wall-time figures are ESTIMATE scaled from one ~25 min / 256-row / "
            "layer-4 capture. Terminal-logit extra depth is NOT_MEASURED. "
            "Co-capture savings are NOT_MEASURED."
        ),
        (
            "This sidecar did not run a capture, take a GPU lease, or acquire a "
            "bench lock. executed=False on every plan."
        ),
        (
            "docs/FLASH_META_REPRESENTATION.md and "
            "crates/hawking-core/examples/flash_meta_teacher_trace.rs are not in "
            "this worktree's HEAD; they were read from the parent hawking checkout. "
            "A sparse miss is not evidence of absence."
        ),
        (
            "Expected route-union growth from a disjoint 256-token hidden window "
            "is a hypothesis, not a measurement."
        ),
    ]

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Plan the next Flash teacher captures so the expensive ones are the "
            "informative ones: name zero-coverage surfaces from the screen and "
            "funnel, refuse a redundant recapture of the real 256-row hidden "
            "corpus, rank by diversity gain (route coverage, token-position "
            "spread, new surfaces) rather than row count, validate a captured "
            "receipt against that contract, and report overlap with rows already "
            "in hand."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "promotion_allowed": False,
        "real_corpus": {
            "path": REAL_CORPUS_REL,
            "schema": real.get("schema"),
            "status": real.get("status"),
            "tensor": (real.get("teacher_trace") or {}).get("surface"),
            "layer": (real.get("teacher_trace") or {}).get("layer"),
            "organ": (real.get("teacher_trace") or {}).get("organ"),
            "n_rows": real_v["n_rows"],
            "n_unique_row_hashes": real_v["n_unique_row_hashes"],
            "n_unique_token_ids": real_v["n_unique_token_ids"],
            "route_union_size": real_v["route_union_size"],
            "unique_ordered_topk_sets": real_v["unique_ordered_topk_sets"],
            "pinned_revision": real.get("pinned_revision"),
            "next_gate": real.get("next_gate"),
            "validate_accepted": real_v["accepted"],
            "calibration_elapsed": CALIBRATION_ELAPSED,
            "calibration_kind": "ESTIMATE",
        },
        "screen": {
            "path": SCREEN_REL,
            "status": screen.get("status"),
            "next_gate": screen.get("next_gate"),
            "declared_surfaces": needed.get("declared_from_screen"),
        },
        "surfaces_needed": needed,
        "plans": plans,
        "broader_traces": broader,
        "rank": ranked,
        "validate_real_corpus": real_v,
        "dedupe_recapture_is_waste": waste,
        "selftest": test,
        "row_count_is_the_least_interesting_axis": ranked.get("rank_rule"),
        "binary_contract": {
            "min_rows": FLASH_MIN_ROWS,
            "max_rows": FLASH_MAX_ROWS,
            "min_route_union": FLASH_MIN_ROUTE_UNION,
            "equals_teacher_corpus_bounded_target_rows": BOUNDED_TARGET_ROWS == FLASH_MIN_ROWS,
        },
        "recovered_implementation": [
            "tools/future/teacher_corpus.py — admission, diversity measures, FLASH_SPECIMEN, 256-row bound; extended the plan, not the guard",
            "tools/future/meta_funnel.py — GATES / required_input list mapped onto screen surface names",
            "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL256.json — the real 256-row layer-4 mlp_input corpus",
            "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL256.json — next_gate names the surfaces",
            "crates/hawking-core/examples/flash_meta_teacher_trace.rs — CLI, 256-row refusal, duplicate-token refusal, degenerate-route refusal (parent checkout)",
            "docs/FLASH_META_REPRESENTATION.md — hidden / routed-output / terminal-logit distillation contract (parent checkout)",
        ],
        "gaps_closed": [
            "surfaces_needed() recovers screen-named surfaces and reports which of them have zero teacher coverage",
            "plan(surface) writes a concrete capture (token ids, strategy, layers, ESTIMATE wall time, diversity contract) and refuses a covered surface as redundant",
            "validate() checks every required diversity axis on a FLASH receipt and passes the real corpus",
            "dedupe_against() reports WASTE for a same-surface recapture and SAME_TOKENS_NEW_SURFACE for a joinable new surface",
            "plans ranked by expected diversity gain with the real 256-row / 117-expert corpus as the worked example",
        ],
        "negative_findings": extra_neg,
        "resident_callable": {
            "entry_point": "tools.future.teacher_corpus_expansion.surfaces_needed() / plan() / validate() / dedupe_against()",
            "workunit": (
                "one CPU_ANALYSIS unit; rank next Flash teacher captures by "
                "diversity gain and emit ESTIMATE-labelled plans. Not a GPU capture."
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_REPRESENTATION.teacher-capture",
            "fails_closed": (
                "absent corpus, unparseable next_gate, unknown surface, redundant "
                "covered surface, duplicate token ids, degenerate route union, "
                "unique rows below the binary's 256-row refusal, prior too thin to join"
            ),
        },
    }
    doc = _strip_hardware(doc)
    out = write_receipt(RECEIPT, doc, RECORDED_BY)
    written = load_json(out)
    if written.get("schema") != SCHEMA or not written.get("seal_sha256"):
        raise SystemExit(f"receipt {out} failed round-trip")
    if written.get("bench", {}).get("gpu_authority") is not False:
        raise SystemExit("receipt gpu_authority is not false")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
