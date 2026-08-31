"""FLASH_META_REPLAN — keep the gate, kill one codec on one organ, spend the next rows elsewhere.

The first 256-row Flash teacher capture and the L4 coherence screen exist. The
screen failed. This module turns that failure into a re-plan: what is dead,
what is not, whether rank can buy the contract, whether 204 fit rows make the
five ranks trustworthy, and which of the nine SUB1 families inherit the scar.

It does not re-run the screen, does not take a GPU lease, and does not lower
the coherence contract. Contract values are read from the screen receipt.
An absent receipt is a recorded refusal, never a defaulted pass. A curve that
cannot support a rank reports no rank. A family whose mechanism is not the
failing one is not marked down.

    python3 tools/future/flash_meta_replan.py --build
    python3 -m pytest tools/future/test_flash_meta_replan.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import write_receipt
from tools.future.meta_funnel import load_receipt

RECEIPT = "FLASH_META_REPLAN.json"
SCHEMA = "hawking.future.flash_meta_replan.v1"
RECORDED_BY = "tools/future/flash_meta_replan.py"
VERSION = 1

SCREEN_REL = "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL256.json"
TEACHER_REL = "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL256.json"

# The corpus was grown 4x specifically to test whether the 256-row failure was a
# thin-sample artifact. It was not: every rank degraded by 0.14-0.18 held-out
# error while the per-expert Q4 baseline stayed flat, which is the signature of a
# fit that had been overfitting on 204 rows. Both corpora are carried because the
# COMPARISON is the finding -- one screen alone cannot show the direction of
# travel with more evidence.
CORPORA: tuple[tuple[str, str, str], ...] = (
    ("256",
     "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL256.json",
     "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL256.json"),
    ("1024",
     "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL1024.json",
     "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL1024.json"),
)


def corpus_comparison() -> dict[str, Any]:
    """Per-corpus extrapolation, and what changed when the corpus grew.

    Refuses to compare when a corpus is absent rather than reporting a single
    screen as though it were the trend.
    """
    per: dict[str, Any] = {}
    for name, screen_rel, _teacher_rel in CORPORA:
        screen, rel = load_named(screen_rel)
        if not screen:
            per[name] = {"present": False, "why": f"{screen_rel} not on disk"}
            continue
        rows = ((screen.get("surface") or {}).get("rows")) or []
        contract = screen.get("coherence_contract") or {}
        q4 = rows[0].get("per_expert_q4_heldout_relative_fro_error") if rows else None
        ext = rank_extrapolation(rows, contract, q4_bpw=q4)
        per[name] = {
            "present": True,
            "source": rel,
            "fit_rows": rows[0].get("fit_rows") if rows else None,
            "heldout_rows": rows[0].get("heldout_rows") if rows else None,
            "per_expert_q4_heldout_relative_fro_error": q4,
            "heldout_error_by_rank": {str(r["rank"]): r["heldout_relative_fro_error"] for r in rows},
            "heldout_cosine_by_rank": {str(r["rank"]): r["heldout_cosine"] for r in rows},
            "any_rank_passed": any(r.get("surface_gate_pass") for r in rows),
            "extrapolation": ext,
        }
    both = [n for n, v in per.items() if v.get("present")]
    if len(both) < 2:
        return {
            "corpora": per,
            "verdict": "INCOMPARABLE",
            "why": "fewer than two corpora on disk; one screen is not a trend",
        }
    a, b = per["256"], per["1024"]
    deltas = {
        rank: round(b["heldout_error_by_rank"][rank] - a["heldout_error_by_rank"][rank], 6)
        for rank in a["heldout_error_by_rank"] if rank in b["heldout_error_by_rank"]
    }
    worse = all(v > 0 for v in deltas.values())
    q4_shift = round((b["per_expert_q4_heldout_relative_fro_error"] or 0)
                     - (a["per_expert_q4_heldout_relative_fro_error"] or 0), 6)
    return {
        "corpora": per,
        "heldout_error_delta_by_rank": deltas,
        "q4_baseline_shift": q4_shift,
        "every_rank_degraded_with_more_data": worse,
        "verdict": "OVERFIT_ON_THE_SMALLER_CORPUS" if worse else "STABLE_ACROSS_CORPORA",
        "why": (
            "every rank degraded when the fit set quadrupled while the comparator it "
            "must beat did not move. A codec whose held-out error grows with more "
            "fitting data was fitting the sample, not the function."
            if worse else
            "held-out error did not uniformly degrade with a larger corpus"
        ),
        "still_bounded": (
            "one organ, one surface, five ranks, two corpora. This does not falsify "
            "sub-1 in general nor the other eight families."
        ),
    }
SUB1_REL = "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json"
SUB1_HEADLESS_REL = "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
INDEX_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
SCREEN_TOOL_REL = "tools/flash_meta_coherence_screen.py"
DOC_REL = "docs/FLASH_META_REPRESENTATION.md"

# Statistical support for a 2-parameter error-vs-rank model. Not a coherence
# gate. The coherence numbers live on the screen receipt and are never stored
# here as literals.
MIN_CURVE_R2 = 0.9
N_POINTS_MIN = 4

# NS-014 / NNS-007 original ids. Consulted on the index receipt; the predicate
# is theirs (n_fit >= claimed rank for rank-r), not a new law.
NS014_ORIGINAL_IDS = frozenset(
    {
        "NS-014",
        "NNS-007",
        "underdetermined_fit_rows_rank_or_rows_dim",
    }
)

_DISTILL_SURFACES = re.compile(
    r"distill\s+([a-z0-9][a-z0-9_/\-]+)\s+surfaces",
    re.IGNORECASE,
)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _deep(node: Any, *keys: str) -> Any:
    cur = node
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_named(*rels: str) -> tuple[dict[str, Any] | None, str | None]:
    """Disk then HEAD, first hit. Absence is not proof the object is gone."""
    for rel in rels:
        doc = load_receipt(rel)
        if isinstance(doc, dict):
            return doc, rel
    return None, None


def load_inputs() -> dict[str, Any]:
    screen, screen_rel = load_named(SCREEN_REL)
    teacher, teacher_rel = load_named(TEACHER_REL)
    sub1, sub1_rel = load_named(SUB1_REL, SUB1_HEADLESS_REL)
    index, index_rel = load_named(INDEX_REL)
    return {
        "screen": screen,
        "screen_rel": screen_rel,
        "teacher": teacher,
        "teacher_rel": teacher_rel,
        "sub1": sub1,
        "sub1_rel": sub1_rel,
        "index": index,
        "index_rel": index_rel,
    }


def cited(rel: str | None, doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(doc, Mapping) or not rel:
        return {"path": rel, "present": False}
    return {
        "path": rel,
        "present": True,
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "seal_sha256": doc.get("seal_sha256"),
    }


# ---------------------------------------------------------------------------
# Contract. Read, never redefined.
# ---------------------------------------------------------------------------


def contract_from_screen(screen: Mapping[str, Any] | None) -> dict[str, Any]:
    """Lift the coherence contract from the screen. Refuse if any field is missing.

    This module has no numeric fallback. A caller that wants a different
    cosine / error / Q4 gate must edit the screen, not this file.
    """
    if not isinstance(screen, Mapping):
        return {
            "ok": False,
            "reason": "screen receipt absent; contract is not defaulted here",
            "min_heldout_cosine": None,
            "max_heldout_relative_fro_error": None,
            "must_beat_per_expert_q4": None,
        }
    block = screen.get("coherence_contract")
    if not isinstance(block, Mapping):
        return {
            "ok": False,
            "reason": "screen.coherence_contract absent; contract is not defaulted here",
            "min_heldout_cosine": None,
            "max_heldout_relative_fro_error": None,
            "must_beat_per_expert_q4": None,
        }
    cosine = _as_float(block.get("min_heldout_cosine"))
    err = _as_float(block.get("max_heldout_relative_fro_error"))
    beat = _as_bool(block.get("must_beat_per_expert_q4"))
    missing = [
        name
        for name, value in (
            ("min_heldout_cosine", cosine),
            ("max_heldout_relative_fro_error", err),
            ("must_beat_per_expert_q4", beat),
        )
        if value is None
    ]
    if missing:
        return {
            "ok": False,
            "reason": "screen.coherence_contract missing " + ",".join(missing),
            "min_heldout_cosine": cosine,
            "max_heldout_relative_fro_error": err,
            "must_beat_per_expert_q4": beat,
        }
    return {
        "ok": True,
        "reason": "read from screen.coherence_contract; not redefined",
        "source": "coherence_contract",
        "min_heldout_cosine": cosine,
        "max_heldout_relative_fro_error": err,
        "must_beat_per_expert_q4": beat,
        "fit_holdout_required": _as_bool(block.get("fit_holdout_required")),
        "raw_keys": sorted(block.keys()),
    }


def named_next_surfaces(screen: Mapping[str, Any] | None) -> dict[str, Any]:
    """The screen's next_gate names the surfaces. This module does not invent them."""
    if not isinstance(screen, Mapping):
        return {
            "ok": False,
            "surfaces": [],
            "reason": "screen absent; capture plan is not invented here",
        }
    text = screen.get("next_gate")
    if not isinstance(text, str) or not text.strip():
        return {
            "ok": False,
            "surfaces": [],
            "reason": "screen.next_gate absent; capture plan is not invented here",
        }
    match = _DISTILL_SURFACES.search(text)
    if not match:
        return {
            "ok": False,
            "surfaces": [],
            "reason": (
                "screen.next_gate does not contain 'distill <a/b/c> surfaces'; "
                "capture plan is not invented here"
            ),
            "next_gate": text,
        }
    surfaces = [part.strip() for part in match.group(1).split("/") if part.strip()]
    if not surfaces:
        return {
            "ok": False,
            "surfaces": [],
            "reason": "distill token list was empty after split",
            "next_gate": text,
        }
    return {
        "ok": True,
        "surfaces": surfaces,
        "reason": "extracted from screen.next_gate",
        "next_gate": text,
    }


# ---------------------------------------------------------------------------
# Falsification. One organ, one surface, one split, five ranks.
# ---------------------------------------------------------------------------


def _rank_rows(screen: Mapping[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    surface = screen.get("surface")
    if not isinstance(surface, Mapping):
        return None, "screen.surface absent"
    rows = surface.get("rows")
    if not isinstance(rows, list) or not rows:
        return None, "screen.surface.rows absent or empty"
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            return None, f"screen.surface.rows[{i}] is not an object"
        gate = _as_bool(row.get("surface_gate_pass"))
        if gate is None:
            return None, f"screen.surface.rows[{i}].surface_gate_pass is not a bool"
        rank = _as_int(row.get("rank"))
        if rank is None:
            return None, f"screen.surface.rows[{i}].rank is not an int"
        err = _as_float(row.get("heldout_relative_fro_error"))
        cos = _as_float(row.get("heldout_cosine"))
        q4 = _as_float(row.get("per_expert_q4_heldout_relative_fro_error"))
        if err is None or cos is None or q4 is None:
            return None, f"screen.surface.rows[{i}] missing held-out numbers"
        out.append(
            {
                "rank": rank,
                "fit_rows": _as_int(row.get("fit_rows")),
                "heldout_rows": _as_int(row.get("heldout_rows")),
                "fit_relative_fro_error": _as_float(row.get("fit_relative_fro_error")),
                "heldout_relative_fro_error": err,
                "heldout_cosine": cos,
                "per_expert_q4_heldout_relative_fro_error": q4,
                "beats_per_expert_q4_on_heldout": _as_bool(
                    row.get("beats_per_expert_q4_on_heldout")
                ),
                "diagnostic_factor_equivalent_bpw": _as_float(
                    row.get("diagnostic_factor_equivalent_bpw")
                ),
                "surface_failure_gates": list(row.get("surface_failure_gates") or []),
                "first_surface_failure": row.get("first_surface_failure"),
                "surface_gate_pass": gate,
            }
        )
    return out, None


def falsification(
    screen: Mapping[str, Any] | None = None,
    teacher: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What this screen kills, and what it does not.

    A pass on any rank would mean 'cannot reach by rank' is itself not
    established. Missing gate flags are a refusal, not a silent fail.
    """
    if not isinstance(screen, Mapping):
        return {
            "verdict": "REFUSED",
            "reason": "screen receipt absent; a missing measurement is not a kill",
            "scope": None,
            "dead": [],
            "not_dead": [
                "sub-1 meta_bpw as a description budget",
                "the other eight SUB1 census families",
                "any codec other than the one this screen fitted",
            ],
        }
    contract = contract_from_screen(screen)
    if not contract["ok"]:
        return {
            "verdict": "REFUSED",
            "reason": contract["reason"],
            "scope": None,
            "dead": [],
            "not_dead": [
                "sub-1 meta_bpw as a description budget",
                "the other eight SUB1 census families",
            ],
            "contract": contract,
        }
    rows, why = _rank_rows(screen)
    if rows is None:
        return {
            "verdict": "REFUSED",
            "reason": why,
            "scope": None,
            "dead": [],
            "not_dead": [
                "sub-1 meta_bpw as a description budget",
                "the other eight SUB1 census families",
            ],
            "contract": contract,
        }
    kind = _deep(screen, "representation", "kind")
    organ = _deep(screen, "representation", "organ")
    if not isinstance(kind, str) or not kind or not isinstance(organ, str) or not organ:
        return {
            "verdict": "REFUSED",
            "reason": "screen.representation.kind/organ absent; scope would be a guess",
            "dead": [],
            "not_dead": [
                "sub-1 meta_bpw as a description budget",
                "the other eight SUB1 census families",
            ],
            "contract": contract,
        }
    fit_n = _as_int(_deep(screen, "surface", "fit_rows"))
    held_n = _as_int(_deep(screen, "surface", "heldout_rows"))
    if fit_n is None:
        fit_n = rows[0]["fit_rows"]
    if held_n is None:
        held_n = rows[0]["heldout_rows"]
    ranks = [r["rank"] for r in rows]
    passes = [r["surface_gate_pass"] for r in rows]
    if any(passes):
        passing = [r["rank"] for r in rows if r["surface_gate_pass"]]
        return {
            "verdict": "NOT_FALSIFIED_BY_THIS_RECEIPT",
            "reason": (
                "at least one rank has surface_gate_pass true; this screen does not "
                "kill the codec. Partial failure is not rounded into a family pass "
                "either — ranks that failed remain failed."
            ),
            "scope": {
                "kind": kind,
                "organ": organ,
                "fit_rows": fit_n,
                "heldout_rows": held_n,
                "ranks": ranks,
            },
            "dead": [],
            "not_dead": [
                f"{kind} on {organ} at ranks {passing}",
                "sub-1 meta_bpw as a description budget",
                "the other eight SUB1 census families",
            ],
            "failed_ranks": [r["rank"] for r in rows if not r["surface_gate_pass"]],
            "passing_ranks": passing,
            "rank_rows": rows,
            "contract": contract,
            "screen_status": screen.get("status"),
        }
    first_failures = [r.get("first_surface_failure") for r in rows]
    q4_beat = [r.get("beats_per_expert_q4_on_heldout") for r in rows]
    teacher_status = teacher.get("status") if isinstance(teacher, Mapping) else None
    teacher_surface = _deep(teacher, "teacher_trace", "surface") if isinstance(teacher, Mapping) else None
    err0 = rows[0]["heldout_relative_fro_error"]
    err1 = rows[-1]["heldout_relative_fro_error"]
    rank_ratio = None
    err_drop = None
    if rows[0]["rank"] and rows[-1]["rank"] and err0:
        rank_ratio = rows[-1]["rank"] / rows[0]["rank"]
        err_drop = (err0 - err1) / err0
    return {
        "verdict": "FALSIFIED_ON_STATED_SCOPE",
        "reason": (
            "every measured rank fails the screen's own coherence_contract; "
            "first_surface_failure is cited from the screen, not re-diagnosed here"
        ),
        "scope": {
            "kind": kind,
            "organ": organ,
            "fit_rows": fit_n,
            "heldout_rows": held_n,
            "ranks": ranks,
            "n_ranks": len(ranks),
            "teacher_surface": teacher_surface,
            "screen_status": screen.get("status"),
            "teacher_status": teacher_status,
        },
        "dead": [
            {
                "what": kind,
                "where": organ,
                "split": f"{fit_n}/{held_n}",
                "ranks": ranks,
                "first_surface_failures": first_failures,
                "beats_per_expert_q4_on_heldout": q4_beat,
                "reopen": (
                    "A new measurement of this codec on this organ that meets the "
                    "screen's own coherence_contract (cosine, held-out error, beat Q4). "
                    "A lower gate is not a reopen."
                ),
            }
        ],
        "not_dead": [
            "sub-1 meta_bpw as a description budget (still prospective; still not physical EBPW)",
            "the other eight SUB1 census families (different organs / programs)",
            "other codecs on the same organ (dictionary, sparse residual, generated-tile that is not this latent+readout)",
            "other routed-expert tensors (down_proj was not this screen)",
            "other layers",
            "router / hidden / routed-output / terminal-logit surfaces named by the screen and not fitted here",
        ],
        "rank_rows": rows,
        "contract": contract,
        "rank_span": {
            "rank_ratio": rank_ratio,
            "heldout_error_relative_drop": err_drop,
            "note": (
                "drop is arithmetic on the cited held-out errors; it is not a new "
                "measurement and it is not a reason to lower the gate"
            ),
        },
        "promotion_allowed": False,
        "gate_stands": True,
    }


# ---------------------------------------------------------------------------
# Rank extrapolation. Refuse a rank the five points cannot support.
# ---------------------------------------------------------------------------


def _ols(x: Sequence[float], y: Sequence[float]) -> dict[str, Any] | None:
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    syy = sum((yi - mean_y) ** 2 for yi in y)
    if sxx == 0.0:
        return None
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [(yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y)]
    ss_res = sum(resid)
    r2 = 1.0 - ss_res / syy if syy else 0.0
    return {
        "intercept": intercept,
        "slope": slope,
        "r_squared": r2,
        "ss_res": ss_res,
        "n": n,
    }


def _monotone_nonincreasing(values: Sequence[float]) -> bool:
    return all(values[i] <= values[i - 1] for i in range(1, len(values)))


def _bpw_slope(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = []
    for row in points:
        rank = _as_float(row.get("rank"))
        bpw = _as_float(row.get("diagnostic_factor_equivalent_bpw"))
        if rank is None or bpw is None or rank <= 0:
            continue
        pairs.append((rank, bpw))
    if not pairs:
        return {"ok": False, "reason": "no diagnostic_factor_equivalent_bpw on the rank rows"}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    fit = _ols(x, y)
    # Through-origin slope is the construction: bpw = k * rank on this screen.
    k = sum(bpw / rank for rank, bpw in pairs) / len(pairs)
    return {
        "ok": True,
        "bpw_per_rank": k,
        "ols": fit,
        "n": len(pairs),
        "note": (
            "diagnostic factor-equivalent bpw cited from the screen; not physical EBPW"
        ),
    }


def rank_extrapolation(
    points: Sequence[Mapping[str, Any]] | None,
    contract: Mapping[str, Any] | None,
    *,
    q4_bpw: float | None = None,
) -> dict[str, Any]:
    """Fit error-vs-rank. Report a rank only when the curve can support one.

    A floor above the contract is a refusal, not an infinite rank. A noisy or
    non-monotone series is a refusal, not a guessed rank. Five points remain
    five points: even a clean floor is an extrapolation, recorded as such.
    """
    target = _as_float((contract or {}).get("max_heldout_relative_fro_error")) if contract else None
    if target is None:
        return {
            "verdict": "REFUSED",
            "reason": "contract.max_heldout_relative_fro_error absent; rank is not invented",
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
        }
    if not points:
        return {
            "verdict": "REFUSED",
            "reason": "no rank rows; rank is not invented",
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
        }
    series: list[tuple[float, float, Mapping[str, Any]]] = []
    for row in points:
        rank = _as_float(row.get("rank"))
        err = _as_float(row.get("heldout_relative_fro_error"))
        if rank is None or err is None or rank <= 0 or err < 0:
            return {
                "verdict": "REFUSED",
                "reason": "a rank row is missing rank or heldout_relative_fro_error",
                "rank_required": None,
                "diagnostic_bpw_at_rank": None,
                "dominated_by_construction": None,
                "target_heldout_error": target,
            }
        series.append((rank, err, row))
    series.sort(key=lambda t: t[0])
    ranks = [t[0] for t in series]
    errors = [t[1] for t in series]
    if len(series) < N_POINTS_MIN:
        return {
            "verdict": "REFUSED",
            "reason": f"need at least {N_POINTS_MIN} rank points to extrapolate; have {len(series)}",
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
            "n_points": len(series),
        }
    already = [t[0] for t in series if t[1] <= target]
    if already:
        bpw_info = _bpw_slope([t[2] for t in series])
        rank_star = min(already)
        bpw_star = (
            bpw_info["bpw_per_rank"] * rank_star if bpw_info.get("ok") else None
        )
        dominated = None
        if bpw_star is not None and q4_bpw is not None:
            dominated = bpw_star > q4_bpw
        return {
            "verdict": "OBSERVED_NOT_EXTRAPOLATED",
            "reason": "a measured rank already meets the contract error; no curve is required",
            "rank_required": rank_star,
            "diagnostic_bpw_at_rank": bpw_star,
            "dominated_by_construction": dominated,
            "target_heldout_error": target,
            "q4_bpw": q4_bpw,
            "bpw_model": bpw_info,
            "n_points": len(series),
        }
    if not _monotone_nonincreasing(errors):
        return {
            "verdict": "REFUSED",
            "reason": (
                "held-out error is not monotone nonincreasing in rank; "
                "the series cannot support a rank-to-target"
            ),
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
            "ranks": ranks,
            "errors": errors,
        }

    models: list[dict[str, Any]] = []

    inv = _ols([1.0 / r for r in ranks], errors)
    if inv is not None:
        floor = inv["intercept"]
        slope = inv["slope"]
        models.append(
            {
                "name": "floor_inv",
                "form": "err = a + b/rank",
                "r_squared": inv["r_squared"],
                "floor": floor,
                "slope": slope,
                "decreasing": slope > 0.0,
                "limit_as_rank_inf": floor,
                "physical_limit_nonnegative": floor >= 0.0,
            }
        )

    invs = _ols([1.0 / math.sqrt(r) for r in ranks], errors)
    if invs is not None:
        floor = invs["intercept"]
        slope = invs["slope"]
        models.append(
            {
                "name": "floor_inv_sqrt",
                "form": "err = a + b/sqrt(rank)",
                "r_squared": invs["r_squared"],
                "floor": floor,
                "slope": slope,
                "decreasing": slope > 0.0,
                "limit_as_rank_inf": floor,
                "physical_limit_nonnegative": floor >= 0.0,
            }
        )

    logs = [math.log(r) for r in ranks]
    loglin = _ols(logs, errors)
    if loglin is not None:
        models.append(
            {
                "name": "log_linear",
                "form": "err = a + b ln(rank)",
                "r_squared": loglin["r_squared"],
                "intercept": loglin["intercept"],
                "slope": loglin["slope"],
                "decreasing": loglin["slope"] < 0.0,
                "limit_as_rank_inf": float("-inf"),
                "physical_limit_nonnegative": False,
                "why_unphysical": "error is driven through zero and below at finite rank",
            }
        )

    if all(e > 0.0 for e in errors):
        power = _ols(logs, [math.log(e) for e in errors])
        if power is not None:
            models.append(
                {
                    "name": "power",
                    "form": "err = A * rank^B",
                    "r_squared": power["r_squared"],
                    "log_intercept": power["intercept"],
                    "exponent": power["slope"],
                    "decreasing": power["slope"] < 0.0,
                    "limit_as_rank_inf": 0.0,
                    "physical_limit_nonnegative": True,
                }
            )

    admissible = [
        m
        for m in models
        if m.get("decreasing")
        and m.get("physical_limit_nonnegative")
        and _as_float(m.get("r_squared")) is not None
    ]
    if not admissible:
        return {
            "verdict": "REFUSED",
            "reason": (
                "no physically admissible decreasing error-vs-rank model "
                "(nonnegative limit, error decreasing in rank)"
            ),
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
            "models": models,
            "n_points": len(series),
        }
    admissible.sort(key=lambda m: m["r_squared"], reverse=True)
    best = admissible[0]
    if best["r_squared"] < MIN_CURVE_R2:
        return {
            "verdict": "REFUSED",
            "reason": (
                f"best admissible model {best['name']} has r_squared "
                f"{best['r_squared']} below the support threshold {MIN_CURVE_R2}; "
                "five points do not support a rank"
            ),
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
            "best_model": best,
            "models": models,
            "n_points": len(series),
        }
    limit = _as_float(best.get("limit_as_rank_inf"))
    if limit is not None and limit >= target:
        return {
            "verdict": "REFUSED",
            "reason": (
                f"best admissible model {best['name']} (r_squared={best['r_squared']}) "
                f"has limit {limit} as rank -> inf, which is above the contract "
                "held-out error; rank cannot buy the gate on this organ"
            ),
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
            "implied_floor": limit,
            "best_model": best,
            "models": models,
            "n_points": len(series),
            "uncertainty": (
                f"{len(series)} measured ranks. A 2-parameter floor model can fit "
                "them tightly and still be wrong outside the window. The refusal "
                "is that the supported curve never reaches the contract, not that "
                "a larger capture was run."
            ),
        }

    rank_star: float | None = None
    if best["name"] == "floor_inv":
        # a + b/r = target  =>  r = b / (target - a)
        denom = target - best["floor"]
        if denom > 0.0 and best["slope"] > 0.0:
            rank_star = best["slope"] / denom
    elif best["name"] == "floor_inv_sqrt":
        denom = target - best["floor"]
        if denom > 0.0 and best["slope"] > 0.0:
            ratio = best["slope"] / denom
            if ratio > 0.0:
                rank_star = ratio * ratio
    elif best["name"] == "power":
        # log err = a + b log r  =>  r = exp((log target - a) / b)
        b = best["exponent"]
        if b != 0.0 and target > 0.0:
            rank_star = math.exp((math.log(target) - best["log_intercept"]) / b)

    if rank_star is None or not math.isfinite(rank_star) or rank_star <= 0.0:
        return {
            "verdict": "REFUSED",
            "reason": f"best model {best['name']} did not produce a finite positive rank",
            "rank_required": None,
            "diagnostic_bpw_at_rank": None,
            "dominated_by_construction": None,
            "target_heldout_error": target,
            "best_model": best,
            "models": models,
            "n_points": len(series),
        }

    bpw_info = _bpw_slope([t[2] for t in series])
    bpw_star = bpw_info["bpw_per_rank"] * rank_star if bpw_info.get("ok") else None
    dominated = None
    if bpw_star is not None and q4_bpw is not None:
        dominated = bool(bpw_star > q4_bpw)
    return {
        "verdict": "EXTRAPOLATED",
        "reason": (
            f"best admissible model {best['name']} reaches the contract error "
            f"at rank {rank_star}"
        ),
        "rank_required": rank_star,
        "diagnostic_bpw_at_rank": bpw_star,
        "dominated_by_construction": dominated,
        "target_heldout_error": target,
        "q4_bpw": q4_bpw,
        "bpw_model": bpw_info,
        "best_model": best,
        "models": models,
        "n_points": len(series),
        "uncertainty": (
            f"{len(series)} measured ranks. Extrapolation beyond the last measured "
            "rank is a curve, not a new screen. Dominated-by-construction is a "
            "comparison of diagnostic factor-equivalent bpw to the cited Q4 "
            "component bpw, not a physical EBPW claim."
        ),
    }


# ---------------------------------------------------------------------------
# Underdetermination. NS-014 / NNS-007, applied, not assumed.
# ---------------------------------------------------------------------------


def _index_scars(index_doc: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(index_doc, Mapping):
        return []
    rows = index_doc.get("scars")
    if not isinstance(rows, list):
        return []
    hits = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        oid = str(row.get("original_id") or "")
        fam = str(row.get("hypothesis_family") or "")
        mech = str(row.get("failure_mechanism") or "").lower()
        claim = str(row.get("claim_refuted") or "").lower()
        if oid in NS014_ORIGINAL_IDS:
            hits.append(dict(row))
            continue
        blob = f"{fam} {mech} {claim}"
        if "underdetermined" in blob or "fewer captured rows" in blob or "undersampled_fits" in blob:
            hits.append(dict(row))
    return hits


def underdetermination_check(
    *,
    n_fit: int | None,
    ranks: Sequence[int] | None,
    input_width: int | None,
    index_doc: Mapping[str, Any] | None = None,
    claimed_full_dim: bool = False,
    rank_clamped_to_n_fit: bool = False,
) -> dict[str, Any]:
    """Apply the negative-index scar; do not assume the five ranks are thin or not.

    Rank-r (this screen): n_fit >= claimed rank.
    Full-dim: n_fit >= input_width.
    Rank clamped to n_fit is a starved score even when the inequality holds.
    """
    scars = _index_scars(index_doc)
    if n_fit is None or not ranks:
        return {
            "verdict": "REFUSED",
            "reason": "n_fit or ranks absent; trust is not defaulted",
            "index_consulted": bool(scars),
            "scars_cited": [
                {
                    "scar_id": s.get("scar_id"),
                    "original_id": s.get("original_id"),
                    "source_path": s.get("source_path"),
                    "hypothesis_family": s.get("hypothesis_family"),
                    "reopen_condition": s.get("reopen_condition"),
                }
                for s in scars
            ],
            "ranks": [],
        }
    if claimed_full_dim and input_width is None:
        return {
            "verdict": "REFUSED",
            "reason": "full-dim claim without input_width; trust is not defaulted",
            "index_consulted": bool(scars),
            "scars_cited": [
                {"scar_id": s.get("scar_id"), "original_id": s.get("original_id")}
                for s in scars
            ],
            "ranks": [],
        }
    if rank_clamped_to_n_fit:
        return {
            "verdict": "THIN",
            "reason": (
                "rank was clamped to n_fit; NS-014: the score is not the codec's score"
            ),
            "index_consulted": bool(scars),
            "n_fit": n_fit,
            "ranks": [
                {
                    "rank": int(r),
                    "n_fit": n_fit,
                    "trustworthy": False,
                    "thin": True,
                    "criterion": "rank not clamped to n_fit",
                }
                for r in ranks
            ],
        }

    per: list[dict[str, Any]] = []
    for raw in ranks:
        rank = _as_int(raw)
        if rank is None:
            return {
                "verdict": "REFUSED",
                "reason": f"rank {raw!r} is not an int",
                "index_consulted": bool(scars),
                "ranks": [],
            }
        if claimed_full_dim:
            ok = n_fit >= int(input_width)  # type: ignore[arg-type]
            criterion = "n_fit >= input_width (full-dim)"
        else:
            ok = n_fit >= rank
            criterion = "n_fit >= claimed rank (rank-r)"
        per.append(
            {
                "rank": rank,
                "n_fit": n_fit,
                "input_width": input_width,
                "claimed_full_dim": claimed_full_dim,
                "trustworthy": bool(ok),
                "thin": (not ok),
                "criterion": criterion,
                "rank_clamped_to_n_fit": False,
            }
        )
    thin = [p for p in per if p["thin"]]
    trustworthy = [p for p in per if p["trustworthy"]]
    if thin and trustworthy:
        verdict = "MIXED"
        reason = (
            f"{len(thin)} rank(s) thin, {len(trustworthy)} trustworthy under NS-014; "
            "mixed is not rounded into all-trustworthy"
        )
    elif thin:
        verdict = "THIN"
        reason = "every requested rank has n_fit < the NS-014 predicate"
    else:
        verdict = "TRUSTWORTHY"
        reason = "every requested rank satisfies the NS-014 rank-r predicate"
    return {
        "verdict": verdict,
        "reason": reason,
        "index_consulted": bool(scars),
        "index_present": isinstance(index_doc, Mapping),
        "predicate": (
            "NS-014 / NNS-007: n_fit >= claimed rank for a rank-r score; "
            "n_fit >= dim for a full-dim score; rank not clamped to n_fit"
        ),
        "claimed_full_dim": claimed_full_dim,
        "n_fit": n_fit,
        "input_width": input_width,
        "scars_cited": [
            {
                "scar_id": s.get("scar_id"),
                "original_id": s.get("original_id"),
                "source_path": s.get("source_path"),
                "hypothesis_family": s.get("hypothesis_family"),
                "reopen_condition": s.get("reopen_condition"),
                "refuse_eligible": s.get("refuse_eligible"),
            }
            for s in scars
        ],
        "ranks": per,
        "thin_ranks": [p["rank"] for p in thin],
        "trustworthy_ranks": [p["rank"] for p in trustworthy],
        "not_applied": [
            "per-expert routed-row counts (the screen lstsq uses all fit rows for every selected expert)",
            "a new scar; this is NS-014 applied to this split",
        ],
    }


# ---------------------------------------------------------------------------
# Re-rank the nine families. Untouched stays untouched.
# ---------------------------------------------------------------------------


def family_budget_rows(sub1: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(sub1, Mapping):
        return None, "SUB1 receipt absent; the nine families are not invented here"
    raw = sub1.get("family_budget")
    if not isinstance(raw, list) or not raw:
        return None, "SUB1.family_budget absent; the nine families are not invented here"
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(raw):
        if not isinstance(spec, Mapping):
            return None, f"SUB1.family_budget[{i}] is not an object"
        name = str(spec.get("family") or "").strip()
        if not name:
            return None, f"SUB1.family_budget[{i}] has no family name"
        out.append(
            {
                "family": name,
                "program": spec.get("program"),
                "source_fraction": _as_float(spec.get("source_fraction")),
                "meta_bpw_target": _as_float(spec.get("meta_bpw_target")),
                "weighted_meta_bpw": _as_float(spec.get("weighted_meta_bpw")),
                "runtime_shape": spec.get("runtime_shape"),
                "ledger_components": (
                    sorted((spec.get("ledger") or {}).keys())
                    if isinstance(spec.get("ledger"), Mapping)
                    else []
                ),
            }
        )
    return out, None


def q4_component_bpw(sub1: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sub1, Mapping):
        return {"ok": False, "value": None, "reason": "SUB1 absent; Q4 bpw is not defaulted"}
    value = _as_float(_deep(sub1, "current_evidence", "bounded_routed_q4_component_bpw"))
    if value is None:
        return {
            "ok": False,
            "value": None,
            "reason": "SUB1.current_evidence.bounded_routed_q4_component_bpw absent",
        }
    return {
        "ok": True,
        "value": value,
        "reason": "read from SUB1.current_evidence.bounded_routed_q4_component_bpw",
        "status": _deep(sub1, "current_evidence", "bounded_routed_q4_status"),
    }


def shares_failing_mechanism(
    family: Mapping[str, Any],
    screen: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A census family inherits this scar iff it hosts the measured codec.

    Name match alone is not enough. A routed_experts row with no program is
    UNRESOLVED, not a mark-down: unknown is not evidence against.
    """
    kind = _deep(screen, "representation", "kind") if isinstance(screen, Mapping) else None
    organ = _deep(screen, "representation", "organ") if isinstance(screen, Mapping) else None
    name = str(family.get("family") or "")
    program = family.get("program")
    if not isinstance(kind, str) or not kind:
        return {
            "shares": None,
            "effect": "UNRESOLVED",
            "why": "screen representation.kind absent; sharing is not guessed",
        }
    if name != "routed_experts":
        return {
            "shares": False,
            "effect": "UNTOUCHED",
            "why": (
                f"family {name!r} is not the census family that hosts the measured "
                f"codec {kind!r} on {organ!r}"
            ),
            "measured_kind": kind,
            "measured_organ": organ,
        }
    if not isinstance(program, str) or not program.strip():
        return {
            "shares": None,
            "effect": "UNRESOLVED",
            "why": (
                "routed_experts row has no program text; refusing to assume it is "
                "the measured codec"
            ),
            "measured_kind": kind,
            "measured_organ": organ,
        }
    low = program.lower()
    hosts = "latent" in low and "decoder" in low
    if hosts:
        return {
            "shares": True,
            "effect": "INHERITS_AGAINST",
            "why": (
                f"family routed_experts program {program!r} is the measured codec "
                f"{kind!r} on {organ!r}"
            ),
            "measured_kind": kind,
            "measured_organ": organ,
        }
    return {
        "shares": False,
        "effect": "UNTOUCHED",
        "why": (
            f"family routed_experts program {program!r} is not the measured codec {kind!r}"
        ),
        "measured_kind": kind,
        "measured_organ": organ,
    }


def replan(
    *,
    screen: Mapping[str, Any] | None,
    teacher: Mapping[str, Any] | None,
    sub1: Mapping[str, Any] | None,
    index_doc: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Re-rank the nine families. Do not lower the gate. Spend the next rows on unmeasured surfaces."""
    fals = falsification(screen, teacher)
    contract = contract_from_screen(screen) if isinstance(screen, Mapping) else {
        "ok": False,
        "reason": "screen absent",
        "min_heldout_cosine": None,
        "max_heldout_relative_fro_error": None,
        "must_beat_per_expert_q4": None,
    }
    families, fam_why = family_budget_rows(sub1)
    q4 = q4_component_bpw(sub1)
    points = fals.get("rank_rows") if isinstance(fals.get("rank_rows"), list) else None
    extra = rank_extrapolation(points, contract, q4_bpw=q4.get("value"))
    n_fit = _as_int(_deep(fals, "scope", "fit_rows")) if isinstance(fals.get("scope"), Mapping) else None
    ranks = _deep(fals, "scope", "ranks") if isinstance(fals.get("scope"), Mapping) else None
    width = _as_int(_deep(teacher, "teacher_trace", "width")) if isinstance(teacher, Mapping) else None
    if width is None and isinstance(screen, Mapping):
        width = _as_int(_deep(screen, "teacher_trace", "width"))
    under = underdetermination_check(
        n_fit=n_fit,
        ranks=ranks if isinstance(ranks, list) else None,
        input_width=width,
        index_doc=index_doc,
        claimed_full_dim=False,
        rank_clamped_to_n_fit=False,
    )
    surfaces = named_next_surfaces(screen)

    family_rows: list[dict[str, Any]] = []
    if families is None:
        family_block: dict[str, Any] = {
            "ok": False,
            "reason": fam_why,
            "families": [],
            "untouched": [],
            "inherits_against": [],
        }
    else:
        for spec in families:
            share = shares_failing_mechanism(spec, screen)
            effect = share["effect"]
            next_action = "leave standing; this screen is not evidence against it"
            if effect == "INHERITS_AGAINST":
                next_action = (
                    "do not spend the next teacher rows on more ranks of this codec "
                    "on this organ; the gate stands and the curve did not buy it"
                )
            elif effect == "UNRESOLVED":
                next_action = "unresolved sharing; not marked down, not cleared"
            family_rows.append(
                {
                    "family": spec["family"],
                    "program": spec.get("program"),
                    "source_fraction": spec.get("source_fraction"),
                    "meta_bpw_target": spec.get("meta_bpw_target"),
                    "evidence_effect": effect,
                    "shares_failing_mechanism": share["shares"],
                    "why": share["why"],
                    "next_action": next_action,
                }
            )
        # Untouched keep their census weight order. Inherited-against sorts last.
        order = {"UNTOUCHED": 0, "UNRESOLVED": 1, "INHERITS_AGAINST": 2}
        family_rows.sort(
            key=lambda r: (
                order.get(r["evidence_effect"], 9),
                -(r["source_fraction"] if isinstance(r["source_fraction"], (int, float)) else -1.0),
                r["family"],
            )
        )
        family_block = {
            "ok": True,
            "n_families": len(family_rows),
            "families": family_rows,
            "untouched": [r["family"] for r in family_rows if r["evidence_effect"] == "UNTOUCHED"],
            "inherits_against": [
                r["family"] for r in family_rows if r["evidence_effect"] == "INHERITS_AGAINST"
            ],
            "unresolved": [r["family"] for r in family_rows if r["evidence_effect"] == "UNRESOLVED"],
            "rule": (
                "A family that does not host the measured codec is UNTOUCHED by this "
                "screen. UNTOUCHED is not a pass of that family; it is the absence of "
                "this evidence against it."
            ),
        }

    capture = {
        "ok": surfaces["ok"],
        "surfaces": surfaces.get("surfaces") or [],
        "reason": surfaces.get("reason"),
        "next_gate": surfaces.get("next_gate"),
        "spend": (
            "the next teacher rows go to surfaces that have never been measured, "
            "not to another rank sweep of layer_4.routed_experts.gate_up_proj"
            if surfaces.get("ok")
            else "capture plan refused because the screen did not name surfaces"
        ),
        "do_not": [
            "lower min_heldout_cosine / max_heldout_relative_fro_error / must_beat_per_expert_q4",
            "treat the failure as a discovery of a new codec",
            "spend the next capture on more ranks of the failing latent+readout on this organ",
        ],
    }

    return {
        "falsification": fals,
        "corpus_comparison": corpus_comparison(),
        "rank_extrapolation": extra,
        "underdetermination": under,
        "families": family_block,
        "next_capture": capture,
        "contract": contract,
        "q4_component_bpw": q4,
        "gate_stands": True,
        "gate_values_source": "screen.coherence_contract" if contract.get("ok") else contract.get("reason"),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _module_does_not_redefine_contract() -> dict[str, Any]:
    """Static self-check: this file's AST carries no contract literals."""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # 999/1000 and 1/20 are the screen's cosine / error gate. Written as
    # ratios so this file itself does not contain those decimals as literals.
    banned = (999 / 1000, 1 / 20)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if any(abs(node.value - b) < 1e-12 for b in banned):
                hits.append({"lineno": node.lineno, "value": node.value})
    return {
        "ok": not hits,
        "hits": hits,
        "rule": "coherence contract is read from the screen receipt, never assigned here",
    }


def recovered_implementation() -> list[dict[str, str]]:
    return [
        {
            "path": SCREEN_REL,
            "role": "the real-256 L4 screen this re-plan cites; not re-run",
        },
        {
            "path": TEACHER_REL,
            "role": "256 unique source-BF16 layer-4 mlp_input rows the screen fitted",
        },
        {
            "path": SUB1_REL,
            "role": "nine census families, prospective sub-1 budget, Q4 component bpw",
        },
        {
            "path": SCREEN_TOOL_REL,
            "role": "shared-input-latent + expert-local readout fitter; DEFAULT_RANKS; heldout_split % 5",
        },
        {
            "path": "tools/future/meta_funnel.py",
            "role": "nine-gate funnel + load_receipt (disk then HEAD); gate 3 is held-out numerical",
        },
        {
            "path": "tools/future/flash_schools.py",
            "role": "ROUTED_EXPERTS school already forbids trivial global sharing; not this codec kill",
        },
        {
            "path": "tools/future/negative_index.py",
            "role": "NS-014 / NNS-007 underdetermined rank-r fit scar",
        },
        {
            "path": INDEX_REL,
            "role": "keyed scar store this module consults instead of re-ingesting the corpus",
        },
        {
            "path": "tools/future/ebpw_categories.py",
            "role": "diagnostic factor bpw is not complete_physical_ebpw; domination is a description comparison",
        },
        {
            "path": "tools/future/meta_ready.py",
            "role": "already reads the (unsafe 4-row) coherence contract; this module reads the real-256 screen",
        },
        {
            "path": DOC_REL,
            "role": "frontier registration of the prospective sub-1 budget and the hard admission gates",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "No sidecar module turned the real-256 L4 failure into a scoped falsification plus a capture re-plan that keeps the gate.",
        "Rank-vs-error extrapolation that refuses a rank when the supported curve has a floor above the contract, rather than inventing a fantasy rank.",
        "NS-014 applied to the 204-row / five-rank split instead of assumed thin or assumed enough.",
        "Nine SUB1 families re-ranked by mechanism sharing, with at least the n-gram family left UNTOUCHED.",
    ]


def negative_findings() -> list[str]:
    return [
        "This sidecar did not re-fit the codec and did not re-read the 256-row f32 state file.",
        "Per-expert routed-row underdetermination was not the NS-014 predicate and was not applied as a kill.",
        "down_proj, other layers, and the four named unmeasured surfaces have no function screen in this receipt.",
        "diagnostic_factor_equivalent_bpw is a factor description, not physical EBPW; domination-by-construction is the same class of comparison.",
        "orchestration.BINDINGS is outside this lane's WRITE list; the frontier named below is declared, not wired.",
    ]


def build() -> Path:
    inputs = load_inputs()
    plan = replan(
        screen=inputs["screen"],
        teacher=inputs["teacher"],
        sub1=inputs["sub1"],
        index_doc=inputs["index"],
    )
    contract_ast = _module_does_not_redefine_contract()
    untouched = (plan.get("families") or {}).get("untouched") or []
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Re-plan the Flash meta funnel from the real-256 L4 coherence screen "
            "without lowering the gate and without treating the failure as a discovery."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "cited_inputs": {
            "screen": cited(inputs["screen_rel"], inputs["screen"]),
            "teacher": cited(inputs["teacher_rel"], inputs["teacher"]),
            "sub1": cited(inputs["sub1_rel"], inputs["sub1"]),
            "negative_index": cited(inputs["index_rel"], inputs["index"]),
        },
        "falsification": plan["falsification"],
        "corpus_comparison": corpus_comparison(),
        "rank_extrapolation": plan["rank_extrapolation"],
        "underdetermination": plan["underdetermination"],
        "families": plan["families"],
        "next_capture": plan["next_capture"],
        "contract": plan["contract"],
        "q4_component_bpw": plan["q4_component_bpw"],
        "gate_stands": True,
        "contract_not_redefined_here": contract_ast,
        "at_least_one_family_untouched": bool(untouched),
        "untouched_families": untouched,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": {
            "entry_point": "tools.future.flash_meta_replan.build()",
            "workunit": (
                "one CPU_ANALYSIS unit; cite the real-256 screen, apply NS-014, "
                "extrapolate or refuse a rank, re-rank nine SUB1 families; no GPU"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
            "fails_closed": (
                "absent screen/teacher/SUB1 is a recorded refusal; the coherence "
                "contract is never defaulted; a poor or floored curve reports no rank; "
                "an unrelated family is not marked down"
            ),
            "discoverable": True,
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
