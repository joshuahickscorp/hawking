"""Eleven-axis campaign Pareto frontier, extended from G033 not forked.

`pareto_disposition.py` already has six axes, `dominates()`, `frontier()` and
`disposition()`. Those six are NOT a strict subset of the eleven this campaign
must carry: G033 includes `active_bytes_per_token`, which is bytes of weights
touched per generated token, not resident memory. Five of the six map cleanly:

    capability_passes          -> capability
    complete_ebpw              -> complete_ebpw
    decode_tok_s               -> tps
    representation_complexity  -> representation_complexity
    transfer_value             -> transfer_value

G033's `dominates()` skips any axis where either side is None. That lets a
candidate that never measured TPS dominate one that measured a poor TPS, which
is exactly the "quietly rewards not measuring" failure this module exists to
close. We reuse `disposition()` and the frontier shape, and replace domination
with the coverage-subset rule below.

Partial-candidate rule (coverage-subset)
---------------------------------------
An absent axis is not a zero and it is not a pass. It is not a comparable
value.

    A dominates B iff
      (1) A and B are the same organism (a dead O003 binary does not knock a
          different lake body off the frontier),
      (2) every axis on which B has a comparable value, A also has one
          (a candidate missing TPS cannot dominate one that measured TPS),
      (3) A is no worse than B on each of those axes,
      (4) A is strictly better on at least one,
      (5) a capability failure never dominates a capability pass (G033 §
          inherited; also follows from (3) once capability is in the support).

A candidate with no comparable axis at all is unscored, not a frontier point:
there is nothing to be better or worse on. A candidate that measured some
axes and not others still enters the frontier computation; it is excluded
only when something covers those axes and is better on them, never because
the missing ones were treated as zeros.

Epistemic labels stay distinct. A representation being BUILT is not it being
CAPABILITY-PRESERVING. Gravity `CANDIDATE_PASS` is a Doctor verdict, not the
ppl ∧ 4-gram gate. Gravity `tps_specimen` is a per-specimen constant, not a
per-variant measurement, and is not ingested as TPS.

The frontier is recomputed from harvested points every call. A stored
`frontier` list in a receipt is never the result.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from pareto_disposition import (  # noqa: E402
    AXES as G033_AXES,
    disposition,
)
from _common import (  # noqa: E402
    RECEIPTS,
    UnknownFlag,
    blob_text,
    git,
    prefetch_blobs,
    require_known_flags,
)

# The eleven axes the campaign must carry. Order is the contract's order.
CAMPAIGN_AXES: tuple[str, ...] = (
    "capability",
    "complete_ebpw",
    "tps",
    "prefill",
    "resident_memory",
    "temporary_memory",
    "state",
    "reliability",
    "role_suitability",
    "representation_complexity",
    "transfer_value",
)

# Lower is better / higher is better. capability is higher (pass > fail).
LOWER_BETTER = frozenset({
    "complete_ebpw", "resident_memory", "temporary_memory",
    "state", "representation_complexity",
})
HIGHER_BETTER = frozenset({
    "capability", "tps", "prefill", "reliability",
    "role_suitability", "transfer_value",
})

# Six distinct states. Collapsing any pair is a recognised failure here.
EPISTEMIC: tuple[str, ...] = (
    "architecturally_supported",
    "implemented",
    "wired",
    "accepted",
    "verified",
    "physically_measured",
)
_EPISTEMIC_RANK = {s: i for i, s in enumerate(EPISTEMIC)}

ALIASES = {
    "capability_passes": "capability",
    "capability_ok": "capability",
    "passes_gate": "capability",
    "passes": "capability",
    "decode_tok_s": "tps",
    "prefill_tok_s": "prefill",
    "complete_bpw": "complete_ebpw",
    "ebpw": "complete_ebpw",
    "stored_bpw": "complete_ebpw",
}

# G033's extra axis is traffic-per-token, not resident RAM. We refuse the
# alias so a byte-of-weights figure cannot masquerade as a memory axis.
NOT_RESIDENT_MEMORY = frozenset({"active_bytes_per_token"})

PARTIAL_RULE = (
    "coverage-subset: A dominates B only when they share an organism, A has a "
    "comparable value for every axis B has one for, A is no worse on each of "
    "those, and A is strictly better on at least one. An absent axis is not a "
    "zero and is not a pass; a candidate missing TPS cannot dominate one that "
    "measured TPS, and a partial candidate is excluded only if something "
    "covers the axes it did measure and is better on them."
)

# O003 gate, loaded from the receipt that published it. Not a guess.
_O003_GATE: dict[str, float] | None = None

GiB = 1 << 30


# ---------------------------------------------------------------------------
# comparable values
# ---------------------------------------------------------------------------

def has_value(point: dict, axis: str) -> bool:
    """False and 0.0 are values. None and a missing key are not."""
    if axis not in point:
        return False
    return point[axis] is not None


def measured_support(point: dict) -> frozenset[str]:
    return frozenset(a for a in CAMPAIGN_AXES if has_value(point, a))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _no_worse(axis: str, a: Any, b: Any) -> bool | None:
    """True if a is no worse than b; False if worse; None if incomparable."""
    x, y = _as_number(a), _as_number(b)
    if x is None or y is None:
        return None
    if axis in LOWER_BETTER:
        return x <= y
    if axis in HIGHER_BETTER:
        return x >= y
    return None


def _strictly_better(axis: str, a: Any, b: Any) -> bool:
    x, y = _as_number(a), _as_number(b)
    if x is None or y is None:
        return False
    if axis in LOWER_BETTER:
        return x < y
    if axis in HIGHER_BETTER:
        return x > y
    return False


def dominates(a: dict, b: dict) -> bool:
    """Coverage-subset domination. See module docstring."""
    if a is b:
        return False
    org_a, org_b = a.get("organism"), b.get("organism")
    if org_a is None or org_b is None or org_a != org_b:
        return False

    # G033: a capability failure cannot dominate a pass, even before numbers.
    if has_value(b, "capability") and b.get("capability") and (
            not has_value(a, "capability") or not a.get("capability")):
        return False

    support_b = measured_support(b)
    if not support_b:
        return False
    support_a = measured_support(a)
    if not support_b.issubset(support_a):
        return False

    better_any = False
    for axis in support_b:
        nw = _no_worse(axis, a.get(axis), b.get(axis))
        if nw is None or nw is False:
            return False
        if _strictly_better(axis, a.get(axis), b.get(axis)):
            better_any = True
    return better_any


def frontier(points: Iterable[dict]) -> list[dict]:
    """Undominated points. Same shape as G033; uses this module's dominates."""
    pts = list(points)
    return [p for p in pts if not any(dominates(q, p) for q in pts if q is not p)]


def g033_is_strict_subset() -> bool:
    """False: G033 carries active_bytes_per_token, which is not a campaign axis."""
    mapped = {
        "capability_passes": "capability",
        "complete_ebpw": "complete_ebpw",
        "decode_tok_s": "tps",
        "representation_complexity": "representation_complexity",
        "transfer_value": "transfer_value",
    }
    extra = [a for a in G033_AXES if a not in mapped]
    return extra == [] and all(v in CAMPAIGN_AXES for v in mapped.values())


# ---------------------------------------------------------------------------
# harvest / merge
# ---------------------------------------------------------------------------

def _receipt_name(source: str) -> str:
    return Path(source).name


MEASURED, REFUSED, OWED = "MEASURED", "REFUSED", "OWED"


def _new_point(name: str, organism: str, kind: str, source: str) -> dict:
    return {
        "id": f"{organism}:{name}",
        "name": name,
        "organism": organism,
        "kind": kind,
        "axis_meta": {a: {"epistemic": None, "receipts": [], "state": OWED,
                          "reason": None}
                      for a in CAMPAIGN_AXES},
        "provenance": [source],
        "conflicts": [],
    }


def _rank(epistemic: str) -> int:
    return _EPISTEMIC_RANK.get(epistemic, -1)


def _set_axis(point: dict, axis: str, value: Any, *, epistemic: str,
              source: str, state: str = "MEASURED") -> None:
    if axis not in CAMPAIGN_AXES:
        return
    if value is None:
        return
    if isinstance(value, float) and value != value:  # NaN
        return
    meta = point["axis_meta"].setdefault(axis, {
        "epistemic": None, "receipts": [], "state": OWED, "reason": None,
    })
    incoming = _rank(epistemic)
    current = _rank(meta["epistemic"]) if meta["epistemic"] else -1
    src = _receipt_name(source)
    if incoming > current:
        if has_value(point, axis) and point[axis] != value:
            point["conflicts"].append({
                "axis": axis, "kept": value, "displaced": point[axis],
                "kept_from": src, "displaced_from": meta["receipts"][-1:],
            })
        point[axis] = value
        meta["epistemic"] = epistemic
        meta["state"] = state
    elif incoming == current and has_value(point, axis):
        old = point[axis]
        if isinstance(old, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
            if old != 0 and abs(float(value) - old) / max(abs(old), 1e-12) > 1e-3:
                point["conflicts"].append({
                    "axis": axis, "kept": old, "other": value,
                    "kept_epistemic": epistemic, "other_from": src,
                })
        elif old != value and not (isinstance(old, float) and isinstance(value, (int, float))):
            point["conflicts"].append({
                "axis": axis, "kept": old, "other": value, "other_from": src,
            })
    if src not in meta["receipts"]:
        meta["receipts"].append(src)
    if source not in point["provenance"]:
        point["provenance"].append(source)
    if meta.get("state") == MEASURED:
        meta["reason"] = None


def disclose_axes(point: dict) -> dict:
    """G011: every axis is populated or explicitly absent. Never a dropped key.

    MEASURED — a comparable value, with receipts
    REFUSED  — a named mechanism saying why this axis cannot be a value here
    OWED     — nothing yet; outstanding work. Not a zero and not a pass.
    """
    meta = point.setdefault("axis_meta", {})
    for axis in CAMPAIGN_AXES:
        slot = meta.setdefault(axis, {
            "epistemic": None, "receipts": [], "state": OWED, "reason": None,
        })
        if has_value(point, axis):
            slot["state"] = MEASURED
            continue
        if slot.get("state") == REFUSED and slot.get("reason"):
            continue
        slot["state"] = OWED
        if not slot.get("reason"):
            slot["reason"] = (
                f"{axis} has no comparable value on disk for {point.get('id')}; "
                "absent is not zero and is not a pass"
            )
    return point


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _floatish(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value == value:
        return float(value)
    return None


def _load_o003_gate() -> dict[str, float]:
    global _O003_GATE
    if _O003_GATE is not None:
        return _O003_GATE
    path = RECEIPTS / "O003_PARETO_COMPLETE.json"
    gate = {"ppl_max": 3.617, "median_4gram_max": 0.1614}
    if path.is_file():
        try:
            doc = json.loads(path.read_text())
            g = doc.get("gate") or {}
            if "ppl_max" in g:
                gate["ppl_max"] = float(g["ppl_max"])
            if "median_4gram_max" in g:
                gate["median_4gram_max"] = float(g["median_4gram_max"])
        except (OSError, ValueError, TypeError):
            pass
    _O003_GATE = gate
    return gate


def _capability_from(obj: dict) -> tuple[bool | None, str]:
    """Return (value, epistemic). Conjunction of ppl and 4-gram when both exist.

    A lone ppl_passes is not capability: PQ cleared perplexity and failed
    generation. Built / CANDIDATE_PASS / execution_complete is not capability.
    """
    ppl = _floatish(obj.get("ppl"))
    r4 = _floatish(obj.get("median_4gram"))
    if r4 is None:
        r4 = _floatish(obj.get("median_4gram_repeat"))
    if r4 is None:
        r4 = _floatish(obj.get("r4"))
    if ppl is not None and r4 is not None:
        g = _load_o003_gate()
        return (ppl <= g["ppl_max"] and r4 <= g["median_4gram_max"],
                "physically_measured")
    for key in ("passes_gate", "capability_passes", "capability_ok", "passes"):
        b = _boolish(obj.get(key))
        if b is not None:
            # A boolean with no gate numbers is accepted, not verified.
            return b, "accepted"
    return None, "accepted"


def _canon_spec(spec: str) -> str:
    s = spec.strip()
    low = s.lower().replace("_", " ")
    if low in {"bf16", "bf16-source", "bf16 source", "source bf16", "source_bf16"}:
        return "bf16"
    return s


def complexity_of(spec: str, klass: str | None = None) -> int | None:
    """Ordinal from the representation class. Architecturally supported, not measured."""
    s = (spec or "").lower()
    k = (klass or "").lower()
    blob = s + " " + k
    if "sharedbasis" in blob or "shared_basis" in blob:
        return 4
    if blob.startswith("pq") or " pq" in blob or "product" in k:
        return 3
    if "outlier" in blob or "hotcold" in blob:
        return 2
    if "resbinary" in blob or "residual" in k:
        return 2
    if "binary" in blob or blob.startswith("sparse") or "sparse" in k:
        return 2
    if s.startswith("q") or s.startswith("mixed-q") or "affine" in k:
        return 1
    if "bf16" in blob or k in {"source_bf16", "source"}:
        return 0
    return None


def _organism_maps() -> tuple[dict[str, str], dict[str, str]]:
    """oxx -> slug-ish, and hf-id -> oxx, from the Odyssey manifest."""
    oxx_to_src: dict[str, str] = {}
    src_to_oxx: dict[str, str] = {}
    path = REPO / "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json"
    if not path.is_file():
        return oxx_to_src, src_to_oxx
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return oxx_to_src, src_to_oxx
    if not isinstance(rows, list):
        return oxx_to_src, src_to_oxx
    for row in rows:
        if not isinstance(row, dict):
            continue
        oxx = row.get("oxx")
        src = row.get("canonical_source") or row.get("model")
        if oxx and src:
            oxx_to_src[str(oxx)] = str(src)
            src_to_oxx[str(src).lower()] = str(oxx)
            src_to_oxx[str(src).split("/")[-1].lower()] = str(oxx)
    return oxx_to_src, src_to_oxx


def _organism_from_slug(slug: str, src_to_oxx: dict[str, str]) -> str:
    # moonshotai--Kimi-VL-A3B-Instruct@hash
    body = slug.split("@", 1)[0].replace("--", "/", 1)
    low = body.lower()
    if low in src_to_oxx:
        return src_to_oxx[low]
    name = body.split("/")[-1].lower()
    if name in src_to_oxx:
        return src_to_oxx[name]
    return slug


class _Pool:
    def __init__(self) -> None:
        self.by_id: dict[str, dict] = {}
        self.oxx_to_src, self.src_to_oxx = _organism_maps()
        self.notes: list[str] = []

    def get(self, organism: str, name: str, kind: str, source: str) -> dict:
        cid = f"{organism}:{name}"
        p = self.by_id.get(cid)
        if p is None:
            p = _new_point(name, organism, kind, source)
            self.by_id[cid] = p
        else:
            if source not in p["provenance"]:
                p["provenance"].append(source)
        return p

    def ingest_representation(self, obj: dict, source: str,
                              organism: str | None = None) -> dict | None:
        spec = obj.get("spec") or obj.get("config") or obj.get("name")
        if not spec or not isinstance(spec, str):
            return None
        spec = spec.split("(")[0].strip().split()[0]
        spec = _canon_spec(spec)
        if not spec or spec.endswith(":"):
            return None
        oxx = organism or obj.get("oxx") or obj.get("specimen") or "O003"
        oxx = str(oxx)
        point = self.get(oxx, spec, "representation", source)

        ebpw = None
        for k in ("complete_ebpw", "complete_bpw", "ebpw", "stored_bpw"):
            ebpw = _floatish(obj.get(k))
            if ebpw is not None:
                break
        if ebpw is not None and 0.0 < ebpw <= 64.0:
            _set_axis(point, "complete_ebpw", ebpw,
                      epistemic="physically_measured", source=source)

        cap, cap_ep = _capability_from(obj)
        if cap is not None:
            _set_axis(point, "capability", cap, epistemic=cap_ep, source=source)

        tps = _floatish(obj.get("decode_tok_s") or obj.get("tps"))
        # Gravity's per-specimen constant must never land here. Dedicated
        # gravity ingest skips it; a generic walk that sees tps_specimen
        # still has to refuse. COMPLETE/MULTIAXIS decode numbers are not
        # ingested: GAUNTLET (the cited source) carries none, and MEASURED
        # disagrees with COMPLETE on the arms both list.
        if tps is not None and "tps_specimen" not in obj:
            if "PARETO_MEASURED" in source or "TPS_BASELINE" in source:
                _set_axis(point, "tps", tps, epistemic="physically_measured",
                          source=source)

        pre = _floatish(obj.get("prefill_tok_s") or obj.get("prefill"))
        if pre is not None:
            if "PARETO_MEASURED" in source or "TPS_BASELINE" in source:
                _set_axis(point, "prefill", pre, epistemic="physically_measured",
                          source=source)

        active_gb = _floatish(obj.get("active_gb"))
        peak_gb = _floatish(obj.get("peak_gb"))
        if active_gb is not None:
            _set_axis(point, "resident_memory", active_gb * GiB,
                      epistemic="physically_measured", source=source)
        if active_gb is not None and peak_gb is not None and peak_gb > active_gb:
            _set_axis(point, "temporary_memory", (peak_gb - active_gb) * GiB,
                      epistemic="physically_measured", source=source)

        cx = obj.get("representation_complexity")
        if isinstance(cx, int):
            _set_axis(point, "representation_complexity", cx,
                      epistemic="accepted", source=source)
        else:
            derived = complexity_of(spec, obj.get("class") or obj.get("representation_class"))
            if derived is not None:
                _set_axis(point, "representation_complexity", derived,
                          epistemic="architecturally_supported", source=source)

        tv = obj.get("transfer_value")
        if isinstance(tv, (int, float)) and not isinstance(tv, bool):
            _set_axis(point, "transfer_value", float(tv),
                      epistemic="accepted", source=source)

        ppl = _floatish(obj.get("ppl"))
        r4 = _floatish(obj.get("median_4gram"))
        if r4 is None:
            r4 = _floatish(obj.get("median_4gram_repeat"))
        if r4 is None:
            r4 = _floatish(obj.get("r4"))
        if ppl is not None:
            point["ppl"] = ppl
        if r4 is not None:
            point["median_4gram"] = r4
        return point


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _walk_candidate_dicts(obj: Any) -> Iterator[dict]:
    if isinstance(obj, dict):
        spec = obj.get("spec") or obj.get("config") or obj.get("name")
        ebpw_present = any(k in obj for k in
                           ("complete_ebpw", "complete_bpw", "ebpw", "stored_bpw"))
        if isinstance(spec, str) and ebpw_present:
            yield obj
        for k, v in obj.items():
            if k in {"accounting", "bench", "tensors", "disk_tensors",
                     "protected_components", "battery"}:
                continue
            yield from _walk_candidate_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_candidate_dicts(v)


def harvest_census(pool: _Pool) -> None:
    path = RECEIPTS / "G034_LAKE_CENSUS_FULL.json"
    doc = _read_json(path)
    if not isinstance(doc, dict):
        pool.notes.append("G034_LAKE_CENSUS_FULL.json unreadable")
        return
    src = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    for row in doc.get("rows") or []:
        if not isinstance(row, dict) or not row.get("slug"):
            continue
        slug = row["slug"]
        organism = _organism_from_slug(slug, pool.src_to_oxx)
        # Lake parent of an Odyssey patient merges with that patient's bf16.
        if organism.startswith("O") and len(organism) == 4:
            name, kind = "bf16", "representation"
        else:
            name, kind = slug, "lake_body"
        point = pool.get(organism, name, kind, src)
        ebpw = _floatish(row.get("complete_ebpw"))
        # Census is the parent default. A measured NR/execution ebpw already on
        # the point is more specific and must not be replaced by dtype-16.0.
        if ebpw is not None and not has_value(point, "complete_ebpw"):
            _set_axis(point, "complete_ebpw", ebpw,
                      epistemic="physically_measured", source=src)
        point.setdefault("slug", slug)
        point.setdefault("klass", row.get("klass"))
        point.setdefault("gib", row.get("gib"))


def harvest_ledger(pool: _Pool) -> None:
    """Overlay MEASURED/REFUSED/OWED. A REFUSED axis is not a comparable value."""
    path = RECEIPTS / "G034_ODYSSEY_LEDGER.json"
    doc = _read_json(path)
    if not isinstance(doc, dict):
        pool.notes.append("G034_ODYSSEY_LEDGER.json unreadable")
        return
    src = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    for row in doc.get("specimens") or []:
        if not isinstance(row, dict) or not row.get("slug"):
            continue
        slug = row["slug"]
        organism = _organism_from_slug(slug, pool.src_to_oxx)
        if organism.startswith("O") and len(organism) == 4:
            name, kind = "bf16", "representation"
        else:
            name, kind = slug, "lake_body"
        point = pool.get(organism, name, kind, src)
        axes = row.get("axes") or {}
        ebpw = axes.get("ebpw") or {}
        if (ebpw.get("state") == "MEASURED" and _floatish(ebpw.get("value")) is not None
                and not has_value(point, "complete_ebpw")):
            _set_axis(point, "complete_ebpw", float(ebpw["value"]),
                      epistemic="physically_measured",
                      source=ebpw.get("receipt") or src)
        elif ebpw.get("state") == "REFUSED":
            slot = point["axis_meta"].setdefault("complete_ebpw", {
                "epistemic": None, "receipts": [], "state": OWED, "reason": None,
            })
            if not has_value(point, "complete_ebpw"):
                slot["state"] = REFUSED
                slot["reason"] = ebpw.get("reason")
        tps = axes.get("tps") or {}
        if tps.get("state") == "REFUSED":
            slot = point["axis_meta"].setdefault("tps", {
                "epistemic": None, "receipts": [], "state": OWED, "reason": None,
            })
            if not has_value(point, "tps"):
                slot["state"] = REFUSED
                slot["reason"] = tps.get("reason")
        elif tps.get("state") == "MEASURED" and _floatish(tps.get("value")) is not None:
            # Ledger TPS would be per-body, not Gravity's constant. Still require
            # a receipt; odyssey_ledger already refuses a value without one.
            _set_axis(point, "tps", float(tps["value"]),
                      epistemic="physically_measured",
                      source=tps.get("receipt") or src)


def harvest_measured_execution(pool: _Pool) -> None:
    """O003_PARETO_MEASURED: the first per-variant decode/prefill/memory numbers."""
    path = RECEIPTS / "O003_PARETO_MEASURED.json"
    doc = _read_json(path)
    if not isinstance(doc, dict):
        return
    src = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    for row in doc.get("frontier") or []:
        # These dicts are MEASUREMENTS of configs, not a stored campaign frontier.
        if not isinstance(row, dict):
            continue
        cfg = dict(row)
        cfg["spec"] = row.get("config")
        cfg["specimen"] = "O003"
        pool.ingest_representation(cfg, src, organism="O003")


def harvest_tps_baseline(pool: _Pool) -> None:
    path = RECEIPTS / "O003_TPS_BASELINE.json"
    doc = _read_json(path)
    if not isinstance(doc, dict):
        return
    src = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    tps = (doc.get("BASELINE_TPS") or {})
    res = (doc.get("RESIDENCY") or {})
    obj = {
        "spec": "bf16",
        "specimen": "O003",
        "decode_tok_s": tps.get("decode_tok_s"),
        "prefill_tok_s": tps.get("prefill_tok_s_marginal_576_64"),
        "active_gb": res.get("active_gb"),
        "peak_gb": res.get("peak_gb"),
        "class": "SOURCE_BF16",
    }
    pool.ingest_representation(obj, src, organism="O003")


def harvest_json_file(pool: _Pool, path: Path, organism: str | None = None) -> None:
    doc = _read_json(path)
    if doc is None:
        return
    src = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    if path.name == "O003_PARETO_MULTIAXIS.json" and isinstance(doc, dict):
        # POINTS only. The stored "frontier" list is not the result.
        for row in doc.get("points") or []:
            if isinstance(row, dict):
                pool.ingest_representation(row, src, organism="O003")
        return
    if path.name == "O003_PARETO_COMPLETE.json" and isinstance(doc, dict):
        for row in doc.get("points") or []:
            if isinstance(row, dict):
                pool.ingest_representation(row, src, organism="O003")
        return
    if path.name == "O003_PARETO_MEASURED.json":
        return  # dedicated; would otherwise walk the stored "frontier" key
    for obj in _walk_candidate_dicts(doc):
        pool.ingest_representation(obj, src, organism=organism)


def harvest_named_o003(pool: _Pool) -> None:
    names = (
        "O003_PARETO_MULTIAXIS.json",
        "O003_PARETO_COMPLETE.json",
        "O003_GAP_PROBE.json",
        "O003_PQ_GEOMETRY_TRADE.json",
        "O003_PQ_PERCAL_NEAR_MISS.json",
        "O003_PQ_SCAR.json",
        "O003_PQMIX_REFUTED.json",
        "O003_PQ_SPARSE_AND_ACCOUNTING_FIX.json",
        "O003_SHAREDBASIS_MEASURED.json",
        "O003_BINARY_SCALED.json",
        "O003_SPARSE_BINARY.json",
        "O003_RESIDUAL_BINARY.json",
        "O003_OUTLIER_SPLIT.json",
        "O003_RESIDENT_GROUP_LEVER.json",
        "O003_NON_EXPERT_FLOOR.json",
        "O003_HOT_COLD.json",
        "O003_GAUNTLET_DEEP.json",
        "O003_MARGIN_METRIC_REFUTED.json",
        "REPRESENTATION_VOCABULARY.json",
    )
    for name in names:
        p = RECEIPTS / name
        if p.is_file():
            harvest_json_file(pool, p, organism="O003")


def harvest_gravity(pool: _Pool) -> int:
    """Byte accounting from Gravity receipts. TPS and Doctor verdict are refused.

    Gravity receipts live under receipts/odyssey-i/ and may be absent from a
    sparse worktree; we read HEAD via git. `tps_specimen` is a per-specimen
    CONSTANT (O003=51.627 for every variant) — O003_PARETO_MEASURED recorded
    that no variant had been executed. CANDIDATE_PASS is Doctor, not the
    capability gate.
    """
    listing = git("ls-tree", "-r", "--name-only", "HEAD")
    paths = [p for p in listing.splitlines()
             if p.startswith("receipts/") and "GRAVITY" in p and p.endswith(".json")]
    if not paths:
        # Sparse or git-less: try on-disk glob without assuming a filename.
        for root, _dirs, files in os.walk(REPO / "receipts"):
            for fn in files:
                if "GRAVITY" in fn and fn.endswith(".json"):
                    paths.append(str(Path(root, fn).relative_to(REPO)))
    if not paths:
        pool.notes.append("no Gravity receipts found on disk or in HEAD")
        return 0
    prefetch_blobs([f"HEAD:{p}" for p in paths if not (REPO / p).is_file()])
    n = 0
    for rel in paths:
        on_disk = REPO / rel
        if on_disk.is_file():
            doc = _read_json(on_disk)
        else:
            raw = blob_text(f"HEAD:{rel}")
            try:
                doc = json.loads(raw) if raw else None
            except ValueError:
                doc = None
        if not isinstance(doc, dict):
            continue
        spec = doc.get("spec")
        oxx = doc.get("oxx")
        if not spec or not oxx:
            continue
        ebpw = _floatish(doc.get("complete_bpw"))
        if ebpw is None:
            ebpw = _floatish((doc.get("accounting") or {}).get("complete_bpw"))
        if ebpw is None:
            ebpw = _floatish(doc.get("stored_bpw"))
        if ebpw is None:
            continue
        # Deliberately do not read baseline.tps_specimen or verdict.
        obj = {
            "spec": spec,
            "oxx": oxx,
            "complete_ebpw": ebpw,
            "class": doc.get("conventionality_mechanism") or "affine",
        }
        pool.ingest_representation(obj, rel, organism=str(oxx))
        n += 1
    return n


def harvest(receipts_dir: Path | None = None) -> dict[str, Any]:
    """Build candidates from what is on disk (and Gravity via git). Always live."""
    global RECEIPTS
    saved = RECEIPTS
    if receipts_dir is not None:
        RECEIPTS = Path(receipts_dir)
    try:
        return _harvest()
    finally:
        RECEIPTS = saved


def _harvest() -> dict[str, Any]:
    pool = _Pool()
    # Representation receipts first so census dtype-16.0 cannot overwrite a
    # measured complete_ebpw on the same organism's bf16 point.
    harvest_measured_execution(pool)
    harvest_tps_baseline(pool)
    harvest_named_o003(pool)
    n_gravity = harvest_gravity(pool)
    harvest_census(pool)
    harvest_ledger(pool)

    candidates = list(pool.by_id.values())
    for p in candidates:
        disclose_axes(p)
    scored = [p for p in candidates if measured_support(p)]
    unscored = [p for p in candidates if not measured_support(p)]
    front = frontier(scored)
    front_ids = {p["id"] for p in front}

    for p in scored:
        p["on_frontier"] = p["id"] in front_ids
        contrib = []
        if has_value(p, "transfer_value") and p["transfer_value"]:
            contrib.append(f"transfer_value={p['transfer_value']}")
        p["contributions"] = contrib
        p["disposition"] = disposition(p)

    by_org: dict[str, list[dict]] = {}
    for p in front:
        by_org.setdefault(p["organism"], []).append(p)
    for rows in by_org.values():
        rows.sort(key=lambda r: (
            0 if r.get("capability") else 1,
            _floatish(r.get("complete_ebpw")) if has_value(r, "complete_ebpw") else 99.0,
            r["name"],
        ))

    debt = measurement_debt(scored)
    return {
        "schema": "campaign-pareto-1",
        "partial_rule": PARTIAL_RULE,
        "g033_axes_are_strict_subset": g033_is_strict_subset(),
        "g033_axes": list(G033_AXES),
        "campaign_axes": list(CAMPAIGN_AXES),
        "epistemic_states": list(EPISTEMIC),
        "n_candidates": len(candidates),
        "n_scored": len(scored),
        "n_unscored": len(unscored),
        "n_frontier": len(front),
        "n_gravity_receipts": n_gravity,
        "scored": [_public(p) for p in scored],
        "frontier": [_public(p) for p in front],
        "by_organism": {k: [_public(p) for p in v] for k, v in sorted(by_org.items())},
        "unscored": [_public(p) for p in unscored],
        "measurement_debt": debt,
        "notes": pool.notes,
        "recomputed": True,
    }


def _public(p: dict) -> dict:
    out = {
        "id": p["id"],
        "name": p["name"],
        "organism": p["organism"],
        "kind": p["kind"],
        "on_frontier": p.get("on_frontier"),
        "support": sorted(measured_support(p)),
        "values": {a: p[a] for a in CAMPAIGN_AXES if has_value(p, a)},
        "axis_meta": p.get("axis_meta") or {},
        "provenance": p.get("provenance") or [],
        "conflicts": p.get("conflicts") or [],
        "disposition": p.get("disposition"),
    }
    for extra in ("slug", "klass", "gib", "ppl", "median_4gram"):
        if extra in p:
            out[extra] = p[extra]
    return out


def measurement_debt(candidates: Iterable[dict]) -> dict[str, Any]:
    counts = {a: 0 for a in CAMPAIGN_AXES}
    n = 0
    for p in candidates:
        n += 1
        for a in CAMPAIGN_AXES:
            if has_value(p, a):
                counts[a] += 1
    nothing = [a for a, c in counts.items() if c == 0]
    return {
        "n_scored_candidates": n,
        "n_with_axis": counts,
        "unmeasured_for_everyone": nothing,
        "why_unmeasured_is_debt": (
            "an axis that no candidate has a comparable value for cannot "
            "discriminate, cannot dominate, and cannot be treated as passed. "
            "That list is the campaign's real measurement debt."
        ),
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def _fmt_val(axis: str, value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if axis in {"resident_memory", "temporary_memory"} and isinstance(value, (int, float)):
        return f"{value / GiB:.2f}GiB"
    if isinstance(value, float):
        return f"{value:.4f}" if value < 1000 else f"{value:.1f}"
    return str(value)


def format_report(doc: dict) -> str:
    lines = [
        "CAMPAIGN PARETO FRONTIER (recomputed from receipts, not a stored list)",
        f"rule: {doc['partial_rule']}",
        f"G033 six axes strict subset of eleven: {doc['g033_axes_are_strict_subset']}"
        " (active_bytes_per_token is the extra G033 axis; it is not resident memory)",
        f"candidates: {doc['n_candidates']}  scored: {doc['n_scored']}  "
        f"frontier: {doc['n_frontier']}  unscored: {doc['n_unscored']}  "
        f"gravity receipts: {doc['n_gravity_receipts']}",
        "",
        "measurement debt — axes with a comparable value on NOTHING:",
    ]
    debt = doc["measurement_debt"]
    nothing = debt["unmeasured_for_everyone"]
    lines.append("  " + (", ".join(nothing) if nothing else "(none — every axis has at least one value)"))
    lines.append("  counts: " + ", ".join(f"{a}={debt['n_with_axis'][a]}" for a in CAMPAIGN_AXES))
    lines.append("")
    for org, rows in doc["by_organism"].items():
        lake = [r for r in rows if r["kind"] == "lake_body"]
        reps = [r for r in rows if r["kind"] != "lake_body"]
        if lake and not reps:
            continue  # summarised below
        lines.append(f"--- {org}  {len(rows)} frontier point(s) ---")
        for r in rows:
            vals = " ".join(
                f"{a}={_fmt_val(a, r['values'][a])}" for a in CAMPAIGN_AXES
                if a in r["values"]
            )
            prov = ",".join(_receipt_name(p) for p in r["provenance"][:4])
            lines.append(f"  {r['name']:28} {vals}")
            lines.append(f"    support={r['support']}  provenance={prov}")
        lines.append("")
    watch = (
        ("q3-g128-experts", "lowest capability-preserving (~3.2831)"),
        ("pqpercal4k512", "likelihood good, diversity/repetition failing (~2.4059)"),
        ("pqpercal2k16", "diversity better, likelihood failing (~2.1864)"),
        ("binarypercal-g512", "dead (~1.35; receipt is 1.3639)"),
    )
    scored = {p["name"]: p for p in doc.get("scored") or [] if p.get("organism") == "O003"}
    front_names = {p["name"] for p in doc["frontier"] if p.get("organism") == "O003"}
    lines.append("--- contract watch points (O003), verified against receipts ---")
    for name, why in watch:
        p = scored.get(name)
        if p is None:
            lines.append(f"  {name:28} UNSUPPORTED on disk  ({why})")
            continue
        ebpw = p["values"].get("complete_ebpw")
        cap = p["values"].get("capability")
        loc = "FRONTIER" if name in front_names else "scored, dominated on the 11 axes"
        extra = ""
        if p.get("ppl") is not None or p.get("median_4gram") is not None:
            extra = f"  ppl={p.get('ppl')} r4={p.get('median_4gram')}"
        lines.append(f"  {name:28} {loc}  ebpw={ebpw} cap={cap}{extra}  ({why})")
    lines.append(
        "  note: pqpercal4k512 and pqpercal2k16 disagree on likelihood vs diversity, "
        "which is why capability is a CONJUNCTION. That disagreement is not its own "
        "campaign axis, so under boolean capability both fail and a cheaper fail "
        "(binarypercal-g512, same support plus transfer_value) dominates them."
    )
    lines.append("")

    lake_front = [r for r in doc["frontier"] if r["kind"] == "lake_body"]
    if lake_front:
        ranked = sorted(
            lake_front,
            key=lambda r: r["values"].get("complete_ebpw", 99.0),
        )
        lines.append(f"--- lake bodies: {len(lake_front)} singleton fronts "
                     f"(different organisms; incomparable) ---")
        lines.append(f"  lowest complete_ebpw: {ranked[0]['name']} "
                     f"{ranked[0]['values'].get('complete_ebpw')}")
        for r in ranked[:8]:
            lines.append(f"  {r['values'].get('complete_ebpw')}  {r.get('klass')}  {r['name']}")
        if len(ranked) > 8:
            lines.append(f"  ... {len(ranked) - 8} more")
        lines.append("")
    if doc.get("notes"):
        lines.append("notes:")
        for n in doc["notes"]:
            lines.append(f"  {n}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    assert set(CAMPAIGN_AXES) == LOWER_BETTER | HIGHER_BETTER, "axis direction incomplete"
    assert not g033_is_strict_subset(), "G033 extra axis was lost"
    assert "active_bytes_per_token" in G033_AXES
    assert "active_bytes_per_token" not in CAMPAIGN_AXES
    assert len(EPISTEMIC) == 6
    assert len(set(EPISTEMIC)) == 6

    same = {"organism": "T"}
    good = {**same, "capability": True, "complete_ebpw": 3.0, "tps": 100.0}
    worse = {**same, "capability": True, "complete_ebpw": 4.0, "tps": 80.0}
    assert dominates(good, worse) and not dominates(worse, good)
    assert frontier([good, worse]) == [good]

    # Tradeoff: better on one, worse on another — both retained.
    a = {**same, "complete_ebpw": 2.0, "tps": 50.0}
    b = {**same, "complete_ebpw": 3.0, "tps": 100.0}
    assert not dominates(a, b) and not dominates(b, a)
    fr = frontier([a, b])
    assert len(fr) == 2, fr

    # Missing TPS cannot dominate a measured (even poor) TPS.
    missing = {**same, "capability": True, "complete_ebpw": 2.0}
    poor_tps = {**same, "capability": True, "complete_ebpw": 3.0, "tps": 10.0}
    assert not dominates(missing, poor_tps), "missing TPS dominated a measured TPS"
    assert not dominates(poor_tps, missing) or True
    # poor_tps does not dominate missing either: worse on ebpw.
    assert not dominates(poor_tps, missing)
    fr = frontier([missing, poor_tps])
    assert missing in fr and poor_tps in fr, "a partial candidate was silently excluded"

    # Missing is not a zero: inventing TPS=0 would make `missing` worse on TPS
    # than any positive measurement. Under the rule it has no TPS, so it does
    # not lose on that axis.
    zeroish = {**same, "complete_ebpw": 2.0, "tps": 0.0}
    assert not dominates(poor_tps, zeroish)  # ebpw worse, TPS better: tradeoff
    assert dominates(poor_tps, {**same, "complete_ebpw": 3.0, "tps": 0.0})

    # A fuller candidate better on every axis the partial measured DOES exclude it.
    fuller = {**same, "capability": True, "complete_ebpw": 1.5, "tps": 200.0}
    assert dominates(fuller, missing)
    assert missing not in frontier([fuller, missing])

    # Capability failure never dominates a pass, even at far fewer bytes.
    failer = {**same, "capability": False, "complete_ebpw": 1.0}
    passer = {**same, "capability": True, "complete_ebpw": 9.0}
    assert not dominates(failer, passer)
    fr = frontier([failer, passer])
    assert failer in fr and passer in fr

    # Built is not capability-preserving: execution_complete must not become PASS.
    built = {"spec": "q2-g32-experts", "complete_ebpw": 3.06,
             "execution_complete": True, "capability_status": "CANDIDATE_PASS"}
    cap, _epi = _capability_from(built)
    assert cap is None, f"Doctor/built collapsed into capability: {cap}"

    # Different organisms never dominate.
    x = {"organism": "O003", "complete_ebpw": 0.8}
    y = {"organism": "lake-bitnet", "complete_ebpw": 3.9}
    assert not dominates(x, y) and not dominates(y, x)

    # Empty support is not a frontier point under harvest(); dominates refuses it.
    empty = {"organism": "T"}
    assert not dominates(good, empty) and not dominates(empty, good)

    # Epistemic labels remain six distinct strings.
    assert "physically_measured" != "verified"
    assert "implemented" != "wired"
    assert "accepted" != "architecturally_supported"

    # Explicit absence: every axis is MEASURED, REFUSED, or OWED — never dropped.
    disclosed = disclose_axes({**same, "complete_ebpw": 2.0, "axis_meta": {}})
    assert set(disclosed["axis_meta"]) == set(CAMPAIGN_AXES)
    assert disclosed["axis_meta"]["complete_ebpw"]["state"] == MEASURED
    assert disclosed["axis_meta"]["tps"]["state"] == OWED
    assert not has_value(disclosed, "tps")

    print("selfcheck OK")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    require_known_flags(("--selfcheck",), argv)
    if "--selfcheck" in argv:
        _selfcheck()
        return 0
    doc = _harvest()
    sys.stdout.write(format_report(doc))
    sys.stdout.write("\n")
    sys.stdout.write(json.dumps({
        "n_frontier": doc["n_frontier"],
        "n_scored": doc["n_scored"],
        "measurement_debt": doc["measurement_debt"],
        "g033_axes_are_strict_subset": doc["g033_axes_are_strict_subset"],
        "partial_rule": doc["partial_rule"],
        "frontier": doc["frontier"],
        "by_organism": {k: v for k, v in doc["by_organism"].items()
                        if not (len(v) == 1 and v[0]["kind"] == "lake_body")},
        "notes": doc["notes"],
    }, indent=1, default=str))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UnknownFlag as e:
        print(e, file=sys.stderr)
        raise SystemExit(2)
