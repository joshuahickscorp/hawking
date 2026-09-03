"""TEACHER_CORPUS_CONTRACT — real captures, bound, diverse, not fabricated.

Every representation experiment downstream depends on teacher captures being
real, bound to the right specimen/layer/surface, and diverse enough to mean
anything. This sidecar module is the contract: a capture manifest, per-row
bindings, computed diversity measures, dedup/near-dup, an anti-fabrication
guard, and bounded capture WorkUnits. It does not run a capture (no GPU).

Recovered, not rebuilt:
  * tools/odyssey/capture_moe_x.py — truncated-model activation capture
  * tools/odyssey/teacher_assess.py — T1/T3 gap assessment against a ledger
  * tools/odyssey/dedup.py + normalize.py — content hash + 5-gram Jaccard
  * research/lab/operators/q80_capture_index.py — per-row layer/token/expert CSR
  * hcli/flash_next.py — pinned specimen identity (repo + revision + seal)
  * hcli/workunit.py — WorkUnit shape (emit only; this lane does not admit)

The thing that did NOT exist: a validator that structurally refuses a corpus
which only meets its sample-count threshold because rows were copied,
resampled, or synthesised. That guard is the point of this lane.

    python3 tools/future/teacher_corpus.py --build
    python3 -m pytest tools/future/test_teacher_corpus.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence


RECEIPT = "TEACHER_CORPUS_CONTRACT.json"
SCHEMA = "hawking.future.teacher_corpus.v1"
RECORDED_BY = "tools/future/teacher_corpus.py"

# ---------------------------------------------------------------------------
# Recovered constants. Cited, not invented. Sidecar-local copies because
# tools/odyssey/* is Codex-owned and is not on this sparse worktree.
# Canonical: tools/odyssey/_paths.py JACCARD_WITHIN_CORPUS_NEAR_DUP / SHINGLE_SIZE
#            tools/odyssey/normalize.py, tools/odyssey/dedup.py
# ---------------------------------------------------------------------------

SHINGLE_SIZE = 5
JACCARD_NEAR_DUP = 0.80
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

SURFACES = (
    "hidden_pre_mlp",  # capture_moe_x MLP pre-hook (role, not a tensor name)
    "router",
    "routed_expert",
    "shared_expert",
    "attention",
    "combine",
)
ROUTED_SURFACES = frozenset({"router", "routed_expert", "attention"})

# Recovered from capture_moe_x.PROMPTS plus T1 math-core (teacher_assess).
CAPABILITY_DOMAINS = ("math", "code", "prose", "tool", "shell")

# Natural exact-collision rate for captured activations is ~0 (Q80 index is
# 1,212,384 distinct (token, layer) rows). Text-level exact dups in a real
# capture stream are rare. Anything above this is copies, not nature.
NATURAL_DUP_RATE = 0.05
UNIQUE_RATIO_MIN = 0.95
UNIQUE_RATIO_FLOOR = 0.85  # below this is implausible even before min_rows
MIN_PROMPTS_FOR_FIT = 4  # capture_moe_x used 5 distinct prompts
MIN_DOMAINS_FOR_FIT = 3
MIN_ROUTES_ROUTED = 2
MIN_POSITIONS = 2
POSITION_MAX_SHARE = 0.90
# Uniform-cycling fingerprint: real Q80 routing is skewed (p10=3, p50=54,
# p90=221 on the 4-layer speed receipt). Perfect equality is synthetic.
ROUTE_UNIFORM_CV_MAX = 0.05
ROUTE_UNIFORM_MIN_ROUTES = 4
ROUTE_UNIFORM_MIN_ROWS = 16
NEAR_DUP_PADDING_MIN_ROWS = 8

PROVENANCE_PADDING_KINDS = frozenset({"duplicated", "resampled", "synthesised"})
AUTHORITIES = ("PROTECTED_ABSOLUTE", "DIAGNOSTIC_RELATIVE", "STATIC_ONLY")

# Recovered specimen identities (disk/git, not guessed).
FLASH_SPECIMEN = {
    "model": "Qwen/Qwen3.8-Flash-Next",
    "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
    "seal_sha256": "1f536f70c963a4d9b800e17c54c0dec54ce2d31a0a36041146b4013a04221fb7",
    "seal_kind": "model_lake_manifest_sha256",
    "source": "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json#source_identity",
}
QWEN80_SPECIMEN = {
    "model": "Qwen/Qwen3-Coder-Next",
    "pinned_revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
    "seal_sha256": "17a1e9b60a53cc491601a549880c2d215ff16395ee36abaa05fb95eb7fe2aabe",
    "seal_kind": "capture_source_json_sha256",
    "source": "receipts/QWEN80_CAPTURE_INDEX.json#source_json.sha256",
}

# capture_moe_x default --layers 4; QWEN80_CAPTURE_SPEED first window is 4 layers.
BOUNDED_LAYER_RANGE = (0, 4)
BOUNDED_TARGET_ROWS = 256  # unit bound; not a hardware measurement

FIXTURE_SPECIMEN = {
    "model": "fixture/teacher-corpus-selftest",
    "pinned_revision": "0" * 40,
    "seal_sha256": hashlib.sha256(b"fixture/teacher-corpus-selftest").hexdigest(),
    "seal_kind": "fixture_identity",
    "source": "tools/future/teacher_corpus.py::FIXTURE_SPECIMEN",
}

# Recovered prompts (capture_moe_x.PROMPTS) plus a math-core prompt so the
# five capability domains are representable in the selftest fixture.
CAPTURE_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "prose",
        "Explain, in ordinary prose and at length, how a compiler turns a "
        "for-loop into basic blocks and then into machine code.",
    ),
    (
        "code",
        "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    "
        "p = a[len(a)//2]\n",
    ),
    (
        "tool",
        '{"tool": "search", "arguments": {"query": "unified memory bandwidth", "limit": 5}}',
    ),
    (
        "shell",
        "$ grep -rn 'threadgroup' crates/hawking-core/shaders/*.metal | head -20",
    ),
    (
        "math",
        "Prove that the square root of 2 is irrational by infinite descent.",
    ),
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CorpusRefused(ValueError):
    """Loud refusal: the corpus is fabricated, padded, or unbound.

    A guard nobody has watched fail is not a guard. Callers must not catch
    this to convert a FAIL into a PASS.
    """

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result
        self.codes = list(result.get("refusals") or [])


class CorpusInadequate(ValueError):
    """Honest corpus that is too thin for a representation fit."""

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result
        self.codes = list(result.get("inadequacy") or [])


# ---------------------------------------------------------------------------
# Text identity (aligned with tools/odyssey/normalize.py + dedup.py)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s


def normalize_for_shingles(text: str) -> str:
    s = normalize_text(text)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def char_shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    s = normalize_for_shingles(text)
    if not s:
        return frozenset()
    if len(s) < size:
        return frozenset([s])
    return frozenset(s[i : i + size] for i in range(len(s) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _sha256_canonical(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_sha256_of(row: Mapping[str, Any]) -> str:
    """Identity of the captured content, not of the envelope.

    row_id and provenance are excluded so a copied row with a fresh id still
    collides. That collision is the fabrication signal.
    """
    specimen = row.get("specimen") or {}
    routes = sorted(int(x) for x in (row.get("route_ids") or []))
    identity = {
        "model": specimen.get("model"),
        "pinned_revision": specimen.get("pinned_revision"),
        "seal_sha256": specimen.get("seal_sha256"),
        "layer": int(row.get("layer", -1)),
        "surface": row.get("surface"),
        "prompt_text": normalize_text(str(row.get("prompt_text") or "")),
        "token_position": int(row.get("token_position", -1)),
        "route_ids": routes,
        "payload": row.get("payload"),
    }
    return _sha256_canonical(identity)


def envelope_sha256_of(row: Mapping[str, Any]) -> str:
    body = {k: v for k, v in row.items() if k not in {"content_sha256", "envelope_sha256", "route_union_membership"}}
    return _sha256_canonical(body)


# ---------------------------------------------------------------------------
# Row / manifest construction
# ---------------------------------------------------------------------------


def make_row(
    *,
    row_id: str,
    specimen: Mapping[str, Any],
    layer: int,
    surface: str,
    prompt_id: str,
    prompt_text: str,
    token_position: int,
    route_ids: Sequence[int],
    capability_domain: str,
    payload: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "row_id": str(row_id),
        "specimen": {
            "model": specimen["model"],
            "pinned_revision": specimen["pinned_revision"],
            "seal_sha256": specimen["seal_sha256"],
            "seal_kind": specimen.get("seal_kind"),
            "source": specimen.get("source"),
        },
        "layer": int(layer),
        "surface": str(surface),
        "prompt_id": str(prompt_id),
        "prompt_text": str(prompt_text),
        "token_position": int(token_position),
        "route_ids": [int(x) for x in route_ids],
        "capability_domain": str(capability_domain),
        "payload": str(payload),
        "provenance": dict(provenance),
    }
    return annotate_row(row)


def annotate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["route_ids"] = sorted(int(x) for x in (out.get("route_ids") or []))
    out["content_sha256"] = content_sha256_of(out)
    out["envelope_sha256"] = envelope_sha256_of(out)
    return out


def annotate_corpus(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    annotated = [annotate_row(r) for r in rows]
    unions = route_unions(annotated)
    for row in annotated:
        key = _union_key(row)
        union = unions.get(key, [])
        membership = sorted(set(row["route_ids"]) & set(union))
        row["route_union_membership"] = membership
    return annotated


def _union_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    spec = row.get("specimen") or {}
    return (
        str(spec.get("model") or ""),
        str(spec.get("pinned_revision") or ""),
        int(row.get("layer", -1)),
        str(row.get("surface") or ""),
    )


def route_unions(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, str], list[int]]:
    """Route union = experts/heads that actually received a row at (specimen, layer, surface).

    Recovered shape: q80_capture_index key_(layer, expert); 24,576 LE pairs
    minus 221 never-routed on the Qwen80 capture index.
    """
    acc: dict[tuple[str, str, int, str], set[int]] = {}
    for row in rows:
        acc.setdefault(_union_key(row), set()).update(int(x) for x in (row.get("route_ids") or []))
    return {k: sorted(v) for k, v in sorted(acc.items())}


def build_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = annotate_corpus(rows)
    unions = route_unions(annotated)
    captures: dict[tuple[Any, ...], dict[str, Any]] = {}
    specimens: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in annotated:
        spec = row["specimen"]
        specimens.setdefault(
            (spec["model"], spec["pinned_revision"], spec["seal_sha256"]),
            dict(spec),
        )
        key = (*_union_key(row),)
        cap = captures.setdefault(
            key,
            {
                "model": spec["model"],
                "pinned_revision": spec["pinned_revision"],
                "seal_sha256": spec["seal_sha256"],
                "layer": row["layer"],
                "surface": row["surface"],
                "n_rows": 0,
                "row_ids": [],
                "route_union": unions.get(_union_key(row), []),
                "provenance_chain": [],
            },
        )
        cap["n_rows"] += 1
        cap["row_ids"].append(row["row_id"])
        prov = row.get("provenance") or {}
        chain_item = {
            "kind": prov.get("kind"),
            "authority": prov.get("authority"),
            "source_path": prov.get("source_path"),
            "source_sha256": prov.get("source_sha256"),
            "capture_tool": prov.get("capture_tool"),
        }
        if chain_item not in cap["provenance_chain"]:
            cap["provenance_chain"].append(chain_item)
    capture_list = [captures[k] for k in sorted(captures)]
    return {
        "schema": "hawking.future.teacher_corpus.manifest.v1",
        "n_rows": len(annotated),
        "n_unique_content": len({r["content_sha256"] for r in annotated}),
        "specimens": [specimens[k] for k in sorted(specimens)],
        "captures": capture_list,
        "rows": annotated,
    }


# ---------------------------------------------------------------------------
# Diversity measures — computed, with the definition recorded next to the value
# ---------------------------------------------------------------------------


def _shannon_norm(counts: Sequence[int]) -> float:
    total = sum(counts)
    k = len([c for c in counts if c > 0])
    if total <= 0 or k <= 1:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h / math.log2(k)


def _cv(counts: Sequence[int]) -> float | None:
    vals = [float(c) for c in counts if c > 0]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    if mean == 0:
        return None
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / mean


DIVERSITY_MEASURES: dict[str, dict[str, Any]] = {
    "row_diversity": {
        "definition": (
            "Unique content_sha256 values divided by total rows. content_sha256 "
            "hashes the captured identity (specimen, layer, surface, prompt, "
            "position, routes, payload) and excludes row_id/provenance so copies "
            "collide."
        ),
        "formula": "|unique content_sha256| / n_rows",
        "threshold": UNIQUE_RATIO_MIN,
        "inadequate_below": UNIQUE_RATIO_MIN,
        "unit": "ratio",
        "recovered_justification": (
            "Q80 capture index is 1,212,384 layer-rows = 25,258 tokens × 48 layers; "
            "exact activation collisions at this grain are not a natural way to "
            "reach a sample-count threshold."
        ),
    },
    "prompt_diversity": {
        "definition": (
            "Number of distinct prompt_id values, plus normalized Shannon entropy "
            "of the prompt histogram (H / log2(k))."
        ),
        "formula": "n_unique_prompts; H_norm(prompt_id)",
        "threshold": MIN_PROMPTS_FOR_FIT,
        "inadequate_below": MIN_PROMPTS_FOR_FIT,
        "unit": "count + entropy-ratio",
        "recovered_justification": (
            "tools/odyssey/capture_moe_x.py ships 5 distinct prompts spanning "
            "prose/code/tool/factual/shell. teacher_assess.py records 0 natural-text "
            "windows on the GLM52 ledger — that is already a documented gap, not a "
            "license to pad one prompt."
        ),
    },
    "token_position_diversity": {
        "definition": (
            "Normalized Shannon entropy of token_position, unique position count, "
            "and the share of the most common position. Position degeneracy is "
            "all rows at one index (typical fabrication: capture position 0, copy)."
        ),
        "formula": "H_norm(token_position); max_share; n_unique_positions",
        "threshold": {"min_unique_positions": MIN_POSITIONS, "max_share": POSITION_MAX_SHARE},
        "inadequate_below": "unique_positions < 2 or max_share > 0.90 (n_rows >= 8)",
        "unit": "entropy-ratio + share",
        "recovered_justification": (
            "q80_capture_index stores step_index (position within probe) per row. "
            "A real capture spans the probe; a padded one does not."
        ),
    },
    "route_diversity": {
        "definition": (
            "Unique route ids in the derived route union, flattened-route "
            "histogram entropy, and coefficient of variation of per-route counts. "
            "Real MoE routing on the Qwen80 4-layer speed receipt is skewed "
            "(p10=3, p50=54, p90=221, 105 zero experts). Perfectly equal counts "
            "across many routes are a cycling synthesizer, not a router."
        ),
        "formula": "n_unique_routes; H_norm(route); CV(route counts)",
        "threshold": {
            "min_routes_on_routed_surface": MIN_ROUTES_ROUTED,
            "uniform_cv_max": ROUTE_UNIFORM_CV_MAX,
        },
        "inadequate_below": "routed surface with < 2 unique routes",
        "unit": "count + entropy-ratio + CV",
        "recovered_justification": (
            "receipts/QWEN80_CAPTURE_INDEX.json n_keys=24355 of 24576 LE pairs "
            "(221 never-routed). Occupancy is heavy-tailed, not uniform."
        ),
    },
    "capability_domain_diversity": {
        "definition": (
            "Distinct capability_domain values over the declared set "
            f"{list(CAPABILITY_DOMAINS)}, plus normalized Shannon entropy of the "
            "domain histogram."
        ),
        "formula": "n_unique_domains; H_norm(domain); n_unique_domains / |declared|",
        "threshold": MIN_DOMAINS_FOR_FIT,
        "inadequate_below": MIN_DOMAINS_FOR_FIT,
        "unit": "count + entropy-ratio",
        "recovered_justification": (
            "teacher_assess T1 needs math-core and support-language; capture_moe_x "
            "prompts already span prose, code, tool, shell. A one-domain pad is "
            "inadequate for a fit that claims generality."
        ),
    },
}


def compute_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = annotate_corpus(rows)
    n = len(annotated)
    hashes = [r["content_sha256"] for r in annotated]
    unique = len(set(hashes))
    prompts = [r.get("prompt_id") for r in annotated]
    positions = [int(r.get("token_position", -1)) for r in annotated]
    domains = [r.get("capability_domain") for r in annotated]
    route_flat: list[int] = []
    for r in annotated:
        route_flat.extend(int(x) for x in (r.get("route_ids") or []))
    prompt_counts = Counter(prompts)
    pos_counts = Counter(positions)
    domain_counts = Counter(domains)
    route_counts = Counter(route_flat)
    max_pos_share = (max(pos_counts.values()) / n) if n else 0.0
    unique_ratio = (unique / n) if n else 0.0
    return {
        "n_rows": n,
        "row_diversity": {
            "measure": unique_ratio,
            "n_unique": unique,
            "n_rows": n,
            "threshold": UNIQUE_RATIO_MIN,
            "inadequate": unique_ratio < UNIQUE_RATIO_MIN if n else True,
            **{k: DIVERSITY_MEASURES["row_diversity"][k] for k in ("definition", "formula", "unit")},
        },
        "prompt_diversity": {
            "measure": len(prompt_counts),
            "n_unique": len(prompt_counts),
            "entropy_norm": _shannon_norm(list(prompt_counts.values())),
            "histogram": dict(sorted((str(k), v) for k, v in prompt_counts.items())),
            "threshold": MIN_PROMPTS_FOR_FIT,
            "inadequate": len(prompt_counts) < MIN_PROMPTS_FOR_FIT,
            **{k: DIVERSITY_MEASURES["prompt_diversity"][k] for k in ("definition", "formula", "unit")},
        },
        "token_position_diversity": {
            "measure": _shannon_norm(list(pos_counts.values())),
            "n_unique": len(pos_counts),
            "max_share": max_pos_share,
            "histogram": {str(k): v for k, v in sorted(pos_counts.items())},
            "threshold": DIVERSITY_MEASURES["token_position_diversity"]["threshold"],
            "inadequate": (n >= 8 and (len(pos_counts) < MIN_POSITIONS or max_pos_share > POSITION_MAX_SHARE)),
            **{k: DIVERSITY_MEASURES["token_position_diversity"][k] for k in ("definition", "formula", "unit")},
        },
        "route_diversity": {
            "measure": len(route_counts),
            "n_unique": len(route_counts),
            "entropy_norm": _shannon_norm(list(route_counts.values())) if route_counts else 0.0,
            "cv": _cv(list(route_counts.values())) if route_counts else None,
            "histogram": {str(k): v for k, v in sorted(route_counts.items())},
            "threshold": DIVERSITY_MEASURES["route_diversity"]["threshold"],
            "inadequate": _route_inadequate(annotated, route_counts),
            **{k: DIVERSITY_MEASURES["route_diversity"][k] for k in ("definition", "formula", "unit")},
        },
        "capability_domain_diversity": {
            "measure": len(domain_counts),
            "n_unique": len(domain_counts),
            "declared": list(CAPABILITY_DOMAINS),
            "coverage": (len(domain_counts) / len(CAPABILITY_DOMAINS)) if CAPABILITY_DOMAINS else 0.0,
            "entropy_norm": _shannon_norm(list(domain_counts.values())),
            "histogram": dict(sorted((str(k), v) for k, v in domain_counts.items())),
            "threshold": MIN_DOMAINS_FOR_FIT,
            "inadequate": len(domain_counts) < MIN_DOMAINS_FOR_FIT,
            **{k: DIVERSITY_MEASURES["capability_domain_diversity"][k] for k in ("definition", "formula", "unit")},
        },
    }


def _route_inadequate(rows: Sequence[Mapping[str, Any]], route_counts: Counter) -> bool:
    routed = [r for r in rows if r.get("surface") in ROUTED_SURFACES]
    if not routed:
        return False
    ids: set[int] = set()
    for r in routed:
        ids.update(int(x) for x in (r.get("route_ids") or []))
    return len(ids) < MIN_ROUTES_ROUTED


# ---------------------------------------------------------------------------
# Dedup + near-dup
# ---------------------------------------------------------------------------


def exact_dedup(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Keep first occurrence of each content_sha256. Returns (kept, hash -> row_ids)."""
    annotated = annotate_corpus(rows)
    keep: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in annotated:
        h = row["content_sha256"]
        groups.setdefault(h, []).append(row["row_id"])
        if h not in seen:
            seen.add(h)
            keep.append(row)
    return keep, groups


def find_near_duplicates(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str = "payload",
    threshold: float = JACCARD_NEAR_DUP,
    shingle_size: int = SHINGLE_SIZE,
) -> dict[int, list[dict[str, Any]]]:
    """Pairwise near-dup hits on `field` (payload or prompt_text). O(n^2), bounded units."""
    annotated = annotate_corpus(rows)
    texts = [str(r.get(field) or "") for r in annotated]
    shingles = [char_shingles(t, shingle_size) for t in texts]
    hits: dict[int, list[dict[str, Any]]] = {}
    n = len(annotated)
    for i in range(n):
        for j in range(i + 1, n):
            score = jaccard(shingles[i], shingles[j])
            if score >= threshold:
                hits.setdefault(i, []).append(
                    {"other_index": j, "jaccard": score, "other_id": annotated[j]["row_id"]}
                )
                hits.setdefault(j, []).append(
                    {"other_index": i, "jaccard": score, "other_id": annotated[i]["row_id"]}
                )
    return hits


def _near_dup_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    threshold: float = JACCARD_NEAR_DUP,
) -> list[list[int]]:
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    texts = [str(r.get(field) or "") for r in rows]
    shingles = [char_shingles(t) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(shingles[i], shingles[j]) >= threshold:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


# ---------------------------------------------------------------------------
# Anti-fabrication guard
# ---------------------------------------------------------------------------


def _missing_binding(row: Mapping[str, Any]) -> list[str]:
    missing = []
    spec = row.get("specimen") or {}
    if not spec.get("model"):
        missing.append("specimen.model")
    if not spec.get("pinned_revision"):
        missing.append("specimen.pinned_revision")
    if not spec.get("seal_sha256"):
        missing.append("specimen.seal_sha256")
    if row.get("layer") is None:
        missing.append("layer")
    if not row.get("surface"):
        missing.append("surface")
    prov = row.get("provenance") or {}
    if not prov.get("kind"):
        missing.append("provenance.kind")
    if not prov.get("authority"):
        missing.append("provenance.authority")
    elif prov.get("authority") not in AUTHORITIES:
        missing.append("provenance.authority_invalid")
    return missing


def _route_cycling(route_counts: Counter, n_rows: int) -> bool:
    if n_rows < ROUTE_UNIFORM_MIN_ROWS:
        return False
    if len(route_counts) < ROUTE_UNIFORM_MIN_ROUTES:
        return False
    vals = list(route_counts.values())
    if max(vals) - min(vals) <= 1:
        return True
    cv = _cv(vals)
    return cv is not None and cv < ROUTE_UNIFORM_CV_MAX


def validate_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_rows: int,
    raise_on_refuse: bool = True,
    raise_on_inadequate: bool = False,
) -> dict[str, Any]:
    """Refuse a corpus that only meets `min_rows` by copying/resampling/synthesis.

    Structural detectors (do not trust labels):
      * exact content-hash collisions beyond NATURAL_DUP_RATE
      * unique-row count below min_rows while total >= min_rows
      * implausible unique/total ratio
      * suspiciously uniform route distribution (cycling)
      * position degeneracy
      * near-duplicate padding (Jaccard >= 0.80 clusters << min_rows)
      * provenance kinds duplicated/resampled/synthesised closing the gap
      * missing specimen/layer/surface/provenance bindings

    Diversity below a fit threshold is inadequacy, not fabrication, unless a
    detector above also fires.
    """
    if min_rows <= 0:
        raise ValueError("min_rows must be positive")
    annotated = annotate_corpus(rows)
    n = len(annotated)
    diversity = compute_diversity(annotated)
    unique = int(diversity["row_diversity"]["n_unique"])
    unique_ratio = float(diversity["row_diversity"]["measure"]) if n else 0.0
    collision_rate = (1.0 - unique_ratio) if n else 0.0

    refusals: list[str] = []
    details: dict[str, Any] = {}

    unbound = []
    for row in annotated:
        miss = _missing_binding(row)
        if miss:
            unbound.append({"row_id": row.get("row_id"), "missing": miss})
    if unbound:
        refusals.append("MISSING_SPECIMEN_OR_PROVENANCE_BINDING")
        details["unbound_rows"] = unbound[:12]

    if n >= min_rows and unique < min_rows:
        refusals.append("THRESHOLD_MET_ONLY_BY_DUPLICATION")
        details["threshold_met_only_by_duplication"] = {
            "min_rows": min_rows,
            "n_rows": n,
            "n_unique_content": unique,
            "copied_rows": n - unique,
        }

    if n >= min_rows and collision_rate > NATURAL_DUP_RATE:
        refusals.append("EXCESS_EXACT_COLLISIONS")
        details["collision_rate"] = collision_rate
        details["natural_dup_rate"] = NATURAL_DUP_RATE

    if n >= min_rows and unique_ratio < UNIQUE_RATIO_FLOOR:
        refusals.append("UNIQUE_RATIO_IMPLausible")
        details["unique_ratio"] = unique_ratio
        details["unique_ratio_floor"] = UNIQUE_RATIO_FLOOR

    pos_div = diversity["token_position_diversity"]
    if n >= 8 and (int(pos_div["n_unique"]) < MIN_POSITIONS or float(pos_div["max_share"]) > POSITION_MAX_SHARE):
        refusals.append("POSITION_DEGENERACY")
        details["position"] = {
            "n_unique": pos_div["n_unique"],
            "max_share": pos_div["max_share"],
        }

    route_counts = Counter(
        int(x) for r in annotated for x in (r.get("route_ids") or [])
    )
    routed_rows = [r for r in annotated if r.get("surface") in ROUTED_SURFACES]
    if routed_rows and _route_cycling(route_counts, n):
        refusals.append("ROUTE_DISTRIBUTION_SUSPICIOUSLY_UNIFORM")
        details["route_cv"] = _cv(list(route_counts.values()))
        details["route_counts"] = {str(k): v for k, v in sorted(route_counts.items())}

    # Near-dup padding on payload: many rows, few clusters.
    if n >= NEAR_DUP_PADDING_MIN_ROWS:
        payload_clusters = _near_dup_components(annotated, field="payload")
        n_clusters = len(payload_clusters)
        if n >= min_rows and n_clusters < min_rows:
            # Exact copies share a payload so they cluster too; still a pad.
            # An honest corpus with n < min_rows never enters this branch.
            refusals.append("NEAR_DUPLICATE_OR_RESAMPLE_PADDING")
            details["payload_clusters"] = n_clusters
            details["payload_cluster_sizes"] = sorted(len(c) for c in payload_clusters)

    padding_kinds = [
        r for r in annotated
        if (r.get("provenance") or {}).get("kind") in PROVENANCE_PADDING_KINDS
    ]
    honest = n - len(padding_kinds)
    # Unique honest content: if labelled padding is what closes min_rows.
    honest_unique = len({
        r["content_sha256"] for r in annotated
        if (r.get("provenance") or {}).get("kind") not in PROVENANCE_PADDING_KINDS
    })
    if n >= min_rows and honest_unique < min_rows and padding_kinds:
        refusals.append("SYNTHESISED_OR_DUPLICATED_TO_THRESHOLD")
        details["honest_unique"] = honest_unique
        details["padding_rows"] = len(padding_kinds)
        details["honest_rows"] = honest

    inadequacy: list[str] = []
    if unique < min_rows:
        inadequacy.append("UNIQUE_ROWS_BELOW_MIN")
    if diversity["prompt_diversity"]["inadequate"]:
        inadequacy.append("PROMPT_DIVERSITY_BELOW_FIT")
    if diversity["token_position_diversity"]["inadequate"]:
        inadequacy.append("POSITION_DIVERSITY_BELOW_FIT")
    if diversity["route_diversity"]["inadequate"]:
        inadequacy.append("ROUTE_DIVERSITY_BELOW_FIT")
    if diversity["capability_domain_diversity"]["inadequate"]:
        inadequacy.append("DOMAIN_DIVERSITY_BELOW_FIT")
    if diversity["row_diversity"]["inadequate"]:
        inadequacy.append("ROW_DIVERSITY_BELOW_FIT")

    # Dedup the refusal list while preserving order.
    seen_r: set[str] = set()
    refusals_u = []
    for c in refusals:
        if c not in seen_r:
            seen_r.add(c)
            refusals_u.append(c)
    refusals = refusals_u

    accepted = (not refusals) and (unique >= min_rows) and (not inadequacy)
    result = {
        "accepted": accepted,
        "min_rows": min_rows,
        "n_rows": n,
        "n_unique_content": unique,
        "unique_ratio": unique_ratio,
        "collision_rate": collision_rate,
        "refusals": refusals,
        "inadequacy": inadequacy,
        "details": details,
        "diversity": diversity,
        "claim_boundary": (
            "STATIC_ONLY validator. No GPU capture was performed. Refusal is a "
            "structural property of the rows, not a hardware measurement."
        ),
    }

    if refusals and raise_on_refuse:
        raise CorpusRefused(
            "REFUSED: teacher corpus failed anti-fabrication guard "
            f"(codes={refusals}; n_rows={n} unique={unique} min_rows={min_rows})",
            result,
        )
    if (not accepted) and (not refusals) and raise_on_inadequate:
        raise CorpusInadequate(
            f"INADEQUATE: teacher corpus below fit thresholds (codes={inadequacy})",
            result,
        )
    return result


# ---------------------------------------------------------------------------
# Fixtures used by selftest and by the required negative-control test
# ---------------------------------------------------------------------------


def _fixture_provenance(kind: str = "captured") -> dict[str, Any]:
    return {
        "kind": kind,
        "authority": "STATIC_ONLY",
        "source_path": "tools/future/teacher_corpus.py::fixture",
        "source_sha256": hashlib.sha256(b"teacher-corpus-fixture-v1").hexdigest(),
        "capture_tool": "tools/future/teacher_corpus.py",
        "note": "deterministic fixture; not a GPU capture and not a promotion",
    }


def make_diverse_corpus(
    n: int = 32,
    *,
    specimen: Mapping[str, Any] | None = None,
    surface: str = "routed_expert",
) -> list[dict[str, Any]]:
    """Genuinely diverse corpus of size n. Zipf-ish routes, many positions/prompts."""
    spec = dict(specimen or FIXTURE_SPECIMEN)
    # Heavy-tailed route occupancy (recovered Q80 shape, scaled down).
    # Length must be >= n; extra entries are ignored.
    zipf = []
    weights = [16, 10, 7, 5, 4, 3, 3, 2, 2, 1]
    for expert, w in enumerate(weights):
        zipf.extend([expert] * w)
    while len(zipf) < n:
        zipf.append(0)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        domain, text = CAPTURE_PROMPTS[i % len(CAPTURE_PROMPTS)]
        layer = i % (BOUNDED_LAYER_RANGE[1] - BOUNDED_LAYER_RANGE[0])
        pos = i % 8
        e = zipf[i]
        routes = [e, (e + 3) % 10]
        payload = hashlib.sha256(
            f"{text}|L{layer}|pos{pos}|i{i}|e{e}".encode("utf-8")
        ).hexdigest()
        rows.append(
            make_row(
                row_id=f"div-{i:04d}",
                specimen=spec,
                layer=layer,
                surface=surface,
                prompt_id=f"p-{domain}-{i % len(CAPTURE_PROMPTS)}",
                prompt_text=text,
                token_position=pos,
                route_ids=routes,
                capability_domain=domain,
                payload=payload,
                provenance=_fixture_provenance("captured"),
            )
        )
    return rows


def make_duplicated_corpus(
    n: int = 32,
    *,
    unique: int = 4,
    specimen: Mapping[str, Any] | None = None,
    surface: str = "routed_expert",
) -> list[dict[str, Any]]:
    """Reach n rows by copying `unique` real rows. The guard must refuse this."""
    base = make_diverse_corpus(unique, specimen=specimen, surface=surface)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        src = dict(base[i % unique])
        src["row_id"] = f"dup-{i:04d}"
        src["provenance"] = _fixture_provenance("duplicated")
        rows.append(annotate_row(src))
    return annotate_corpus(rows)


def make_uniform_route_corpus(n: int = 32, n_routes: int = 8) -> list[dict[str, Any]]:
    """Cycling routes, otherwise diverse. Must trip the uniform-route detector."""
    spec = FIXTURE_SPECIMEN
    rows: list[dict[str, Any]] = []
    for i in range(n):
        domain, text = CAPTURE_PROMPTS[i % len(CAPTURE_PROMPTS)]
        e = i % n_routes
        payload = hashlib.sha256(f"uniform|{i}|{e}".encode()).hexdigest()
        rows.append(
            make_row(
                row_id=f"uni-{i:04d}",
                specimen=spec,
                layer=i % 4,
                surface="routed_expert",
                prompt_id=f"p-{domain}-{i % len(CAPTURE_PROMPTS)}",
                prompt_text=text,
                token_position=i % 8,
                route_ids=[e],
                capability_domain=domain,
                payload=payload,
                provenance=_fixture_provenance("synthesised"),
            )
        )
    return rows


def make_position_degenerate_corpus(n: int = 32) -> list[dict[str, Any]]:
    spec = FIXTURE_SPECIMEN
    rows: list[dict[str, Any]] = []
    for i in range(n):
        domain, text = CAPTURE_PROMPTS[i % len(CAPTURE_PROMPTS)]
        payload = hashlib.sha256(f"pos0|{i}".encode()).hexdigest()
        rows.append(
            make_row(
                row_id=f"pos-{i:04d}",
                specimen=spec,
                layer=i % 4,
                surface="routed_expert",
                prompt_id=f"p-{domain}-{i % len(CAPTURE_PROMPTS)}",
                prompt_text=text,
                token_position=0,
                route_ids=[i % 5, (i + 2) % 7],
                capability_domain=domain,
                payload=payload,
                provenance=_fixture_provenance("synthesised"),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Capture WorkUnits — emit only, never execute
# ---------------------------------------------------------------------------


def emit_capture_workunits() -> list[dict[str, Any]]:
    """Bounded capture units HCLI could later execute. Not run here."""
    units: list[dict[str, Any]] = []
    diversity_target = {
        "row_diversity_min": UNIQUE_RATIO_MIN,
        "min_prompts": MIN_PROMPTS_FOR_FIT,
        "min_domains": MIN_DOMAINS_FOR_FIT,
        "min_routes_routed": MIN_ROUTES_ROUTED,
        "position_max_share": POSITION_MAX_SHARE,
        "validator": "tools/future/teacher_corpus.py:validate_corpus",
    }
    lo, hi = BOUNDED_LAYER_RANGE
    for specimen in (FLASH_SPECIMEN, QWEN80_SPECIMEN):
        for surface in SURFACES:
            payload = {
                "specimen": dict(specimen),
                "layer_range": [lo, hi],
                "surface": surface,
                "target_row_count": BOUNDED_TARGET_ROWS,
                "diversity_target": dict(diversity_target),
            }
            uid = "teacher-capture-" + _sha256_canonical(payload)[:16]
            units.append(
                {
                    "id": uid,
                    "role": "teacher_capture",
                    "description": (
                        f"Capture {BOUNDED_TARGET_ROWS} teacher rows for "
                        f"{specimen['model']} @ {specimen['pinned_revision'][:12]} "
                        f"layers [{lo},{hi}) surface={surface}"
                    ),
                    "dependencies": [],
                    "status": "pending",
                    "resource_class": "GPU_EXCLUSIVE",
                    "effect_class": "read_only_capture",
                    "preferred_backend": None,
                    "verifier": "tools/future/teacher_corpus.py:validate_corpus",
                    "workspace": None,
                    "executed": False,
                    "execution_forbidden_reason": (
                        "Sidecar has no GPU lease. WorkUnits are emitted, not run. "
                        "A later HCLI protected window may execute them; this module must not."
                    ),
                    "payload": payload,
                }
            )
    units.sort(key=lambda u: u["id"])
    return units


# ---------------------------------------------------------------------------
# Recovery (git is authority; sparse checkout is not absence)
# ---------------------------------------------------------------------------


def _head_has(rel: str) -> bool:
    out = git("ls-tree", "-r", "--name-only", "HEAD", "--", rel)
    return bool(out.strip())


def recover_implementation() -> dict[str, Any]:
    probes = [
        "tools/odyssey/capture_moe_x.py",
        "tools/odyssey/teacher_assess.py",
        "tools/odyssey/model_specimen_seal.py",
        "tools/odyssey/dedup.py",
        "tools/odyssey/normalize.py",
        "hcli/agentos/modellake_receipts.py",
        "hcli/flash_next.py",
        "hcli/workunit.py",
        "research/lab/operators/q80_capture_index.py",
        "research/lab/operators/q80_capture_coverage.py",
        "receipts/headless/QWEN80_CAPTURE_INDEX.json",
        "receipts/headless/QWEN80_CAPTURE_SPEED.json",
        "receipts/QWEN80_CAPTURE_INDEX.json",
        "receipts/QWEN80_CAPTURE_SPEED.json",
        "receipts/QWEN80_SOURCE_IDENTITY.json",
        "receipts/headless/FLASH_ATTENTION_ROUTE_UNION_PARITY.json",
        "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json",
        "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json",
        "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json",
        "workspace/campaign/governance/odyssey/resources/teacher_traces/ODYSSEY_TEACHER_TRACE_MANIFEST.json",
        "tools/future/teacher_corpus.py",
    ]
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel in probes:
        exists = _head_has(rel)
        entry: dict[str, Any] = {
            "path": rel,
            "in_HEAD": exists,
            "on_disk": (REPO / rel).is_file(),
        }
        if exists:
            found.append(entry)
        else:
            missing.append(rel)
            found.append(entry)

    summaries: list[dict[str, Any]] = []

    if _head_has("tools/odyssey/capture_moe_x.py"):
        summaries.append({
            "path": "tools/odyssey/capture_moe_x.py",
            "what": (
                "Truncated-model activation capture for layers 0..K-1 (exact, not a "
                "proxy). Writes X_layer{i}.npy + CAPTURE.json. 5 prompts. Refuses "
                "non-finite activations. No specimen seal, no route ids, no "
                "diversity measures, no anti-fabrication guard."
            ),
            "adequate_for_this_lane": False,
        })
    if _head_has("tools/odyssey/teacher_assess.py"):
        summaries.append({
            "path": "tools/odyssey/teacher_assess.py",
            "what": (
                "Assesses GLM52 teacher-capsule ledger against T1/T3 needs. "
                "Reports PARTIAL: 118 captured / 4 failed, 0 trajectory traces, "
                "0 natural-text windows. Not a row-level corpus validator."
            ),
            "adequate_for_this_lane": False,
        })
    if _head_has("tools/odyssey/dedup.py"):
        summaries.append({
            "path": "tools/odyssey/dedup.py",
            "what": (
                "content_sha256 + 5-gram Jaccard near-dup at 0.80. Used for "
                "train/eval overlap. This module aligns those constants rather "
                "than inventing a second threshold. Does not bind specimen/"
                "layer/surface and does not refuse sample-count padding."
            ),
            "adequate_for_this_lane": False,
            "aligned_constants": {
                "SHINGLE_SIZE": SHINGLE_SIZE,
                "JACCARD_NEAR_DUP": JACCARD_NEAR_DUP,
            },
        })
    if _head_has("hcli/agentos/modellake_receipts.py"):
        summaries.append({
            "path": "hcli/agentos/modellake_receipts.py",
            "what": (
                "Canonical vs legacy ModelLake census/supervision receipt names. "
                "Not specimen identity and not a seal."
            ),
            "adequate_for_this_lane": False,
        })
    if _head_has("hcli/flash_next.py"):
        summaries.append({
            "path": "hcli/flash_next.py",
            "what": (
                "Pinned Flash-Next identity: Qwen/Qwen3.8-Flash-Next @ "
                "34567a4712bc9766c4449e2e98e4468bfa24d915. Consumed as specimen "
                "binding, not re-derived."
            ),
            "adequate_for_this_lane": False,
        })
    if _head_has("research/lab/operators/q80_capture_index.py"):
        summaries.append({
            "path": "research/lab/operators/q80_capture_index.py",
            "what": (
                "Per-row arrays: layer, token_index, probe_index, step_index, "
                "input_token_id, expert_ids CSR, key_(layer,expert). This is the "
                "recovered row grain and the recovered route-union shape."
            ),
            "adequate_for_this_lane": False,
        })
    if _head_has("receipts/QWEN80_CAPTURE_INDEX.json"):
        summaries.append({
            "path": "receipts/QWEN80_CAPTURE_INDEX.json",
            "what": (
                "schema hawking.q80.capture_index.measurement.v1; n_rows=1212384; "
                "n_tokens=25258; n_keys=24355 of 24576 LE pairs (221 never-routed). "
                "NOT under receipts/headless/ (contract path was wrong)."
            ),
            "adequate_for_this_lane": False,
        })
    if _head_has("hcli/workunit.py"):
        summaries.append({
            "path": "hcli/workunit.py",
            "what": (
                "WorkUnit dataclass + content_identity. Capture units emitted here "
                "use the same field names; this lane does not admit or run them."
            ),
            "adequate_for_this_lane": False,
        })

    trees = {}
    for prefix in ("tools/accelerator", "tools/headless", "hcli/agentos"):
        names = [l for l in git("ls-tree", "-r", "--name-only", "HEAD", prefix).splitlines() if l]
        trees[prefix] = {"n_files": len(names), "sample": names[:6]}

    return {
        "head": git("rev-parse", "HEAD"),
        "probe_results": found,
        "missing_from_HEAD": missing,
        "summaries": summaries,
        "trees": trees,
        "note": (
            "A path missing from this sparse worktree is not evidence it is "
            "absent from the civilization. Probes used git ls-tree HEAD."
        ),
    }


# ---------------------------------------------------------------------------
# Selftest + receipt
# ---------------------------------------------------------------------------


def selftest() -> dict[str, Any]:
    diverse = make_diverse_corpus(32)
    duped = make_duplicated_corpus(32, unique=4)

    diverse_result = validate_corpus(diverse, min_rows=32, raise_on_refuse=True)
    if not diverse_result["accepted"]:
        raise SystemExit(f"selftest: diverse corpus must be accepted, got {diverse_result}")

    dup_refused = False
    dup_codes: list[str] = []
    dup_message = ""
    try:
        validate_corpus(duped, min_rows=32, raise_on_refuse=True)
    except CorpusRefused as e:
        dup_refused = True
        dup_codes = list(e.codes)
        dup_message = str(e)
        dup_result = e.result
    else:
        raise SystemExit(
            "selftest: duplicated corpus was NOT refused — the anti-fabrication "
            "guard is dead"
        )

    if "THRESHOLD_MET_ONLY_BY_DUPLICATION" not in dup_codes:
        raise SystemExit(
            f"selftest: refusal fired but not on duplication; codes={dup_codes}"
        )

    uniform = make_uniform_route_corpus(32)
    try:
        validate_corpus(uniform, min_rows=32, raise_on_refuse=True)
        raise SystemExit("selftest: uniform-route corpus was NOT refused")
    except CorpusRefused as e:
        if "ROUTE_DISTRIBUTION_SUSPICIOUSLY_UNIFORM" not in e.codes:
            raise SystemExit(f"selftest: expected uniform-route refusal, got {e.codes}")
        uniform_codes = list(e.codes)

    pos = make_position_degenerate_corpus(32)
    try:
        validate_corpus(pos, min_rows=32, raise_on_refuse=True)
        raise SystemExit("selftest: position-degenerate corpus was NOT refused")
    except CorpusRefused as e:
        if "POSITION_DEGENERACY" not in e.codes:
            raise SystemExit(f"selftest: expected position degeneracy, got {e.codes}")
        pos_codes = list(e.codes)

    keep, groups = exact_dedup(duped)
    near = find_near_duplicates(diverse, field="payload")
    manifest = build_manifest(diverse)

    return {
        "diverse_accepted": True,
        "diverse_n": diverse_result["n_rows"],
        "diverse_unique": diverse_result["n_unique_content"],
        "duplicated_refused": dup_refused,
        "duplicated_codes": dup_codes,
        "duplicated_message": dup_message,
        "duplicated_n": dup_result["n_rows"],
        "duplicated_unique": dup_result["n_unique_content"],
        "uniform_route_refused_codes": uniform_codes,
        "position_degeneracy_refused_codes": pos_codes,
        "dedup_kept_from_duplicated": len(keep),
        "dedup_collision_groups": sum(1 for ids in groups.values() if len(ids) > 1),
        "near_dup_hits_on_diverse_payloads": len(near),
        "manifest_n_captures": len(manifest["captures"]),
        "manifest_specimens": len(manifest["specimens"]),
    }


def build() -> Any:
    recovered = recover_implementation()
    test = selftest()
    units = emit_capture_workunits()
    extra_neg = []
    if "receipts/headless/QWEN80_CAPTURE_INDEX.json" in recovered["missing_from_HEAD"]:
        extra_neg.append(
            "contract named receipts/headless/QWEN80_CAPTURE_INDEX.json; the "
            "artifact lives at receipts/QWEN80_CAPTURE_INDEX.json"
        )
    if "receipts/headless/QWEN80_CAPTURE_SPEED.json" in recovered["missing_from_HEAD"]:
        extra_neg.append(
            "contract named receipts/headless/QWEN80_CAPTURE_SPEED.json; the "
            "artifact lives at receipts/QWEN80_CAPTURE_SPEED.json"
        )
    if "tools/odyssey/model_specimen_seal.py" in recovered["missing_from_HEAD"]:
        extra_neg.append(
            "tools/odyssey/model_specimen_seal.py is not in HEAD; specimen "
            "identity recovered from hcli/flash_next.py PINNED_REVISION and "
            "FLASH_NOETIC_ROUTER_SELECTION source_identity.model_lake_manifest"
        )
    if "receipts/headless/FLASH_ATTENTION_ROUTE_UNION_PARITY.json" in recovered["missing_from_HEAD"]:
        extra_neg.append(
            "FLASH_ATTENTION_ROUTE_UNION_PARITY.json is not in HEAD; route-union "
            "shape recovered from research/lab/operators/q80_capture_index.py key_(layer,expert) "
            "and receipts/QWEN80_CAPTURE_INDEX.json (24,355 of 24,576 LE pairs)"
        )
    if "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json" in recovered["missing_from_HEAD"]:
        extra_neg.append(
            "FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json is not in HEAD; no L3/L4 "
            "Flash router sensitivity map was found under receipts/"
        )
    extra_neg.append(
        "No GPU capture was executed. WorkUnits are pending. Existing teacher "
        "ledgers (GLM52 capsules, Qwen80 hidden files) were not re-read as row "
        "bytes because those capture blobs are not in this sparse worktree."
    )

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Teacher-corpus contract: bind every capture to specimen/layer/surface/"
            "provenance, compute diversity, dedup, and structurally refuse a corpus "
            "that only meets its sample-count threshold by duplication, resampling, "
            "or synthesis."
        ),
        "surfaces": list(SURFACES),
        "capability_domains": list(CAPABILITY_DOMAINS),
        "diversity_measures": DIVERSITY_MEASURES,
        "anti_fabrication": {
            "detectors": [
                "THRESHOLD_MET_ONLY_BY_DUPLICATION",
                "EXCESS_EXACT_COLLISIONS",
                "UNIQUE_RATIO_IMPLausible",
                "POSITION_DEGENERACY",
                "ROUTE_DISTRIBUTION_SUSPICIOUSLY_UNIFORM",
                "NEAR_DUPLICATE_OR_RESAMPLE_PADDING",
                "SYNTHESISED_OR_DUPLICATED_TO_THRESHOLD",
                "MISSING_SPECIMEN_OR_PROVENANCE_BINDING",
            ],
            "natural_dup_rate": NATURAL_DUP_RATE,
            "unique_ratio_min": UNIQUE_RATIO_MIN,
            "unique_ratio_floor": UNIQUE_RATIO_FLOOR,
            "route_uniform_cv_max": ROUTE_UNIFORM_CV_MAX,
            "jaccard_near_dup": JACCARD_NEAR_DUP,
            "shingle_size": SHINGLE_SIZE,
            "loud_exception": "CorpusRefused",
            "rule": (
                "A corpus that only meets min_rows because rows were copied must "
                "FAIL. validate_corpus raises CorpusRefused. A return-flag that "
                "nobody checks is not a guard."
            ),
        },
        "dedup": {
            "exact": "content_sha256 of captured identity; first occurrence kept",
            "near": (
                "character shingles of size 5, Jaccard >= 0.80, aligned with "
                "tools/odyssey/dedup.py (canonical) / tools/odyssey/_paths.py"
            ),
            "shingle_size": SHINGLE_SIZE,
            "jaccard_threshold": JACCARD_NEAR_DUP,
        },
        "row_metadata": {
            "required": [
                "row_id",
                "specimen.model",
                "specimen.pinned_revision",
                "specimen.seal_sha256",
                "layer",
                "surface",
                "prompt_id",
                "prompt_text",
                "token_position",
                "route_ids",
                "route_union_membership",
                "capability_domain",
                "payload",
                "content_sha256",
                "envelope_sha256",
                "provenance.kind",
                "provenance.authority",
                "provenance.source_path",
            ],
            "route_union_definition": (
                "Sorted unique route_ids over all rows sharing "
                "(specimen, layer, surface). Membership is the intersection of "
                "the row's route_ids with that union. Recovered from Q80 "
                "key_(layer, expert), not from FLASH_ATTENTION_ROUTE_UNION_PARITY "
                "(that receipt is not in HEAD)."
            ),
        },
        "selftest": test,
        "capture_workunits": units,
        "capture_workunits_note": (
            "Emitted, not executed. resource_class=GPU_EXCLUSIVE. Sidecar must "
            "not seize a GPU lease. target_row_count is a bound, not a measurement."
        ),
        "recovered_implementation": recovered,
        "gaps_closed": [
            "Capture manifest binding every row to specimen identity (model + "
            "pinned revision + seal), layer, surface, and a provenance chain.",
            "Per-row route_ids, route_union_membership, content_sha256, envelope_sha256.",
            "Five computed diversity measures with recorded definitions and "
            "inadequacy thresholds: row, prompt, token-position, route, "
            "capability-domain.",
            "Exact dedup by content hash and 5-gram Jaccard near-dup, constants "
            "aligned with tools/odyssey/dedup.py rather than forked as a new policy.",
            "validate_corpus() anti-fabrication guard with a live negative "
            "control: a corpus that hits min_rows only by copying is REFUSED; a "
            "same-size diverse corpus is accepted.",
            "Bounded capture WorkUnits (specimen, layer range, surface, target "
            "row count, diversity target) emitted for HCLI, not run.",
        ],
        "negative_findings": extra_neg + [
            f"HEAD missing: {p}" for p in recovered["missing_from_HEAD"]
            if p != "tools/future/teacher_corpus.py"
        ],
        "era_vocabulary": {
            "eras": 5,
            "odysseys": 3,
            "fpga_is": "part of Accelerator / Physical Compiler / Fusion, not its own civilization",
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
    }
    out = write_receipt(RECEIPT, doc, RECORDED_BY)
    written = load_json(out)
    if written.get("schema") != SCHEMA or not written.get("seal_sha256"):
        raise SystemExit(f"receipt {out} failed round-trip")
    if written.get("bench", {}).get("state") != "UNKNOWN":
        raise SystemExit("receipt bench.state is not UNKNOWN")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
