"""CANDIDATE_STAGED_PLAN — static planner over the physical qualification queue.

Codex owns the live queue and the protected GPU window. This sidecar never
acquires a lease, never quiesces a process, and never runs a benchmark. It
reads the queue and emits a dependency graph, equivalence classes, redundant
pruning, interaction predictions, lineage/scar propagation, funnel-derived
promotion gates, and a staged factorial plan that is strictly smaller than 2^N.

    python3 tools/future/candidate_planner.py --build
    python3 tools/future/candidate_planner.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import re
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import REPO, git, load_json, sha256_file, write_receipt

RECEIPT = "CANDIDATE_STAGED_PLAN.json"
SCHEMA = "hawking.future.candidate_planner.v1"
QUEUE_REL = Path("receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json")

# Env keys that every Qwen27 child copies from qwen27-fast-profile. They do not
# distinguish a child and they do not create a conflict when two children share
# the same value.
SHARED_ENV_KEYS = frozenset({"HAWKING_QWEN38_FAST"})

# Distinctive env-key stems that name the same physical idea on two models.
# These are control-name aliases recovered from the queue rows, not candidate-id
# string similarity.
STEM_SYNONYM = {
    "FUSE_GQA_QKV": "QKV_FUSION",
    "QKV_GQA_FUSED": "QKV_FUSION",
    "FUSE_ATTENTION_GATE": "ATTENTION_GATE",
}

HOST_CEREMONY_KEYS = frozenset(
    {
        "HAWKING_METAL_PIPELINE_STATE_ELISION",
        "HAWKING_METAL_PIPELINE_CACHE_REUSE",
        "HAWKING_FLASH_PIPELINE_CACHE_REUSE",
        "HAWKING_METAL_PIPELINE_ID_RESOLUTION",
        "HAWKING_METAL_ENCODER_LABEL_ELISION",
        "HAWKING_METAL_COMMIT_TIMING_ELISION",
        "HAWKING_DSV4F_FULLSEQ_ORDERED_ENCODER",
    }
)

GEO_KEYS = frozenset(
    {
        "HAWKING_AFFINE2_GEO",
        "HAWKING_Q2F_GEO",
        "HAWKING_QWEN38_Q4_GEO",
        "HAWKING_FLASH_BF16_GEO",
        "HAWKING_FLASH_BF16_VEC4",
        "HAWKING_FLASH_MOE_GEO",
        "HAWKING_FLASH_MOE_VEC4",
    }
)

_STOP = frozenset(
    """
    the a an and or of to in for on with without per from into by is are be this
    that if only remain remains remaining unchanged no none not measure measured
    while when where which whose their its one two plus than more less over under
    """.split()
)

_DISPATCH_FROM_TO = re.compile(r"from\s+(\d+)\s+dispatches\s+to\s+(\d+)", re.I)
_DISPATCH_BY_NUM = re.compile(r"reduce by (\d+)", re.I)
_WORD = re.compile(r"[a-z0-9]+")


class IncompatibleMutationError(ValueError):
    """A measurement cell asked to co-schedule incompatible mutations."""


class QueueNotFoundError(FileNotFoundError):
    """The physical qualification queue is not visible from this worktree."""


# ---------------------------------------------------------------------------
# Queue recovery (read-only). The live file is Codex-owned and may exist only
# as uncommitted disk state on another worktree of this repo.
# ---------------------------------------------------------------------------

def _worktree_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    blob = git("worktree", "list", "--porcelain")
    for line in blob.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.split(" ", 1)[1]))
    # Unique, stable order. REPO stays first.
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def find_queue_path() -> Path:
    searched: list[str] = []
    for root in _worktree_roots():
        path = root / QUEUE_REL
        searched.append(str(path))
        if path.is_file():
            return path
    raise QueueNotFoundError(
        "physical qualification queue not found. searched: " + "; ".join(searched)
    )


def load_queue(path: Path | None = None) -> dict[str, Any]:
    target = path if path is not None else find_queue_path()
    doc = load_json(target)
    if not isinstance(doc, dict):
        raise ValueError(f"{target}: queue is not an object")
    if "candidates" not in doc or not isinstance(doc["candidates"], list):
        raise ValueError(f"{target}: queue has no candidates list")
    doc["_loaded_from"] = str(target)
    return doc


# ---------------------------------------------------------------------------
# Candidate accessors
# ---------------------------------------------------------------------------

def cid(row: Mapping[str, Any]) -> str:
    return str(row["candidate_id"])


def mutation_env(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("exact_mutation") or {}
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get("child_fusion_env") or raw.get("source_oracle_controls") or raw
    if not isinstance(nested, Mapping):
        return {}
    return {str(k): str(v) for k, v in nested.items()}


def control_env(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("control_configuration") or {}
    if not isinstance(raw, Mapping):
        return {}
    nested = (
        raw.get("child_fusion_env")
        or raw.get("source_oracle_controls")
        or raw
    )
    if not isinstance(nested, Mapping):
        return {}
    return {str(k): str(v) for k, v in nested.items() if not isinstance(v, (dict, list))}


def _norm_stem(key: str) -> str:
    stem = key
    for prefix in ("HAWKING_", "QWEN38_", "FLASH_", "DSV4F_", "METAL_"):
        stem = stem.replace(prefix, "")
    return STEM_SYNONYM.get(stem, stem)


def distinctive_stems(row: Mapping[str, Any]) -> frozenset[str]:
    env = mutation_env(row)
    return frozenset(_norm_stem(k) for k in env if k not in SHARED_ENV_KEYS)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _WORD.findall(str(text).lower()) if len(t) >= 4 and t not in _STOP
    )


def mechanism_tokens(row: Mapping[str, Any]) -> frozenset[str]:
    return _tokens(
        " ".join(
            [
                str(row.get("expected_eliminated_work") or ""),
                str(row.get("expected_gpu_ns_mechanism") or ""),
            ]
        )
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def resource_tags(row: Mapping[str, Any]) -> frozenset[str]:
    tags: set[str] = set()
    env = mutation_env(row)
    region = str(row.get("affected_physical_region") or "").lower()
    for key in env:
        if key in HOST_CEREMONY_KEYS:
            tags.add("host_metal_path")
            if "COMMIT" in key:
                tags.add("host_command_buffer_ceremony")
            else:
                tags.add("host_encoder_ceremony")
        if key in GEO_KEYS:
            if "MOE" in key:
                tags.add("moe_geometry")
            else:
                tags.add("gemm_geometry")
        if "FUSE_ATTENTION_GATE" in key or key.endswith("FUSE_ATTENTION_GATE"):
            tags.add("attention_organ")
        if "GQA" in key or "QKV" in key:
            tags.add("attention_organ")
        if "DN_INPROJ" in key or "BA_DELTA" in key or "DELTANET" in key:
            tags.add("deltanet_organ")
        if "ROUTER" in key or "TOPK" in key:
            tags.add("router_organ")
        if "HC_" in key or key.endswith("_HC_STAGED") or "HC_STAGED" in key:
            tags.add("hc_organ")
        if "SWIGLU" in key:
            tags.add("moe_organ")
    if "gqa" in region or "attention" in region:
        tags.add("attention_organ")
    if "deltanet" in region:
        tags.add("deltanet_organ")
    if "hyperconnection" in region or " hc " in f" {region} " or region.startswith("flash hc"):
        tags.add("hc_organ")
    if "router" in region or "top-k" in region or "topk" in region:
        tags.add("router_organ")
    if "moe" in region or "expert" in region or "swiglu" in region:
        tags.add("moe_organ")
    if "pipeline" in region:
        tags.add("host_pipeline")
        tags.add("host_metal_path")
    if "encoder" in region:
        tags.add("host_encoder")
        tags.add("host_metal_path")
    if "commit" in region or "fence" in region:
        tags.add("host_commit")
        tags.add("host_metal_path")
    if "gemv" in region or "projection family" in region:
        tags.add("gemm_geometry")
    if "complete qwen27 resident token" in region:
        tags.add("whole_token_profile")
    return frozenset(sorted(tags))


def env_subsumes(inner: Mapping[str, str], outer: Mapping[str, str]) -> bool:
    return all(outer.get(k) == v for k, v in inner.items()) and len(inner) <= len(outer)


# ---------------------------------------------------------------------------
# Conflicts — the refusal that must be able to fire
# ---------------------------------------------------------------------------

def conflict_reasons(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, str]]:
    """Why two candidates cannot occupy the same measurement cell.

    An empty list means the pair is cell-legal (they may still be a predicted
    additive pair that the staged plan refuses to schedule for other reasons).
    """
    if cid(a) == cid(b):
        return []
    reasons: list[dict[str, str]] = []
    if str(a.get("model") or "") != str(b.get("model") or ""):
        reasons.append(
            {
                "kind": "distinct_model_executables",
                "mechanism": (
                    "different model identities cannot share one physical run; "
                    "each executable is its own protected window"
                ),
            }
        )
    region_a = str(a.get("affected_physical_region") or "")
    region_b = str(b.get("affected_physical_region") or "")
    env_a = mutation_env(a)
    env_b = mutation_env(b)
    if region_a and region_a == region_b and env_a != env_b:
        reasons.append(
            {
                "kind": "same_region_incompatible_mutation",
                "mechanism": (
                    f"both touch {region_a!r} with different exact_mutation; "
                    "they cannot be enabled in the same run"
                ),
            }
        )
    for key in sorted(set(env_a) & set(env_b)):
        if env_a[key] != env_b[key]:
            reasons.append(
                {
                    "kind": "env_key_collision",
                    "mechanism": (
                        f"{key} is {env_a[key]!r} on {cid(a)} and {env_b[key]!r} "
                        f"on {cid(b)}; one process cannot hold both assignments"
                    ),
                }
            )
    return reasons


def assert_cell_compatible(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a measurement cell that co-schedules incompatible mutations.

    This is the guard. Callers (the planner and the tests) must see it fire on
    a known-bad pair; a function that only returns True on happy input is not
    a guard.
    """
    items = list(rows)
    for left, right in combinations(items, 2):
        reasons = conflict_reasons(left, right)
        if reasons:
            detail = "; ".join(r["kind"] + ": " + r["mechanism"] for r in reasons)
            raise IncompatibleMutationError(
                f"incompatible cell {{{cid(left)}, {cid(right)}}}: {detail}"
            )
    # Also refuse an unsatisfiable merged env even if pairwise reasons were
    # empty (defensive; pairwise env_key_collision already covers this).
    merged: dict[str, str] = {}
    for row in items:
        for key, value in mutation_env(row).items():
            prior = merged.get(key)
            if prior is not None and prior != value:
                raise IncompatibleMutationError(
                    f"incompatible cell env merge on {key}: {prior!r} vs {value!r}"
                )
            merged[key] = value


def cell_is_compatible(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        assert_cell_compatible(rows)
    except IncompatibleMutationError:
        return False
    return True


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {cid(r): r for r in rows}


def declared_edges(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    known = {cid(r) for r in rows}
    edges: list[dict[str, str]] = []
    for row in rows:
        for dep in row.get("dependencies") or []:
            dep_id = str(dep)
            if dep_id not in known:
                continue
            edges.append(
                {
                    "from": dep_id,
                    "to": cid(row),
                    "kind": "declared_dependency",
                    "mechanism": (
                        f"{cid(row)} names {dep_id} in dependencies; "
                        "the parent is a precondition of the child"
                    ),
                }
            )
    edges.sort(key=lambda e: (e["from"], e["to"], e["kind"]))
    return edges


def conflict_edges(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    ordered = sorted(rows, key=cid)
    for a, b in combinations(ordered, 2):
        reasons = conflict_reasons(a, b)
        if not reasons:
            continue
        edges.append(
            {
                "a": cid(a),
                "b": cid(b),
                "kind": "inferred_conflict",
                "reasons": reasons,
            }
        )
    edges.sort(key=lambda e: (e["a"], e["b"]))
    return edges


def build_graph(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nodes = []
    for row in sorted(rows, key=cid):
        nodes.append(
            {
                "candidate_id": cid(row),
                "model": row.get("model"),
                "status": row.get("status"),
                "affected_physical_region": row.get("affected_physical_region"),
                "dependencies": list(row.get("dependencies") or []),
                "resource_tags": sorted(resource_tags(row)),
                "mutation_env": mutation_env(row),
            }
        )
    declared = declared_edges(rows)
    conflicts = conflict_edges(rows)
    return {
        "node_count": len(nodes),
        "declared_edge_count": len(declared),
        "conflict_edge_count": len(conflicts),
        "nodes": nodes,
        "declared_edges": declared,
        "conflict_edges": conflicts,
    }


def descendants_of(root: str, declared: Sequence[Mapping[str, str]]) -> list[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for edge in declared:
        children[edge["from"]].append(edge["to"])
    out: list[str] = []
    seen = {root}
    q: deque[str] = deque(sorted(children.get(root, [])))
    while q:
        node = q.popleft()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        for child in sorted(children.get(node, [])):
            if child not in seen:
                q.append(child)
    return out


# ---------------------------------------------------------------------------
# Equivalence — evidence required, name similarity is not evidence
# ---------------------------------------------------------------------------

def _geometry_refinement(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, str] | None:
    if str(parent.get("affected_physical_region")) != str(child.get("affected_physical_region")):
        return None
    deps = {str(d) for d in (child.get("dependencies") or [])}
    if cid(parent) not in deps:
        return None
    p_env = mutation_env(parent)
    c_env = mutation_env(child)
    refined_keys = []
    for key, p_val in p_env.items():
        c_val = c_env.get(key)
        if c_val is None or c_val == p_val:
            continue
        if c_val == p_val + "_vec" or c_val.startswith(p_val + "_"):
            refined_keys.append(key)
    ctrl = control_env(child)
    geo_control_match = any(ctrl.get(k) == p_env.get(k) for k in GEO_KEYS if k in p_env)
    if not refined_keys and not geo_control_match:
        return None
    return {
        "kind": "geometry_refinement",
        "mechanism": (
            f"{cid(child)} is a geometry refinement of {cid(parent)} on the same "
            f"affected_physical_region; child.dependencies names the parent; "
            f"refined_keys={refined_keys or ['control_matches_parent']}"
        ),
    }


def _mechanism_twin(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, str] | None:
    if cid(a) == cid(b):
        return None
    stems_a = distinctive_stems(a)
    stems_b = distinctive_stems(b)
    shared = stems_a & stems_b
    # Drop the generic FAST-equivalent if it slipped through.
    shared = frozenset(s for s in shared if s not in {"FAST"})
    same_region = str(a.get("affected_physical_region")) == str(b.get("affected_physical_region"))
    different_model = str(a.get("model")) != str(b.get("model"))
    jac = jaccard(mechanism_tokens(a), mechanism_tokens(b))
    tag_overlap = resource_tags(a) & resource_tags(b)
    if shared and (same_region or different_model):
        return {
            "kind": "shared_distinctive_stem",
            "mechanism": (
                f"shared distinctive env stems {sorted(shared)} with "
                f"same_region={same_region} different_model={different_model}; "
                "this is the same control idea under two names, not a merge of ids"
            ),
        }
    if different_model and jac >= 0.45 and tag_overlap:
        return {
            "kind": "cross_model_mechanism_jaccard",
            "mechanism": (
                f"mechanism-token jaccard={jac:.3f} and shared resource tags "
                f"{sorted(tag_overlap)}; candidate_id string similarity was not used"
            ),
        }
    return None


def equivalence_classes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group the same idea under two names. Never union on candidate_id spelling."""
    ids = [cid(r) for r in sorted(rows, key=cid)]
    parent = {i: i for i in ids}
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str, ev: dict[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            evidence[ra].append(ev)
            return
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        evidence[ra].extend(evidence.pop(rb, []))
        evidence[ra].append(ev)

    ordered = sorted(rows, key=cid)
    for a, b in combinations(ordered, 2):
        geo = _geometry_refinement(a, b) or _geometry_refinement(b, a)
        if geo:
            union(cid(a), cid(b), geo)
            continue
        twin = _mechanism_twin(a, b)
        if twin:
            union(cid(a), cid(b), twin)

    grouped: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        grouped[find(i)].append(i)
    classes: list[dict[str, Any]] = []
    for idx, root in enumerate(sorted(grouped), start=1):
        members = sorted(grouped[root])
        ev = evidence.get(root, [])
        # Dedup evidence by (kind, mechanism).
        seen_ev = []
        seen_keys: set[tuple[str, str]] = set()
        for item in ev:
            key = (item["kind"], item["mechanism"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            seen_ev.append(item)
        classes.append(
            {
                "class_id": f"E{idx:03d}",
                "members": members,
                "size": len(members),
                "singleton": len(members) == 1,
                "evidence": seen_ev,
                "merge_rule": (
                    "union only on geometry_refinement (same region + declared "
                    "parent + geo value refined by _vec) or on shared distinctive "
                    "env stems / cross-model mechanism jaccard. candidate_id "
                    "spelling is never a reason to merge."
                ),
            }
        )
    return classes


# ---------------------------------------------------------------------------
# Redundant pruning
# ---------------------------------------------------------------------------

def _dispatch_score(text: Any) -> int | None:
    t = str(text or "").strip().lower()
    if not t:
        return None
    if t.startswith("0") and (len(t) == 1 or not t[1].isdigit()):
        return 0
    if "no assumed" in t or "not countable" in t or "no reduction claimed" in t:
        return None
    m = _DISPATCH_FROM_TO.search(t)
    if m:
        return int(m.group(1)) - int(m.group(2))
    if "reduce by two" in t:
        return 2
    if "reduce by one" in t:
        return 1
    m = _DISPATCH_BY_NUM.search(t)
    if m:
        return int(m.group(1))
    if t.startswith("reduce"):
        return 1
    return None


def _zeroish_effect(text: Any) -> bool:
    t = str(text or "").strip().lower()
    return (
        not t
        or t.startswith("0")
        or t.startswith("none")
        or t.startswith("no representation")
        or "no assumed" in t
    )


def effect_tuple(row: Mapping[str, Any]) -> tuple[int, int, int] | None:
    dispatch = _dispatch_score(row.get("expected_dispatch_reduction"))
    elim = 0 if _zeroish_effect(row.get("expected_eliminated_work")) else 1
    inter = 0 if _zeroish_effect(row.get("expected_intermediate_byte_reduction")) else 1
    if dispatch is None:
        return None
    return (dispatch, elim, inter)


def _same_risk(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (
        str(a.get("parity_contract") or "") == str(b.get("parity_contract") or "")
        and str(a.get("capability_contract") or "") == str(b.get("capability_contract") or "")
        and str(a.get("model") or "") == str(b.get("model") or "")
    )


def _in_lineage(a_id: str, b_id: str, declared: Sequence[Mapping[str, str]]) -> bool:
    return b_id in descendants_of(a_id, declared) or a_id in descendants_of(b_id, declared)


def prune_redundant(
    rows: Sequence[Mapping[str, Any]],
    declared: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Mark B redundant when A shares region, same risk, strictly stronger effect.

    Parent/child geometry refinements are lineage, not domination: the parent is
    the control the child is measured against, so it is never pruned.
    """
    declared = list(declared if declared is not None else declared_edges(rows))
    ordered = sorted(rows, key=cid)
    out: list[dict[str, Any]] = []
    for weak in ordered:
        dominators: list[str] = []
        weak_eff = effect_tuple(weak)
        for strong in ordered:
            if cid(strong) == cid(weak):
                continue
            if str(strong.get("affected_physical_region")) != str(weak.get("affected_physical_region")):
                continue
            if not _same_risk(strong, weak):
                continue
            if _in_lineage(cid(strong), cid(weak), declared):
                continue
            strong_eff = effect_tuple(strong)
            if weak_eff is None or strong_eff is None:
                continue
            if strong_eff > weak_eff and not (weak_eff > strong_eff):
                # strictly greater on the lexicographic (dispatch, elim, inter)
                # AND at least as strong on every axis.
                if all(s >= w for s, w in zip(strong_eff, weak_eff)) and any(
                    s > w for s, w in zip(strong_eff, weak_eff)
                ):
                    dominators.append(cid(strong))
        if dominators:
            out.append(
                {
                    "candidate_id": cid(weak),
                    "redundant": True,
                    "dominated_by": sorted(dominators),
                    "region": weak.get("affected_physical_region"),
                    "mechanism": (
                        "same affected_physical_region, same parity/capability risk, "
                        "strictly weaker expected (dispatch, eliminated, intermediate) "
                        "and not a parent/child of the dominator"
                    ),
                }
            )
        else:
            out.append(
                {
                    "candidate_id": cid(weak),
                    "redundant": False,
                    "dominated_by": [],
                    "region": weak.get("affected_physical_region"),
                    "mechanism": None,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Interaction prediction
# ---------------------------------------------------------------------------

def predict_interactions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pairs whose joint effect is predicted to be non-additive.

    Predicted-additive compatible pairs are intentionally omitted: a stage that
    cannot change a decision is not scheduled.
    """
    ordered = sorted(rows, key=cid)
    index = _by_id(ordered)
    predicted: list[dict[str, Any]] = []
    for a, b in combinations(ordered, 2):
        if not cell_is_compatible([a, b]):
            # Conflicts are recorded on the graph, not as factorial pairs.
            continue
        env_a, env_b = mutation_env(a), mutation_env(b)
        if env_subsumes(env_a, env_b) or env_subsumes(env_b, env_a):
            # Measuring the pair cannot change a decision vs measuring the
            # subsuming child alone.
            continue
        tags = resource_tags(a) & resource_tags(b)
        if not tags:
            continue
        region_shared = str(a.get("affected_physical_region")) == str(b.get("affected_physical_region"))
        deps_a = {str(d) for d in (a.get("dependencies") or [])}
        deps_b = {str(d) for d in (b.get("dependencies") or [])}
        precondition = cid(a) in deps_b or cid(b) in deps_a
        if region_shared:
            kind = "shared_region"
            mechanism = (
                "same affected_physical_region; joint enablement is a different "
                "physical assignment than either single"
            )
        elif precondition:
            kind = "precondition"
            mechanism = "one candidate is a declared dependency of the other"
        else:
            kind = "shared_resource"
            mechanism = (
                "shared resource tags "
                + str(sorted(tags))
                + "; overlapping host ceremony, organ path, or GEMV occupancy "
                "is predicted to make the joint effect non-additive"
            )
        predicted.append(
            {
                "a": cid(a),
                "b": cid(b),
                "kind": kind,
                "mechanism": mechanism,
                "shared_resource_tags": sorted(tags),
                "models": [a.get("model"), b.get("model")],
            }
        )
    predicted.sort(key=lambda p: (p["a"], p["b"]))
    # index is unused except to keep the helper honest if we later want names.
    del index
    return predicted


# ---------------------------------------------------------------------------
# Lineage / scars
# ---------------------------------------------------------------------------

def lineage_scars(
    rows: Sequence[Mapping[str, Any]],
    declared: Sequence[Mapping[str, str]] | None = None,
    classes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    declared = list(declared if declared is not None else declared_edges(rows))
    classes = list(classes if classes is not None else equivalence_classes(rows))
    class_of: dict[str, list[str]] = {}
    for cl in classes:
        for member in cl["members"]:
            class_of[member] = list(cl["members"])
    children: dict[str, list[str]] = defaultdict(list)
    for edge in declared:
        children[edge["from"]].append(edge["to"])
    by_region: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_region[str(row.get("affected_physical_region") or "")].append(cid(row))
    by_parent: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for dep in row.get("dependencies") or []:
            by_parent[str(dep)].append(cid(row))
    out: dict[str, Any] = {}
    for row in sorted(rows, key=cid):
        ident = cid(row)
        hard = descendants_of(ident, declared)
        siblings = sorted(
            {
                other
                for dep in (row.get("dependencies") or [])
                for other in by_parent.get(str(dep), [])
                if other != ident
            }
        )
        eq_sibs = sorted(m for m in class_of.get(ident, []) if m != ident)
        region_peers = sorted(
            p for p in by_region.get(str(row.get("affected_physical_region") or ""), []) if p != ident
        )
        out[ident] = {
            "hard_invalidates_descendants": hard,
            "declared_siblings": siblings,
            "equivalence_siblings_questioned": eq_sibs,
            "same_region_peers_remain_measurable": [
                p for p in region_peers if p not in hard
            ],
            "mechanism": (
                "a rejection hard-invalidates declared descendants. equivalence "
                "siblings are questioned (especially cross-model: Odyssey II "
                "transfer is not automatic). same-region peers that are not "
                "descendants remain measurable because they are a different "
                "exact_mutation."
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Promotion / rejection — derived from the queue, not invented
# ---------------------------------------------------------------------------

def _funnel_lanes(queue: Mapping[str, Any], ident: str) -> list[str]:
    funnel = queue.get("funnel") or {}
    lanes: list[str] = []
    for name, value in funnel.items():
        if isinstance(value, list) and ident in value:
            lanes.append(str(name))
    return sorted(lanes)


def promotion_table(queue: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    transitions = queue.get("status_transitions") or {}
    funnel = queue.get("funnel") or {}
    policy = queue.get("queue_policy") or {}
    meas = queue.get("measurement_contract") or {}
    promotion_rule = funnel.get("promotion_rule")
    table: list[dict[str, Any]] = []
    for row in sorted(rows, key=cid):
        status = str(row.get("status") or "")
        legal_next = list(transitions.get(status) or [])
        ident = cid(row)
        prereqs: list[str] = []
        reject_reasons: list[str] = []
        if promotion_rule:
            prereqs.append(f"funnel.promotion_rule: {promotion_rule}")
        if policy.get("protected_start_requires_existing_hcli_lease"):
            prereqs.append("queue_policy.protected_start_requires_existing_hcli_lease")
        if policy.get("protected_start_requires_machine_quiescence"):
            prereqs.append("queue_policy.protected_start_requires_machine_quiescence")
        if policy.get("diagnostic_results_do_not_promote"):
            prereqs.append(
                "queue_policy.diagnostic_results_do_not_promote "
                "(DIAGNOSTIC_RELATIVE never promotes)"
            )
        if meas.get("protected_pass_requires_all_fields"):
            prereqs.append("measurement_contract.protected_pass_requires_all_fields")
        if meas.get("null_policy"):
            prereqs.append(f"measurement_contract.null_policy: {meas['null_policy']}")
        if row.get("parity_contract"):
            reject_reasons.append(f"parity_contract: {row['parity_contract']}")
        if row.get("capability_contract"):
            reject_reasons.append(f"capability_contract: {row['capability_contract']}")
        if row.get("blocked_reason"):
            reject_reasons.append(f"blocked_reason: {row['blocked_reason']}")
        blocked_from_promotion = ident not in (funnel.get("promotion") or [])
        table.append(
            {
                "candidate_id": ident,
                "status": status,
                "funnel_lanes": _funnel_lanes(queue, ident),
                "legal_next_statuses": sorted(legal_next),
                "can_enter_protected_pass": "PROTECTED_PASS" in legal_next,
                "can_enter_promotion_list": status in {"PROTECTED_PASS", "INTEGRATED"},
                "currently_in_funnel_promotion": not blocked_from_promotion,
                "promotion_prerequisites": prereqs,
                "rejection_reasons_from_queue": reject_reasons,
                "source": (
                    "status_transitions, funnel, queue_policy, measurement_contract, "
                    "parity_contract, capability_contract, blocked_reason — all copied "
                    "from the queue document. this sidecar invents no new status."
                ),
            }
        )
    return table


# ---------------------------------------------------------------------------
# Staged factorial plan
# ---------------------------------------------------------------------------

READY_STATUSES = frozenset({"READY_DIAGNOSTIC", "READY_PROTECTED"})
MEASURABLE_NOW = frozenset({"READY_PROTECTED"})

# The first protected return is intentionally a small, named frontier. These
# are the Flash mechanisms with a concrete source implementation and a
# material physical falsifier; the rest of the blocked queue remains in the
# graph but is not promoted into the first return window.
PROTECTED_FLASH_RETURN_ORDER = (
    "flash-p7-mhc-pre-simdgroup",
    "flash-device-mhc-state",
    "flash-p6-hash-single-command-buffer",
    "flash-p6-prefix-concurrent-wave",
    "flash-p6-routed-fp4-gate-up-swiglu-fused",
    "flash-p6-routed-fp4-down-bf16-fused",
    "flash-p6-batched-down-qat",
    "flash-shared-fp8-gate-up-swiglu-fused",
    "flash-shared-fp8-down-combine-fused",
    "flash-p6-fused-down-shared-combine",
    "flash-pipeline-cache-reuse",
    "flash-pipeline-id-resolution",
    "flash-p6-learned-reader-reuse",
    "flash-p6-learned-expert-cache-reuse",
    "flash-p6-fused-epilogue-stack",
)

# Put the authoritative fast profile and host-ceremony controls first, then
# isolate the geometry/fusion mechanisms. All 13 READY_PROTECTED Qwen rows
# are retained; this is an execution order, not a pruning decision.
PROTECTED_QWEN_FIRST_ORDER = (
    "qwen27-fast-profile",
    "qwen27-pipeline-state-elision",
    "qwen27-pipeline-cache-reuse",
    "qwen27-pipeline-id-resolution",
    "qwen27-commit-timing-elision",
    "qwen27-encoder-label-elision",
    "qwen27-affine2-splitk4",
    "qwen27-q2f-splitk4",
    "qwen27-q4-vecgroup-x64",
    "qwen27-gqa-qkv-fusion",
    "qwen27-attention-gate-fusion",
    "qwen27-deltanet-inproj-fusion",
    "qwen27-ba-delta-fusion",
)

PROTECTED_MEASUREMENT_FIELDS = (
    "total_nx_bytes",
    "resident_bytes",
    "active_representation_bytes_per_token",
    "actual_read_bytes_per_token",
    "transient_bytes_per_token",
    "gpu_ns_per_token",
    "complete_wall_ns_per_accepted_token",
    "dispatches_per_token",
    "sync_ns_per_token",
    "accepted_tps",
    "fallback_count",
)


def _ordered_present(
    identifiers: Iterable[str],
    preferred: Sequence[str],
) -> list[str]:
    present = {str(identifier) for identifier in identifiers}
    ordered = [identifier for identifier in preferred if identifier in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _batch_candidate_spec(
    row: Mapping[str, Any],
    *,
    sequence: int,
    batch: str,
    execution_state: str,
    selection_reason: str,
    requires_survivors: Sequence[str] = (),
) -> dict[str, Any]:
    """Copy the exact queue contract into a non-executable batch plan row."""
    return {
        "sequence": sequence,
        "batch": batch,
        "candidate_id": cid(row),
        "model": row.get("model"),
        "queue_status": row.get("status"),
        "execution_state": execution_state,
        "selection_reason": selection_reason,
        "exact_mutation": row.get("exact_mutation") or {},
        "control_configuration": row.get("control_configuration") or {},
        "mutation_env": dict(sorted(mutation_env(row).items())),
        "control_env": dict(sorted(control_env(row).items())),
        "dependencies": list(row.get("dependencies") or []),
        "requires_survivors": list(requires_survivors),
        "affected_physical_region": row.get("affected_physical_region"),
        "baseline_path": row.get("baseline_path"),
        "diagnostic_command": list(row.get("diagnostic_command") or []),
        "protected_command": list(row.get("protected_command") or []),
        "parity_contract": row.get("parity_contract"),
        "capability_contract": row.get("capability_contract"),
        "expected_eliminated_work": row.get("expected_eliminated_work"),
        "expected_dispatch_reduction": row.get("expected_dispatch_reduction"),
        "expected_intermediate_byte_reduction": row.get(
            "expected_intermediate_byte_reduction"
        ),
        "expected_active_byte_change": row.get("expected_active_byte_change"),
        "expected_gpu_ns_mechanism": row.get("expected_gpu_ns_mechanism"),
        "scope_tags": list(row.get("scope_tags") or []),
        "transfer_evidence": list(row.get("transfer_evidence") or []),
        "source_evidence": list(row.get("source_evidence") or []),
        "blocked_reason": row.get("blocked_reason"),
        "allowed_outcomes": [
            "IMPLEMENTED_UNMEASURED",
            "REJECTED_PARITY",
            "REJECTED_PHYSICAL",
            "PHYSICAL_WIN_MODEL_LOCAL",
            "PHYSICAL_WIN_FAMILY",
            "GENERIC_CANDIDATE",
            "GENERIC_VERIFIED",
        ],
    }


def protected_batch_plan(
    queue: Mapping[str, Any],
    staged: Mapping[str, Any],
    interactions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact first protected return without executing anything.

    Qwen's READY_PROTECTED singleton cells are the first executable batch. The
    Flash rows are an explicit return batch, but remain blocked until a
    source-independent NX and qualified Metal authority exist. This distinction
    keeps the plan useful while preserving the fail-closed physical boundary.
    """
    rows = list(queue.get("candidates") or [])
    index = _row_map(rows)
    stage1_ids = [
        identifier
        for cell in staged.get("cells", [])
        if cell.get("stage") == "1"
        for identifier in cell.get("candidates", [])
    ]
    qwen_ids = _ordered_present(
        (
            identifier
            for identifier in stage1_ids
            if identifier in index and index[identifier].get("model") == "Qwen27"
        ),
        PROTECTED_QWEN_FIRST_ORDER,
    )
    qwen_specs = [
        _batch_candidate_spec(
            index[identifier],
            sequence=sequence,
            batch="QWEN_FIRST_PROTECTED_SINGLES",
            execution_state="READY_ON_AUTHORITY",
            selection_reason=(
                "stage-1 singleton from the dependency-aware plan; run against "
                "this row's exact control_env before any composition"
            ),
            requires_survivors=(),
        )
        for sequence, identifier in enumerate(qwen_ids, start=1)
    ]

    flash_missing = [
        identifier
        for identifier in PROTECTED_FLASH_RETURN_ORDER
        if identifier not in index
    ]
    flash_specs: list[dict[str, Any]] = []
    for sequence, identifier in enumerate(PROTECTED_FLASH_RETURN_ORDER, start=1):
        row = index.get(identifier)
        if row is None:
            continue
        dependencies = [str(value) for value in (row.get("dependencies") or [])]
        if identifier == "flash-p6-fused-down-shared-combine":
            reason = (
                "high-information full downstream fusion; run only after the "
                "routed-down and shared-down singleton outcomes are known"
            )
        elif identifier == "flash-p6-fused-epilogue-stack":
            reason = (
                "terminal composed P6 stack; run only after its primitive and "
                "full-down parents survive parity and physical gates"
            )
        elif identifier == "flash-p6-learned-expert-cache-reuse":
            reason = (
                "highest-EV route/cache descendant; isolate reader reuse before "
                "testing bounded six-expert cache reuse"
            )
        elif identifier == "flash-pipeline-id-resolution":
            reason = (
                "pipeline-cache descendant; isolate stable ID resolution only "
                "after warmed handle persistence is accepted"
            )
        else:
            reason = (
                "material dispatch, wait, resource-admission, fusion, or "
                "route/cache mechanism selected for the Flash return batch"
            )
        execution_state = (
            "CONTINGENT_AFTER_SURVIVORS"
            if dependencies
            else "WAITING_FOR_FLASH_AUTHORITY"
        )
        flash_specs.append(
            _batch_candidate_spec(
                row,
                sequence=sequence,
                batch="FLASH_RETURN_PROTECTED_SINGLETONS_AND_COMPOSITIONS",
                execution_state=execution_state,
                selection_reason=reason,
                requires_survivors=dependencies,
            )
        )

    interaction_pairs = []
    for cell in staged.get("cells", []):
        if cell.get("stage") not in {"2", "3"}:
            continue
        members = [str(identifier) for identifier in cell.get("candidates", [])]
        if not members or not all(identifier in index for identifier in members):
            continue
        prediction = next(
            (
                prediction
                for prediction in interactions
                if set(
                    (str(prediction.get("a")), str(prediction.get("b")))
                )
                == set(members)
            ),
            None,
        )
        interaction_pairs.append(
            {
                "cell_id": cell.get("cell_id"),
                "stage": cell.get("stage"),
                "kind": cell.get("kind"),
                "candidates": members,
                "requires_survivors": list(cell.get("requires_survivors") or []),
                "mutex_group": cell.get("mutex_group"),
                "predicted_interaction": (
                    {
                        "kind": prediction.get("kind"),
                        "mechanism": prediction.get("mechanism"),
                    }
                    if prediction is not None
                    else None
                ),
                "execution_state": "CONTINGENT_AFTER_SINGLETON_SURVIVORS",
            }
        )

    return {
        "schema": "hawking.future.protected_batch_plan.v1",
        "version": 1,
        "status": "WAITING_FOR_AUTHORITY",
        "purpose": (
            "exact first protected batch for the current Accelerator frontier; "
            "Qwen is the authoritative executable control lane, while Flash is "
            "a staged return plan held closed until source-independent NX and "
            "qualified Metal authority return"
        ),
        "frontier_snapshot": {
            "queue_candidate_count": len(rows),
            "queue_fingerprint": queue.get("fingerprint"),
            "qwen_ready_protected_count": len(qwen_specs),
            "qwen_first_batch_count": len(qwen_specs),
            "flash_return_batch_count": len(flash_specs),
            "flash_return_missing_ids": flash_missing,
            "staged_cell_count": (staged.get("staged") or {}).get("cell_count"),
            "blocked_candidates_remain_unscheduled": True,
        },
        "execution_authority": {
            "plan_only": True,
            "executes_benchmark": False,
            "acquires_lease": False,
            "quiesces_machine": False,
            "current_gpu_authority": False,
            "qwen_gate": (
                "queue rows are READY_PROTECTED, but a protected HCLI lease "
                "and QUIESCENT machine are still required at execution time"
            ),
            "flash_gate": (
                "do not run Flash rows until a source-independent Flash NX "
                "executable, Metal GPU/compiler capability, and protected "
                "complete-token receipt path are qualified"
            ),
            "when_metal_returns": (
                "STOP INVENTING: execute the named batch, record parity first, "
                "then prune, combine, and reprofile from protected receipts"
            ),
        },
        "current_environment": {
            "repo_root": str(REPO),
            "queue_source": queue.get("_loaded_from"),
            "qwen_profile": "hcli/hawking-native.sealed-3.14.json",
            "qwen_protected_protocol": {
                "warmup_requests": 2,
                "measure_requests": 10,
                "max_new_tokens": 32,
                "pairing": "ABAB interleaving within one protected lease",
            },
            "flash_command_mode": (
                "source_oracle_or_scaffold only; not a protected timing command"
            ),
            "metal_compiler": "UNAVAILABLE_ON_CURRENT_HOST",
            "metal_gpu": "UNAVAILABLE_ON_CURRENT_HOST",
            "teacher_capture_rows": 0,
            "flash_physical_ebpw": "UNKNOWN",
            "prospective_meta_bpw": 0.8871807728336929,
            "protected_lock_paths": [
                ".hcli/locks/protected-accelerator-bench.lock",
                ".hcli/locks/qwen-protected-bench.lock",
            ],
            "lock_policy": (
                "do not clear or seize a live lock; holder uncertainty is a "
                "fail-closed gate"
            ),
            "machine_contamination": (
                "latest qualification walk classified the host HEAVY; do not "
                "quiesce or pause standing workers from this sidecar"
            ),
            "static_preflight": {
                "error_count": 0,
                "physical_correctness_proven": False,
            },
        },
        "qwen_first_batch": {
            "name": "QWEN_FIRST_PROTECTED_SINGLES",
            "model": "Qwen27",
            "authoritative_control": {
                "profile": "hcli/hawking-native.sealed-3.14.json",
                "fast_profile_anchor": "qwen27-fast-profile",
                "rule": (
                    "qwen27-fast-profile is run first against its empty "
                    "control; every other row uses its copied control_env, "
                    "never an implicit default"
                ),
            },
            "count": len(qwen_specs),
            "run_order": qwen_specs,
            "after_singletons": {
                "predicted_interaction_and_union_cells": interaction_pairs,
                "rule": (
                    "run only cells whose required singleton survivors remain "
                    "parity-clean and physically faster; geometry mutexes and "
                    "same-region conflicts are never co-scheduled"
                ),
            },
        },
        "flash_return_batch": {
            "name": "FLASH_RETURN_PROTECTED_SINGLETONS_AND_COMPOSITIONS",
            "model": "Flash",
            "count": len(flash_specs),
            "run_order": flash_specs,
            "rule": (
                "all rows remain queue BLOCKED and execution-closed now; after "
                "Flash authority returns, run singleton rows first, then the "
                "full-down and terminal-stack descendants only when parents "
                "survive"
            ),
        },
        "rejection_thresholds": {
            "paired_repetitions_minimum": 7,
            "statistics": {
                "primary": (
                    "paired median of "
                    "complete_wall_ns_per_accepted_token"
                ),
                "uncertainty": "paired bootstrap CI95 and IQR",
                "decision_delta": (
                    "candidate minus matched control; lower is faster"
                ),
                "win_requires": (
                    "candidate median strictly below control and paired CI95 "
                    "upper bound strictly below zero; never decide from mean "
                    "alone"
                ),
            },
            "parity": {
                "outcome": "REJECTED_PARITY",
                "reject_if": [
                    "token sequence differs",
                    "accepted-token count or stop behavior differs",
                    "output hash or numeric tolerance contract differs",
                    "route/top-K tie, source-order accumulation, or BF16 round-trip differs",
                ],
            },
            "physical": {
                "outcome": "REJECTED_PHYSICAL",
                "reject_if": [
                    "no protected complete-token receipt",
                    "any required measurement field is missing",
                    "fallback_count is nonzero or capability checks fail",
                    "lease, quiescence, persistent resident identity, or clean-window checks fail",
                    "complete-token latency does not beat the matched control under the paired CI rule",
                ],
                "required_measurement_fields": list(PROTECTED_MEASUREMENT_FIELDS),
            },
            "outcome_vocabulary": [
                "IMPLEMENTED_UNMEASURED",
                "REJECTED_PARITY",
                "REJECTED_PHYSICAL",
                "PHYSICAL_WIN_MODEL_LOCAL",
                "PHYSICAL_WIN_FAMILY",
                "GENERIC_CANDIDATE",
                "GENERIC_VERIFIED",
            ],
            "scope_rule": (
                "a protected win is MODEL_LOCAL first; PHYSICAL_WIN_FAMILY, "
                "GENERIC_CANDIDATE, and GENERIC_VERIFIED require explicit "
                "cross-model transfer evidence, not structural similarity"
            ),
        },
        "representation_boundary": {
            "prospective_meta_bpw": 0.8871807728336929,
            "qualified_physical_ebpw": "UNKNOWN",
            "teacher_capture_rows": 0,
            "binding_dependency": "REAL_TEACHER_CORPUS",
            "rule": (
                "do not synthesize teacher rows or fit downstream meta "
                "encodings before qualified teacher capture"
            ),
        },
        "source_receipts": [
            "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
            "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
            "receipts/headless/ACCELERATOR_REPATRIATION_EFFECTS.json",
            "receipts/future/STATIC_KERNEL_PREFLIGHT.json",
            "receipts/future/QUALIFICATION_PIPELINE.json",
        ],
        "claim_boundary": (
            "static protected-batch plan only; no hardware measurement, "
            "physical EBPW, latency, or throughput is claimed"
        ),
    }


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {cid(r): r for r in rows}


def _redundant_ids(pruned: Sequence[Mapping[str, Any]]) -> set[str]:
    return {p["candidate_id"] for p in pruned if p.get("redundant")}


def staged_factorial_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    interactions: Sequence[Mapping[str, Any]] | None = None,
    pruned: Sequence[Mapping[str, Any]] | None = None,
    declared: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    declared = list(declared if declared is not None else declared_edges(rows))
    interactions = list(interactions if interactions is not None else predict_interactions(rows))
    pruned = list(pruned if pruned is not None else prune_redundant(rows, declared))
    skip = _redundant_ids(pruned)
    index = _row_map(rows)

    measurable = [
        r
        for r in sorted(rows, key=cid)
        if r.get("status") in MEASURABLE_NOW and cid(r) not in skip
    ]
    contingent = [
        r
        for r in sorted(rows, key=cid)
        if r.get("status") == "STATIC_ONLY" and cid(r) not in skip
    ]
    blocked = [
        r
        for r in sorted(rows, key=cid)
        if r.get("status") == "BLOCKED"
    ]

    stages: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    def add_cell(
        *,
        stage_id: str,
        cell_id: str,
        members: Sequence[str],
        kind: str,
        status: str,
        disambiguates: str,
        requires: Sequence[str] = (),
        mutex_group: str | None = None,
    ) -> None:
        picked = [index[m] for m in members if m in index]
        if len(picked) != len(members):
            return
        # The refusal lives here: a cell that cannot be formed is not scheduled.
        assert_cell_compatible(picked)
        cell = {
            "cell_id": cell_id,
            "stage": stage_id,
            "kind": kind,
            "status": status,
            "candidates": list(members),
            "model": picked[0].get("model") if picked else None,
            "planned_evidence_rung": (
                "PROTECTED_ABSOLUTE"
                if all(r.get("status") == "READY_PROTECTED" for r in picked)
                else "STATIC_ONLY"
            ),
            "disambiguates": disambiguates,
            "requires_survivors": list(requires),
            "mutex_group": mutex_group,
            "executes_benchmark": False,
            "acquires_lease": False,
            "note": (
                "STATIC_ONLY plan cell. sidecar will not run diagnostic_command "
                "or protected_command and will not take a GPU lease."
            ),
        }
        cells.append(cell)

    # Stage 1 — singles. Each cell is one candidate vs its matched control.
    stage1_ids: list[str] = []
    for row in measurable:
        ident = cid(row)
        stage1_ids.append(ident)
        add_cell(
            stage_id="1",
            cell_id=f"S1-{ident}",
            members=[ident],
            kind="single",
            status="SCHEDULED",
            disambiguates=(
                f"main effect of {ident} against its control_configuration on "
                f"{row.get('affected_physical_region')}; whether the candidate "
                "survives parity/capability and beats the matched control"
            ),
        )
    stages.append(
        {
            "stage_id": "1",
            "name": "singles",
            "status": "SCHEDULED",
            "disambiguates": (
                "main effect of each READY_PROTECTED candidate vs its matched "
                "control. a singleton cell cannot be skipped: without it later "
                "composition has no survivor set."
            ),
            "cell_ids": [f"S1-{i}" for i in stage1_ids],
        }
    )

    # Stage 2 — only predicted non-additive pairs among measurable candidates.
    measurable_ids = {cid(r) for r in measurable}
    pair_ids: list[str] = []
    for pred in interactions:
        a, b = pred["a"], pred["b"]
        if a not in measurable_ids or b not in measurable_ids:
            continue
        if a in skip or b in skip:
            continue
        left, right = (a, b) if a < b else (b, a)
        cell_id = f"S2-{left}__{right}"
        pair_ids.append(cell_id)
        add_cell(
            stage_id="2",
            cell_id=cell_id,
            members=[left, right],
            kind="pair",
            status="SCHEDULED",
            disambiguates=(
                f"whether {left} × {right} is additive; predicted {pred['kind']} "
                f"non-additivity via {pred['mechanism']}"
            ),
            requires=[left, right],
        )
    stages.append(
        {
            "stage_id": "2",
            "name": "predicted_nonadditive_pairs",
            "status": "SCHEDULED",
            "disambiguates": (
                "pairwise interaction only where a shared region, shared resource, "
                "or precondition predicts non-additivity. predicted-additive pairs "
                "are omitted because they cannot change a compose/drop decision "
                "beyond the two singles."
            ),
            "cell_ids": pair_ids,
        }
    )

    # Stage 3 — higher-order unions of survivors. Mutex on geometry env clique.
    ceremony = [
        cid(r)
        for r in measurable
        if "host_metal_path" in resource_tags(r) and cid(r) != "qwen27-fast-profile"
    ]
    fusion = [
        cid(r)
        for r in measurable
        if resource_tags(r) & {"attention_organ", "deltanet_organ"}
    ]
    geo_affine: list[str] = []
    geo_q2f: list[str] = []
    geo_q4: list[str] = []
    for r in measurable:
        env = mutation_env(r)
        if env.get("HAWKING_AFFINE2_GEO") in {"splitk4", "splitk4_vec"}:
            geo_affine.append(cid(r))
        if env.get("HAWKING_Q2F_GEO") in {"splitk4", "splitk4_vec"}:
            geo_q2f.append(cid(r))
        if "HAWKING_QWEN38_Q4_GEO" in env:
            geo_q4.append(cid(r))

    def _drop_subsumed(ids: Sequence[str]) -> list[str]:
        """If A's env is a proper subset of B's env, A adds nothing to a union cell."""
        keep: list[str] = []
        for ident in ids:
            env = mutation_env(index[ident])
            subsumed = False
            for other in ids:
                if other == ident:
                    continue
                other_env = mutation_env(index[other])
                if env_subsumes(env, other_env) and env != other_env:
                    subsumed = True
                    break
            if not subsumed:
                keep.append(ident)
        return keep

    compose_ids: list[str] = []
    ceremony_union = _drop_subsumed(sorted(ceremony))
    if len(ceremony_union) >= 3:
        compose_ids.append("S3-host-ceremony-union")
        add_cell(
            stage_id="3",
            cell_id="S3-host-ceremony-union",
            members=ceremony_union,
            kind="union",
            status="CONTINGENT",
            disambiguates=(
                "higher-order residual of host Metal ceremony after pairwise "
                "stage-2 accounting; skipped if fewer than three ceremony "
                "singles survive"
            ),
            requires=ceremony_union,
        )
    fusion_union = _drop_subsumed(sorted(fusion))
    if len(fusion_union) >= 3:
        compose_ids.append("S3-fusion-organ-union")
        add_cell(
            stage_id="3",
            cell_id="S3-fusion-organ-union",
            members=fusion_union,
            kind="union",
            status="CONTINGENT",
            disambiguates=(
                "whether GQA + DeltaNet fusions jointly still pass parity and "
                "whether organ-level composition has a residual beyond the two "
                "predicted pairs"
            ),
            requires=fusion_union,
        )

    def _union_branch(name: str, extra: list[str], mutex: str) -> None:
        members = _drop_subsumed(
            sorted(set(ceremony + fusion + extra + geo_q4) - {"qwen27-fast-profile"})
        )
        # Drop members that make the cell illegal (geometry clique).
        legal: list[str] = []
        for ident in members:
            trial = legal + [ident]
            if cell_is_compatible([index[i] for i in trial]):
                legal.append(ident)
        if len(legal) < 3:
            return
        compose_ids.append(name)
        add_cell(
            stage_id="3",
            cell_id=name,
            members=legal,
            kind="union",
            status="CONTINGENT",
            disambiguates=(
                "joint enablement of compatible survivors including one geometry "
                "assignment from the AFFINE2/Q2F mutex clique; the opposite "
                "geometry assignment is a different cell in the same mutex group"
            ),
            requires=legal,
            mutex_group=mutex,
        )

    _union_branch("S3-survivor-union-affine2", geo_affine, "qwen27-geo-assignment")
    _union_branch("S3-survivor-union-q2f", geo_q2f, "qwen27-geo-assignment")
    stages.append(
        {
            "stage_id": "3",
            "name": "survivor_unions",
            "status": "CONTINGENT",
            "disambiguates": (
                "higher-order composition of singles that survived stage 1. "
                "AFFINE2 vs Q2F geometry assignments are mutex because they "
                "collide on HAWKING_AFFINE2_GEO / HAWKING_Q2F_GEO. a union that "
                "equals an already-scheduled pair is not added."
            ),
            "cell_ids": compose_ids,
        }
    )

    # Contingent STATIC_ONLY descendants — only after declared parents survive.
    c_ids: list[str] = []
    for row in contingent:
        ident = cid(row)
        requires = [str(d) for d in (row.get("dependencies") or [])]
        c_ids.append(f"C-{ident}")
        add_cell(
            stage_id="C",
            cell_id=f"C-{ident}",
            members=[ident],
            kind="contingent_single",
            status="CONTINGENT",
            disambiguates=(
                f"main effect of {ident} against its control, only if declared "
                f"parents {requires} survived; skipped entirely if a parent is "
                "rejected (scar propagation)"
            ),
            requires=requires,
        )
    stages.append(
        {
            "stage_id": "C",
            "name": "contingent_static_only",
            "status": "CONTINGENT",
            "disambiguates": (
                "STATIC_ONLY descendants are not in the protected window until "
                "their declared parents survive. scheduling them now as "
                "contingent cells is a plan, not a measurement."
            ),
            "cell_ids": c_ids,
        }
    )

    blocked_unscheduled = [
        {
            "candidate_id": cid(r),
            "status": r.get("status"),
            "blocked_reason": r.get("blocked_reason"),
            "scheduled": False,
            "reason_unscheduled": (
                "status is BLOCKED; the sidecar will not invent a GPU cell for a "
                "candidate the queue itself refuses to run. legal next status is "
                "STATIC_ONLY after the blocked_reason clears."
            ),
        }
        for r in blocked
    ]

    n_all = len(rows)
    n_ready = len(measurable)
    # The exponent is the guard; the integer is only for reading. Python ints are
    # unbounded, but a queue of a few hundred rows would serialize a 100-digit
    # number into the receipt for no benefit -- so keep the value while it stays
    # readable and let the comparison fall back to the exponent. A hard cap on
    # the VALUE is what broke this the moment Codex grew the queue past 40 rows.
    naive_all = 1 << n_all if n_all <= 256 else None
    naive_ready = 1 << n_ready if n_ready <= 256 else None
    cell_count = len(cells)
    pair_count = sum(1 for c in cells if c["kind"] == "pair")
    return {
        "independent_set": {
            "measurable_now": [cid(r) for r in measurable],
            "n_measurable_now": n_ready,
            "contingent_static_only": [cid(r) for r in contingent],
            "blocked_unscheduled": [cid(r) for r in blocked],
            "redundant_skipped": sorted(skip),
            "note": (
                "measurable_now is READY_PROTECTED minus redundant rows. "
                "BLOCKED rows are in the graph and the scar table but are not "
                "factorial factors: the queue already refuses to run them."
            ),
        },
        "naive_power_set": {
            "n_all": n_all,
            "size_all": naive_all,
            "n_ready_protected_measurable": n_ready,
            "size_ready_protected": naive_ready,
            "formula": "2^N including the all-off control; this sidecar does not emit that set",
        },
        "staged": {
            "cell_count": cell_count,
            "pair_cells": pair_count,
            "single_cells": sum(1 for c in cells if c["kind"] == "single"),
            "union_cells": sum(1 for c in cells if c["kind"] == "union"),
            "contingent_cells": sum(1 for c in cells if c["status"] == "CONTINGENT"),
            "scheduled_cells": sum(1 for c in cells if c["status"] == "SCHEDULED"),
            "reduction_vs_all": (
                None if naive_all in (None, 0) else naive_all - cell_count
            ),
            "reduction_vs_ready": (
                None if naive_ready in (None, 0) else naive_ready - cell_count
            ),
            "dramatically_smaller": bool(
                naive_ready is not None
                and naive_all is not None
                and cell_count < naive_ready
                and cell_count * 8 < naive_ready
                and cell_count * 1024 < naive_all
            ),
        },
        "stages": stages,
        "cells": cells,
        "blocked_unscheduled": blocked_unscheduled,
        "conflict_isolation_rule": (
            "incompatible same-region or env-colliding mutations never share a "
            "measurement cell. assert_cell_compatible raises IncompatibleMutationError "
            "rather than silently dropping the cell. they may appear as separate "
            "singleton cells (or a singleton plus a contingent descendant)."
        ),
        "evidence_boundary": (
            "every cell is a PLAN. DIAGNOSTIC_RELATIVE would guide and never "
            "promote. PROTECTED_ABSOLUTE would decide. this sidecar produces "
            "neither. bench state stays UNKNOWN."
        ),
    }


def assert_plan_dramatically_smaller(plan: Mapping[str, Any]) -> None:
    staged = plan["staged"]
    naive = plan["naive_power_set"]
    cells = staged["cell_count"]
    n_ready = naive["n_ready_protected_measurable"]
    n_all = naive["n_all"]
    # Derive from the exponent when the convenience integer was omitted for size.
    # The guard must not weaken just because the queue got bigger -- a bigger
    # queue makes the naive power set MORE absurd, not less.
    size_ready = naive["size_ready_protected"]
    size_all = naive["size_all"]
    if size_ready is None:
        size_ready = 1 << int(n_ready)
    if size_all is None:
        size_all = 1 << int(n_all)
    if not (cells < size_ready):
        raise AssertionError(f"staged cell_count {cells} is not < 2^N_ready {size_ready}")
    if not (cells < size_all):
        raise AssertionError(f"staged cell_count {cells} is not < 2^N_all {size_all}")
    # The 8x / 1024x bars are for the live campaign size. A 3-row synthetic
    # queue cannot be 1024x smaller than 8.
    if size_ready >= 256 and not (cells * 8 < size_ready):
        raise AssertionError(
            f"staged cell_count {cells} is not 8x smaller than 2^N_ready {size_ready}"
        )
    if size_all >= 1024 and not (cells * 1024 < size_all):
        raise AssertionError(
            f"staged cell_count {cells} is not 1024x smaller than 2^N_all {size_all}"
        )
    pair_cap = (n_ready * (n_ready - 1)) // 2
    if n_ready >= 4 and staged["pair_cells"] >= pair_cap:
        raise AssertionError("stage 2 scheduled the full pair set; that is the naive factorial")


def iter_incompatible_pairs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    ordered = sorted(rows, key=cid)
    pairs: list[tuple[str, str]] = []
    for a, b in combinations(ordered, 2):
        if any(r["kind"] == "same_region_incompatible_mutation" for r in conflict_reasons(a, b)):
            left, right = sorted((cid(a), cid(b)))
            pairs.append((left, right))
    return pairs


def assert_no_incompatible_cell(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    index = _row_map(rows)
    for cell in plan["cells"]:
        picked = [index[i] for i in cell["candidates"] if i in index]
        assert_cell_compatible(picked)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def _planning_nodes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for row in sorted(rows, key=cid):
        nodes.append(
            {
                "candidate_id": cid(row),
                "model": row.get("model"),
                "status": row.get("status"),
                "affected_physical_region": row.get("affected_physical_region"),
                "dependencies": list(row.get("dependencies") or []),
                "blocked_reason": row.get("blocked_reason"),
                "mutation_env": mutation_env(row),
                "expected_dispatch_reduction": str(row.get("expected_dispatch_reduction") or ""),
                "expected_active_byte_change": str(row.get("expected_active_byte_change") or ""),
                "expected_eliminated_work": str(row.get("expected_eliminated_work") or ""),
                "expected_intermediate_byte_reduction": str(
                    row.get("expected_intermediate_byte_reduction") or ""
                ),
                "expected_gpu_ns_mechanism": str(row.get("expected_gpu_ns_mechanism") or ""),
            }
        )
    return nodes


def plan_from_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(queue.get("candidates") or [])
    graph = build_graph(rows)
    classes = equivalence_classes(rows)
    pruned = prune_redundant(rows, graph["declared_edges"])
    interactions = predict_interactions(rows)
    scars = lineage_scars(rows, graph["declared_edges"], classes)
    promotions = promotion_table(queue, rows)
    plan = staged_factorial_plan(
        rows,
        interactions=interactions,
        pruned=pruned,
        declared=graph["declared_edges"],
    )
    assert_plan_dramatically_smaller(plan)
    assert_no_incompatible_cell(plan, rows)
    protected_batch = protected_batch_plan(queue, plan, interactions)

    by_status: dict[str, int] = defaultdict(int)
    by_model: dict[str, int] = defaultdict(int)
    for row in rows:
        by_status[str(row.get("status") or "")] += 1
        by_model[str(row.get("model") or "")] += 1

    recovered = [
        {
            "path": "tools/accelerator/physical_qualification.py",
            "role": "producer of the physical qualification queue",
            "visible_in_this_worktree": (REPO / "tools/accelerator/physical_qualification.py").is_file(),
            "recovered_from": (
                "parent Codex checkout; the file is untracked on HEAD and is not "
                "imported here (it pulls hcli, outside this sparse checkout)"
            ),
            "adequate_as": "plan-first queue builder, status-transition authority, WorkUnit emitter",
            "not_adequate_as": (
                "no dependency graph over inferred conflicts, no equivalence classes, "
                "no redundant pruning, no staged factorial, no interaction prediction, "
                "no scar propagation table"
            ),
        },
        {
            "path": "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "role": "live physical qualification queue (disk authority)",
            "visible_in_this_worktree": (REPO / QUEUE_REL).is_file(),
            "loaded_from": queue.get("_loaded_from"),
            "schema": queue.get("schema"),
            "fingerprint": queue.get("fingerprint"),
            "counts": queue.get("counts"),
        },
        {
            "path": "receipts/headless/ACCELERATOR_SCOREBOARD.json",
            "role": "derived receipt view with physical_plan_score over measured receipts",
            "adequate_as": "scoreboard of already-run receipts",
            "not_adequate_as": "not a factorial planner over the 30-candidate queue",
        },
        {
            "path": "receipts/headless/ACCELERATOR_REPATRIATION_QUEUE.json",
            "role": "architecture-repatriation queue (behaviors, not kernel mutations)",
            "adequate_as": "same funnel vocabulary (diagnostic vs protected vs promotion)",
            "not_adequate_as": "different object: atlas behaviors, not these 30 candidates",
        },
        {
            "path": "tools/accelerator/fusion_planner.py",
            "role": "HUMF topology / placement planner",
            "not_adequate_as": "different problem; does not read the qualification queue",
        },
        {
            "path": "receipts/headless/CAPABILITY_TEMPLATE_FACTORIAL.json",
            "role": "prompt-template capability arms",
            "not_adequate_as": "not a candidate-mutation factorial",
        },
        {
            "path": "tools/future/_common.py",
            "role": "sidecar write_receipt / HARDWARE_FIELDS guard",
        },
        {
            "path": "tools/future/global_frontier.py",
            "role": "F002 names this module as the integration target for the idle READY_PROTECTED window",
        },
    ]

    negative = [
        "this worktree's receipts/headless copy of ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json "
        "is absent (the file is untracked on the parent Codex checkout). the planner reads it "
        "read-only via git worktree discovery; it does not copy the file into receipts/headless.",
        "tools/accelerator/physical_qualification.py is likewise untracked on the parent and is "
        "not imported here (it pulls hcli, which is outside this sparse checkout).",
        "no GPU, so no cell was executed and no DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE number exists.",
        "no same-region strictly-dominated pair on the live queue (the only same-region pairs "
        "are parent/child geometry refinements, which are lineage, not redundancy). pruning is "
        "still implemented and is tested with a synthetic dominated pair.",
        "Odyssey II transfer of a Qwen rejection onto a Flash mechanism twin is questioned, not "
        "hard-invalidated: there is no scoped law store (frontier F010).",
    ]

    return {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "static sidecar plan over Codex's live physical qualification queue. "
            "FIVE ERAS, THREE ODYSSEYS. FPGA stays inside Accelerator / Physical "
            "Compiler / Fusion. DISK STATE IS AUTHORITY. this module proposes a "
            "plan; it does not decide a physical result."
        ),
        "input": {
            "queue_schema": queue.get("schema"),
            "queue_version": queue.get("version"),
            "queue_fingerprint": queue.get("fingerprint"),
            "loaded_from": queue.get("_loaded_from"),
            "queue_sha256": (
                sha256_file(queue["_loaded_from"])
                if queue.get("_loaded_from") and Path(str(queue["_loaded_from"])).is_file()
                else None
            ),
            "candidate_count": len(rows),
            "by_status": dict(sorted(by_status.items())),
            "by_model": dict(sorted(by_model.items())),
            "funnel_promotion_rule": (queue.get("funnel") or {}).get("promotion_rule"),
        },
        "planning_nodes": _planning_nodes(rows),
        "dependency_graph": graph,
        "equivalence_classes": classes,
        "redundant_pruning": pruned,
        "predicted_interactions": interactions,
        "lineage_scars": scars,
        "promotion_and_rejection": promotions,
        "staged_factorial_plan": plan,
        "protected_batch": protected_batch,
        "recovered_implementation": recovered,
        "gaps_closed": [
            "dependency graph over declared dependencies AND inferred conflicts "
            "(same-region incompatible mutation, env-key collision, distinct model)",
            "equivalence classes with per-class evidence; splitk4 vs splitk4-vec "
            "grouped as geometry refinement; *-splitk4 name twins across regions refused",
            "redundant pruning that names the dominator; live queue has none, synthetic test covers the fire",
            "staged factorial: singles, predicted-nonadditive pairs, contingent survivor unions, "
            "contingent STATIC_ONLY descendants — not 2^N",
            "interaction/conflict prediction with a stated mechanism per pair",
            "lineage/scar table: hard-invalidate descendants, question equivalence siblings",
            "promotion prerequisites and rejection reasons copied from the queue funnel, "
            "status_transitions, queue_policy, measurement_contract, parity/capability contracts",
            "exact protected batch receipt: Qwen READY_PROTECTED singletons first, "
            "serious Flash return rows held behind authority and survivor gates, "
            "copied controls/commands, parity thresholds, and outcome vocabulary",
        ],
        "negative_findings": negative,
        "non_goals": [
            "does not execute a benchmark",
            "does not acquire or wait on an HCLI protected lease",
            "does not quiesce the machine",
            "does not advance queue status (HCLI / physical_qualification.py owns that)",
            "does not write receipts/headless or tools/accelerator",
        ],
    }


def build(queue_path: Path | None = None) -> Path:
    queue = load_queue(queue_path)
    doc = plan_from_queue(queue)
    return write_receipt(RECEIPT, doc, "tools/future/candidate_planner.py")


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
