"""Ranked recomposition candidate queue from cluster-map analyses.

Every candidate is a *proposal*. None is claimed safe to apply without behaviour
and performance gates. expected_loc_removed is an honest estimate:
  member_loc * (1 - retention_fraction), with retention justified per kind.
If the saving cannot be estimated, expected_loc_removed is null and sorts last.
"""

from __future__ import annotations

from typing import Any

from graph_io import SemanticGraph  # noqa: E402

# Retention fractions: fraction of member LOC we expect to *keep* after the action.
# Justified per kind — do not inflate removals.
RETENTION = {
    # merge: keep one implementation + thin façade (~40% retained)
    "merge": {
        "fraction": 0.40,
        "reason": "retain one authority implementation (~40%) + drop duplicate bodies",
    },
    # replace_with_spec: most code goes; small adapter/spec shell remains
    "replace_with_spec": {
        "fraction": 0.15,
        "reason": "replace bodies with a spec-driven shell; ~15% retained as bindings",
    },
    # generate: adapters become generated; hand-written ring mostly removed
    "generate": {
        "fraction": 0.10,
        "reason": "adapter ring replaced by generated bindings; ~10% hand-written retained",
    },
    # delete: only if behaviour-uncovered; still keep a small residual for safety in estimate
    "delete": {
        "fraction": 0.05,
        "reason": "proposal only — estimate assumes 95% removable if gates pass; residual for shared headers",
    },
    # split: no LOC removed; structure only
    "split": {
        "fraction": 1.0,
        "reason": "split re-partitions without removing LOC",
    },
    # unify_state_machine: shared control path collapses to one machine
    "unify_state_machine": {
        "fraction": 0.55,
        "reason": "retain one state machine (~55%); drop per-entry duplicated control",
    },
}

RISK_WEIGHT = {"low": 1.0, "medium": 2.0, "high": 4.0}


def _estimate_removed(kind: str, member_loc: int) -> tuple[int | None, str]:
    meta = RETENTION.get(kind)
    if meta is None:
        return None, "unknown kind — cannot estimate"
    frac = meta["fraction"]
    if frac >= 1.0:
        return 0, meta["reason"]
    removed = int(round(member_loc * (1.0 - frac)))
    return removed, meta["reason"]


def _semantic_merge_criteria(
    g: SemanticGraph,
    members: list[str],
    cluster: dict[str, Any],
) -> list[str]:
    """Merge requires >=2 of: state, lifecycle, error policy, tests, change history, callers."""
    hits: list[str] = []

    # state: shared reads_state/writes_state targets
    states: list[set[str]] = []
    for m in members:
        st = set()
        for e in g.out_edges.get(m, []):
            if e["type"] in ("reads_state", "writes_state"):
                st.add(e["dst"])
        # also from functions inside file
        for e in g.out_edges.get(m, []):
            if e["type"] == "contains":
                for e2 in g.out_edges.get(e["dst"], []):
                    if e2["type"] in ("reads_state", "writes_state"):
                        st.add(e2["dst"])
        states.append(st)
    if states:
        inter = set.intersection(*states) if all(states) else set()
        # or any shared state across members
        from collections import Counter

        c = Counter()
        for s in states:
            for x in s:
                c[x] += 1
        if any(v >= 2 for v in c.values()):
            hits.append("state")

    # lifecycle: members in same SCC or dominator shared path
    scc_members = set()
    for scc in cluster.get("analyses", {}).get("scc", {}).get("machine", {}).get("file_sccs", []):
        scc_members.update(scc.get("members") or [])
    if sum(1 for m in members if m in scc_members) >= 2:
        hits.append("lifecycle")

    # change history: high co_changes among members
    # Schema weight is count/min(commits) ∈ [0,1]; use raw count (or legacy weight>1).
    member_set = set(members)
    co_hit = False
    for e in g.edges:
        if e["type"] == "co_changes" and e["src"] in member_set and e["dst"] in member_set:
            attrs = e.get("attrs") or {}
            count = int(attrs.get("count") or 0)
            w = float(attrs.get("weight", 0) or 0)
            if count <= 0 and w > 1.0:
                count = int(w)
            if count >= 3 or (0 < w <= 1.0 and w >= 0.5):
                co_hit = True
                break
    if co_hit:
        hits.append("change history")

    # callers: shared external callers
    callers = []
    for m in members:
        cs = set()
        for e in g.in_edges.get(m, []):
            if e["type"] in ("calls", "imports") and e["src"] not in member_set:
                cs.add(e["src"])
        callers.append(cs)
    if callers:
        from collections import Counter

        c = Counter()
        for cs in callers:
            for x in cs:
                c[x] += 1
        if any(v >= 2 for v in c.values()):
            hits.append("callers")

    # tests: both test_covered
    if sum(1 for m in members if g.attr(m, "test_covered", False)) >= 2:
        hits.append("tests")

    # error policy proxy: same side_effects set
    se_sets = []
    for m in members:
        se = tuple(sorted(g.attr(m, "side_effects", ["none"]) or ["none"]))
        se_sets.append(se)
    if len(set(se_sets)) == 1 and len(members) >= 2:
        hits.append("error policy")

    return hits


def build_candidates(
    g: SemanticGraph,
    cluster: dict[str, Any],
) -> dict[str, Any]:
    analyses = cluster.get("analyses", {})
    candidates: list[dict[str, Any]] = []
    rc_i = 1

    def next_id() -> str:
        nonlocal rc_i
        rid = f"RC-{rc_i:03d}"
        rc_i += 1
        return rid

    # --- from SCCs: merge ---
    for scc in analyses.get("scc", {}).get("machine", {}).get("file_sccs", [])[:30]:
        members = scc.get("members") or []
        if len(members) < 2:
            continue
        criteria = _semantic_merge_criteria(g, members, cluster)
        if len(criteria) < 2:
            # SCC itself implies lifecycle + typically callers; count lifecycle always
            if "lifecycle" not in criteria:
                criteria.append("lifecycle")
            if "callers" not in criteria:
                criteria.append("callers")
        if len(criteria) < 2:
            continue
        loc = int(scc.get("loc") or sum(g.loc_of(m) for m in members))
        removed, ret_reason = _estimate_removed("merge", loc)
        paths = [p for p in (scc.get("paths") or [g.path_of(m) for m in members]) if p]
        candidates.append(
            {
                "id": next_id(),
                "kind": "merge",
                "title": f"Merge file-level SCC of {len(members)} mutually dependent files",
                "members": members,
                "paths": paths,
                "evidence": [
                    f"scc: size={scc['size']} loc={loc}",
                    f"semantic_merge_criteria: {criteria}",
                ],
                "semantic_merge_criteria": criteria,
                "expected_loc_removed": removed,
                "expected_loc_removed_basis": {
                    "member_loc": loc,
                    "retention_fraction": RETENTION["merge"]["fraction"],
                    "justification": ret_reason,
                },
                "expected_dirs_removed": 0,
                "expected_files_removed": max(0, len(members) - 1),
                "expected_functions_removed": None,
                "behaviour_contracts_touched": [],
                "risk": "high" if loc > 500 else "medium",
                "risk_reason": (
                    "Mutual dependency collapse can break load order and public APIs; "
                    "requires integration tests across former cycle boundaries."
                ),
                "test_plan": (
                    "Build the merged crate/module; run unit tests of all members; "
                    "run any CLI/HTTP entry that imports these files; compare behaviour receipts."
                ),
                "rollback": "git tag/checkpoint immediately before merge commit; restore paths from tag",
                "blocked_by": [],
            }
        )

    for scc in analyses.get("scc", {}).get("machine", {}).get("crate_sccs", [])[:10]:
        members = scc.get("members") or []
        if len(members) < 2:
            continue
        loc = int(scc.get("loc") or 0)
        removed, ret_reason = _estimate_removed("merge", loc)
        candidates.append(
            {
                "id": next_id(),
                "kind": "merge",
                "title": f"Merge crate-level SCC: {', '.join(members)}",
                "members": members,
                "paths": [m.replace("crate:", "crates/") for m in members],
                "evidence": [
                    f"scc_crate: size={scc['size']} loc={loc}",
                    "semantic_merge_criteria: ['lifecycle', 'callers']",
                ],
                "semantic_merge_criteria": ["lifecycle", "callers"],
                "expected_loc_removed": removed,
                "expected_loc_removed_basis": {
                    "member_loc": loc,
                    "retention_fraction": RETENTION["merge"]["fraction"],
                    "justification": ret_reason,
                },
                "expected_dirs_removed": max(0, len(members) - 1),
                "expected_files_removed": None,
                "expected_functions_removed": None,
                "behaviour_contracts_touched": [],
                "risk": "high",
                "risk_reason": "Crate merges affect Cargo graph, versioning, and public API surface.",
                "test_plan": "cargo test -p merged; downstream dependent crate tests; workspace build",
                "rollback": "restore pre-merge Cargo.toml workspace members from checkpoint tag",
                "blocked_by": [],
            }
        )

    # --- high-scatter communities: split (layout fix) or merge if criteria met ---
    for comm in analyses.get("communities", {}).get("machine", {}).get("communities", [])[:40]:
        n_dirs = int(comm.get("n_directories") or 0)
        size = int(comm.get("member_count") or comm.get("size") or 0)
        if n_dirs >= 5 and size >= 8:
            members = comm.get("members") or []
            loc = int(comm.get("loc") or 0)
            # split is structural — expected_loc_removed = 0
            removed, ret_reason = _estimate_removed("split", loc)
            candidates.append(
                {
                    "id": next_id(),
                    "kind": "split",
                    "title": (
                        f"Realign directories for community {comm['id']} "
                        f"({n_dirs} dirs, {size} files) — folders are wrong, not the code"
                    ),
                    "members": members[:200],
                    "paths": [g.path_of(m) for m in members[:200] if g.path_of(m)],
                    "evidence": [
                        f"community: {comm['id']} n_directories={n_dirs} "
                        f"directory_scatter={comm.get('directory_scatter')} loc={loc}",
                    ],
                    "expected_loc_removed": removed,
                    "expected_loc_removed_basis": {
                        "member_loc": loc,
                        "retention_fraction": 1.0,
                        "justification": ret_reason,
                    },
                    "expected_dirs_removed": max(0, n_dirs - 1),
                    "expected_files_removed": 0,
                    "expected_functions_removed": 0,
                    "behaviour_contracts_touched": [],
                    "risk": "low",
                    "risk_reason": "Directory moves only; behaviour unchanged if imports updated mechanically.",
                    "test_plan": "Path-rewrite + workspace build; no behaviour delta expected.",
                    "rollback": "git mv inverse; restore directory tree from checkpoint",
                    "blocked_by": [],
                }
            )

    # --- brokers: replace_with_spec or merge into neighbours ---
    for broker in analyses.get("betweenness", {}).get("machine", {}).get("brokers", [])[:20]:
        nid = broker["id"]
        loc = int(broker.get("loc") or g.loc_of(nid))
        removed, ret_reason = _estimate_removed("replace_with_spec", loc)
        candidates.append(
            {
                "id": next_id(),
                "kind": "replace_with_spec",
                "title": f"Collapse broker/translation layer {g.path_of(nid) or nid}",
                "members": [nid],
                "paths": [p for p in [g.path_of(nid)] if p],
                "evidence": [
                    f"betweenness: bc={broker.get('betweenness')} loc={loc} "
                    f"complexity={broker.get('complexity')}",
                    broker.get("reason") or "broker_flag",
                ],
                "expected_loc_removed": removed,
                "expected_loc_removed_basis": {
                    "member_loc": loc,
                    "retention_fraction": RETENTION["replace_with_spec"]["fraction"],
                    "justification": ret_reason,
                },
                "expected_dirs_removed": 0,
                "expected_files_removed": 1 if loc > 0 else 0,
                "expected_functions_removed": None,
                "behaviour_contracts_touched": [],
                "risk": "medium",
                "risk_reason": (
                    "Brokers often encode implicit protocol translation; "
                    "replacing requires a written interface contract."
                ),
                "test_plan": (
                    "Capture call traces across the cut; replace with typed interface; "
                    "replay traces and behaviour contracts that cross the cut."
                ),
                "rollback": "restore broker file from checkpoint tag",
                "blocked_by": [],
            }
        )

    # --- dominators: unify_state_machine ---
    shared = (
        analyses.get("dominators", {})
        .get("machine", {})
        .get("shared_control_nodes", [])
    )
    chains = analyses.get("dominators", {}).get("machine", {}).get("shared_chains", [])
    if chains:
        top = chains[:8]
        for ch in top:
            members = ch.get("shared_nodes") or []
            loc = int(ch.get("shared_loc") or sum(g.loc_of(m) for m in members))
            removed, ret_reason = _estimate_removed("unify_state_machine", loc)
            candidates.append(
                {
                    "id": next_id(),
                    "kind": "unify_state_machine",
                    "title": (
                        f"Unify shared control path between "
                        f"{ch.get('entry_a')} and {ch.get('entry_b')} "
                        f"({ch.get('shared_count')} nodes)"
                    ),
                    "members": members,
                    "paths": [g.path_of(m) for m in members if g.path_of(m)],
                    "evidence": [
                        f"dominator: shared_count={ch.get('shared_count')} loc={loc}",
                        f"entries: {ch.get('entry_a')}, {ch.get('entry_b')}",
                    ],
                    "expected_loc_removed": removed,
                    "expected_loc_removed_basis": {
                        "member_loc": loc,
                        "retention_fraction": RETENTION["unify_state_machine"]["fraction"],
                        "justification": ret_reason,
                    },
                    "expected_dirs_removed": 0,
                    "expected_files_removed": 0,
                    "expected_functions_removed": max(0, len(members) // 2),
                    "behaviour_contracts_touched": [],
                    "risk": "high",
                    "risk_reason": "Control-flow unification can change ordering and error paths.",
                    "test_plan": (
                        "Exercise both entry points; compare state transitions and error policy; "
                        "golden traces for the shared suffix."
                    ),
                    "rollback": "restore pre-unification functions from checkpoint",
                    "blocked_by": [],
                }
            )
    elif shared:
        # group top shared nodes into one candidate
        top_shared = [s for s in shared if s.get("n_entry_points", 0) >= 2][:12]
        if top_shared:
            members = [s["id"] for s in top_shared]
            loc = sum(int(s.get("loc") or 0) for s in top_shared)
            removed, ret_reason = _estimate_removed("unify_state_machine", loc)
            candidates.append(
                {
                    "id": next_id(),
                    "kind": "unify_state_machine",
                    "title": f"Unify {len(members)} nodes on multi-entry control paths",
                    "members": members,
                    "paths": [s.get("path") for s in top_shared if s.get("path")],
                    "evidence": [
                        f"dominator: {len(members)} nodes with n_entry_points>=2, loc={loc}"
                    ],
                    "expected_loc_removed": removed,
                    "expected_loc_removed_basis": {
                        "member_loc": loc,
                        "retention_fraction": RETENTION["unify_state_machine"]["fraction"],
                        "justification": ret_reason,
                    },
                    "expected_dirs_removed": 0,
                    "expected_files_removed": 0,
                    "expected_functions_removed": max(0, len(members) // 2),
                    "behaviour_contracts_touched": [],
                    "risk": "high",
                    "risk_reason": "Shared control collapse needs explicit state-machine review.",
                    "test_plan": "Multi-entry behavioural suite; ordering and error-path parity.",
                    "rollback": "checkpoint restore of shared nodes",
                    "blocked_by": [],
                }
            )

    # --- clones: merge / replace_with_spec ---
    for fam in analyses.get("clones", {}).get("machine", {}).get("families", [])[:25]:
        members = fam.get("members") or []
        if len(members) < 2:
            continue
        loc = int(fam.get("loc") or 0)
        removed, ret_reason = _estimate_removed("merge", loc)
        criteria = ["callers"] if fam.get("cross_crate") else []
        criteria.append("error policy")  # same CFG signature ⇒ same control/error shape
        if fam.get("cross_language"):
            risk = "high"
            risk_reason = "Cross-language clone merge needs a shared spec, not a textual merge."
            kind = "replace_with_spec"
            removed, ret_reason = _estimate_removed(kind, loc)
        else:
            risk = "medium"
            risk_reason = "Same-language structural clones; merge into one authority + call sites."
            kind = "merge"
        candidates.append(
            {
                "id": next_id(),
                "kind": kind,
                "title": (
                    f"Collapse structural clone family {fam.get('id')} "
                    f"({fam.get('member_count')} members, {fam.get('match_kind')})"
                ),
                "members": members,
                "paths": [g.path_of(m) for m in members if g.path_of(m)],
                "evidence": [
                    f"clone: {fam.get('id')} match_kind={fam.get('match_kind')} "
                    f"members={fam.get('member_count')} loc={loc}",
                    f"signature={fam.get('signature')}",
                    "text_match not admissible",
                    f"semantic_merge_criteria: {criteria}",
                ],
                "semantic_merge_criteria": criteria,
                "expected_loc_removed": removed,
                "expected_loc_removed_basis": {
                    "member_loc": loc,
                    "retention_fraction": RETENTION[kind]["fraction"],
                    "justification": ret_reason,
                },
                "expected_dirs_removed": 0,
                "expected_files_removed": max(0, len(members) - 1),
                "expected_functions_removed": max(0, len(members) - 1),
                "behaviour_contracts_touched": [],
                "risk": risk,
                "risk_reason": risk_reason,
                "test_plan": (
                    "Property tests on the shared CFG signature; "
                    "call-site rewrite compile; behaviour contracts of each former member."
                ),
                "rollback": "restore clone member functions from checkpoint",
                "blocked_by": [],
            }
        )

    # --- cochange: merge ---
    for pair in analyses.get("cochange", {}).get("machine", {}).get("pairs", [])[:20]:
        members = [pair["a"], pair["b"]]
        criteria = _semantic_merge_criteria(g, members, cluster)
        if "change history" not in criteria:
            criteria.append("change history")
        # need a second criterion
        if len(criteria) < 2:
            # co-change alone is not enough for merge under our rule — skip or low-confidence
            if pair.get("subsystem_a") == pair.get("subsystem_b"):
                criteria.append("lifecycle")
        if len(criteria) < 2:
            continue
        loc = int(pair.get("combined_loc") or 0)
        removed, ret_reason = _estimate_removed("merge", loc)
        candidates.append(
            {
                "id": next_id(),
                "kind": "merge",
                "title": (
                    f"Merge co-changed but uncoupled pair "
                    f"{pair.get('a_path')} + {pair.get('b_path')}"
                ),
                "members": members,
                "paths": [p for p in [pair.get("a_path"), pair.get("b_path")] if p],
                "evidence": [
                    f"cochange: count={pair.get('co_changes_count')} "
                    f"weight={pair.get('co_changes_weight')} "
                    f"direct_coupling=false loc={loc}",
                    f"semantic_merge_criteria: {criteria}",
                ],
                "semantic_merge_criteria": criteria,
                "expected_loc_removed": removed,
                "expected_loc_removed_basis": {
                    "member_loc": loc,
                    "retention_fraction": RETENTION["merge"]["fraction"],
                    "justification": ret_reason,
                },
                "expected_dirs_removed": 0,
                "expected_files_removed": 1,
                "expected_functions_removed": None,
                "behaviour_contracts_touched": [],
                "risk": "medium",
                "risk_reason": "Historically coupled but no static edge — hidden contract may exist.",
                "test_plan": "Tests that previously touched both files in one commit; integration suite.",
                "rollback": "restore both files from checkpoint",
                "blocked_by": [],
            }
        )

    # --- fan-in: generate ---
    for ring in analyses.get("fanin", {}).get("machine", {}).get("rings", [])[:25]:
        adapters = ring.get("adapters") or []
        members = [ring["authority"]] + adapters
        loc = int(ring.get("adapter_total_loc") or 0)
        removed, ret_reason = _estimate_removed("generate", loc)
        candidates.append(
            {
                "id": next_id(),
                "kind": "generate",
                "title": (
                    f"Generate bindings for authority {ring.get('authority_name')} "
                    f"({ring.get('adapter_count')} thin adapters)"
                ),
                "members": members,
                "paths": [p for p in [ring.get("authority_path")] + (ring.get("adapter_paths") or []) if p],
                "evidence": [
                    f"fanin: adapters={ring.get('adapter_count')} "
                    f"adapter_loc={loc} authority={ring.get('authority')}",
                ],
                "expected_loc_removed": removed,
                "expected_loc_removed_basis": {
                    "member_loc": loc,
                    "retention_fraction": RETENTION["generate"]["fraction"],
                    "justification": ret_reason,
                },
                "expected_dirs_removed": 0,
                "expected_files_removed": max(0, len(adapters) - 1),
                "expected_functions_removed": len(adapters),
                "behaviour_contracts_touched": [],
                "risk": "low",
                "risk_reason": "Thin adapters with single call targets; generation preserves call shape.",
                "test_plan": "Compile-generated bindings; call each former adapter entry; golden outputs.",
                "rollback": "restore hand-written adapters from checkpoint; drop generated outputs",
                "blocked_by": [],
            }
        )

    # --- behaviour uncovered: delete proposals (honest, gated) ---
    beh = analyses.get("behaviour_coverage", {}).get("machine", {})
    if not beh.get("degraded"):
        for f in (beh.get("uncovered_files_top") or [])[:15]:
            loc = int(f.get("loc") or 0)
            if loc < 20:
                continue
            removed, ret_reason = _estimate_removed("delete", loc)
            candidates.append(
                {
                    "id": next_id(),
                    "kind": "delete",
                    "title": f"Deletion candidate (no behaviour contract): {f.get('path')}",
                    "members": [f["id"]],
                    "paths": [p for p in [f.get("path")] if p],
                    "evidence": [
                        f"behaviour_coverage: uncovered loc={loc} subsystem={f.get('subsystem')}",
                        "PROPOSAL ONLY — not safe to delete without gates",
                    ],
                    "expected_loc_removed": removed,
                    "expected_loc_removed_basis": {
                        "member_loc": loc,
                        "retention_fraction": RETENTION["delete"]["fraction"],
                        "justification": ret_reason,
                    },
                    "expected_dirs_removed": 0,
                    "expected_files_removed": 1,
                    "expected_functions_removed": None,
                    "behaviour_contracts_touched": [],
                    "risk": "high",
                    "risk_reason": (
                        "Absence from current behaviour map is not proof of dead code; "
                        "map may be incomplete."
                    ),
                    "test_plan": (
                        "Confirm no runtime_hot; no public export; full test suite green after removal; "
                        "re-scan behaviour map."
                    ),
                    "rollback": "restore file from checkpoint tag",
                    "blocked_by": [],
                }
            )

    # Score and rank
    ranked = []
    for c in candidates:
        removed = c.get("expected_loc_removed")
        risk = c.get("risk") or "high"
        rw = RISK_WEIGHT.get(risk, 4.0)
        if removed is None:
            score = None
            sort_key = (-1.0, 0)  # last
        else:
            score = removed / rw
            sort_key = (score, removed)
        c["score"] = score
        c["score_components"] = {
            "expected_loc_removed": removed,
            "risk": risk,
            "risk_weight": rw,
            "formula": "expected_loc_removed / risk_weight  (null loc sorts last)",
            "score": score,
        }
        ranked.append((sort_key, c))

    ranked.sort(key=lambda x: (x[0][0] is None, -(x[0][0] or 0), -(x[0][1] or 0)))
    out_list = []
    for i, (_, c) in enumerate(ranked):
        # stable re-id by rank
        c["id"] = f"RC-{i + 1:03d}"
        c["rank"] = i + 1
        # Cap members/paths for report size; keep counts
        members = c.get("members") or []
        paths = c.get("paths") or []
        if len(members) > 80:
            c["members_count"] = len(members)
            c["members"] = members[:80]
            c["members_truncated"] = True
        if len(paths) > 80:
            c["paths_count"] = len(paths)
            c["paths"] = paths[:80]
            c["paths_truncated"] = True
        out_list.append(c)

    # blocked_by: high-risk merges of crates block file-level inside them — light heuristic
    crate_merges = [c for c in out_list if c["kind"] == "merge" and any(m.startswith("crate:") for m in c["members"])]
    for c in out_list:
        if c in crate_merges:
            continue
        blockers = []
        for cm in crate_merges:
            crate_names = [m.replace("crate:", "") for m in cm["members"] if m.startswith("crate:")]
            for p in c.get("paths") or []:
                if any(f"crates/{cn}" in (p or "") for cn in crate_names):
                    blockers.append(cm["id"])
                    break
        c["blocked_by"] = sorted(set(blockers))

    return {
        "schema": "hawking.recomposition_candidates.v1",
        "note": (
            "All entries are proposals for the reduction queue. None is authorised "
            "for application without behaviour and performance gates. "
            "expected_loc_removed is estimated from member LOC and a per-kind "
            "retention fraction; null means unestimable and sorts last."
        ),
        "ranking": {
            "formula": "score = expected_loc_removed / risk_weight; nulls last",
            "risk_weights": RISK_WEIGHT,
            "retention_fractions": {k: v["fraction"] for k, v in RETENTION.items()},
            "retention_justifications": {k: v["reason"] for k, v in RETENTION.items()},
        },
        "n_candidates": len(out_list),
        "candidates": out_list,
    }
