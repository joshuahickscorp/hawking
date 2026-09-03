"""Compact per-organ score index + pure-CPU budget EVALUATE.

A downward density ladder step changes only the global complete_physical_bpw
budget. Per-organ fits (output_cosine, surplus_over_null, weight_cosine,
component_bpw, payload_delta_bytes, physical_payload_sha256) are invariant.

EVALUATE is therefore:

    sort by surplus-first density key   ~ms
    greedy pack under budget ceiling    ~ms
    expert-atomic pass                  ~ms
    one-bit ceiling verdict             ~µs

NO payload I/O, NO weight loads, NO refit. The index is resident across
evaluates in one process. SEAL reads the same selection and writes payloads
(delta-only against a prior seal).
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

INDEX_SCHEMA = "hawking.ascension.activation_weighted_svd_score_index.v1"
EVALUATE_SCHEMA = "hawking.ascension.activation_weighted_svd_evaluate.v1"
# Selection sort / pack semantics — identical to the historical score path.
# Primary: surplus_over_null (activation quality). NOT weight cosine.
COMPONENTS = ("gate_proj", "up_proj", "down_proj")
MODEL_LAYERS = 48
ONE_BIT_CEILING_BPW = 1.0

# Compact fields required for evaluate + seal identity. No budget_sweep,
# no codec_metadata, no reconstruction.
INDEX_FIELDS: tuple[str, ...] = (
    "tensor_name",
    "layer",
    "expert",
    "component",
    "shape",
    "elements",
    "source_shard",
    "source_value_sha256",
    "family",
    "budget_label",
    "rank",
    "bits",
    "component_bpw",
    "under_ceiling",
    "weight_cosine",
    "weight_relative_l2",
    "output_cosine",
    "null_baseline",
    "surplus_over_null",
    "beats_null",
    "distribution_local_only",
    "n_fit_tokens",
    "n_hold_tokens",
    "physical_payload_bytes",
    "baseline_payload_bytes",
    "payload_delta_bytes",
    "physical_payload_sha256",
    "payload_path",
    "selection_metric",
    "fit_cache_key",
)


class ScoreIndexError(RuntimeError):
    """Score index is missing, corrupt, or evaluate inputs are inconsistent."""


def density_sort_key(row: Mapping[str, Any]) -> tuple:
    """Historical surplus-first sort (same as _build_selection scored.sort)."""
    return (
        -float(row["surplus_over_null"]),
        -float(row["weight_cosine"]),
        float(row["component_bpw"]),
        int(row["layer"]),
        int(row["expert"]),
        str(row["component"]),
    )


def compact_organ_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strip bulky nested fields; keep everything evaluate + seal need."""
    out: dict[str, Any] = {}
    for k in INDEX_FIELDS:
        if k in row:
            out[k] = row[k]
    # Required identity fields.
    for req in (
        "tensor_name",
        "layer",
        "expert",
        "component",
        "surplus_over_null",
        "weight_cosine",
        "component_bpw",
        "payload_delta_bytes",
        "physical_payload_sha256",
        "physical_payload_bytes",
        "output_cosine",
        "beats_null",
    ):
        if req not in out:
            raise ScoreIndexError(f"organ record missing {req}: {row.get('tensor_name')}")
    out["layer"] = int(out["layer"])
    out["expert"] = int(out["expert"])
    out["component"] = str(out["component"])
    out["surplus_over_null"] = float(out["surplus_over_null"])
    out["weight_cosine"] = float(out["weight_cosine"])
    out["component_bpw"] = float(out["component_bpw"])
    out["output_cosine"] = float(out["output_cosine"])
    out["payload_delta_bytes"] = int(out["payload_delta_bytes"])
    out["physical_payload_bytes"] = int(out["physical_payload_bytes"])
    out["baseline_payload_bytes"] = int(out.get("baseline_payload_bytes") or 0)
    out["beats_null"] = bool(out["beats_null"])
    out["elements"] = int(out.get("elements") or 0)
    return out


def filter_admitted(
    records: Sequence[Mapping[str, Any]],
    *,
    min_surplus: float = 0.0,
) -> list[dict[str, Any]]:
    """Surplus-over-null gate — same bar as selection (beats_null and min_surplus)."""
    admitted: list[dict[str, Any]] = []
    for row in records:
        if not bool(row.get("beats_null")):
            continue
        if float(row["surplus_over_null"]) < float(min_surplus):
            continue
        admitted.append(compact_organ_record(row))
    return admitted


def greedy_pack_under_ceiling(
    scored: Sequence[Mapping[str, Any]],
    *,
    budget_bpw: float,
    base_payload: int,
    elements: int,
    manifest_reserve: int,
    record_deferred: bool = True,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], int, float]:
    """Greedy pack in pre-sorted order under complete_physical_bpw ceiling.

    `scored` must already be sorted by density_sort_key. Returns
    (selected, deferred, payload_running, projected_bpw).

    Selected rows are the *same objects* as in `scored` (no per-row dict copy)
    until a later annotate pass; projected bpw is attached only when needed.
    """
    payload_running = int(base_payload)
    organs: list[Mapping[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    elems = max(int(elements), 1)
    ceiling = float(budget_bpw)
    reserve = int(manifest_reserve)
    inv_elems_8 = 8.0 / elems
    for row in scored:
        trial_payload = payload_running + int(row["payload_delta_bytes"])
        trial_bpw = (trial_payload + reserve) * inv_elems_8
        if trial_bpw <= ceiling + 1e-9:
            organs.append(row)
            payload_running = trial_payload
        elif record_deferred:
            deferred.append(
                {
                    "tensor_name": row["tensor_name"],
                    "layer": row["layer"],
                    "expert": row["expert"],
                    "component": row["component"],
                    "surplus_over_null": row["surplus_over_null"],
                    "payload_delta_bytes": row["payload_delta_bytes"],
                    "reason": "would_exceed_complete_physical_bpw_ceiling",
                }
            )
    projected = (payload_running + reserve) * inv_elems_8
    return organs, deferred, payload_running, float(projected)


def expert_atomic_enforce(
    organs: Sequence[Mapping[str, Any]],
    deferred: list[dict[str, Any]],
    payload_running: int,
    *,
    record_deferred: bool = True,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], int, dict[str, int]]:
    """Demote incomplete expert triples AFTER greedy pack (second independent source).

    Strengthening only — never admits an organ that failed earlier gates.
    """
    counts: dict[tuple[int, int], int] = {}
    for row in organs:
        key = (int(row["layer"]), int(row["expert"]))
        counts[key] = counts.get(key, 0) + 1
    incomplete = {k for k, n in counts.items() if n < len(COMPONENTS)}
    n_experts_demoted = len(incomplete)
    if not incomplete:
        return list(organs), deferred, int(payload_running), {
            "experts_demoted_for_incomplete_triple": 0,
            "organs_demoted_for_incomplete_triple": 0,
        }

    running = int(payload_running)
    kept: list[Mapping[str, Any]] = []
    n_organs_demoted = 0
    out_deferred = deferred
    for r in organs:
        key = (int(r["layer"]), int(r["expert"]))
        if key in incomplete:
            running -= int(r["payload_delta_bytes"])
            n_organs_demoted += 1
            if record_deferred:
                out_deferred.append(
                    {
                        "tensor_name": r["tensor_name"],
                        "layer": r["layer"],
                        "expert": r["expert"],
                        "component": r["component"],
                        "surplus_over_null": r["surplus_over_null"],
                        "payload_delta_bytes": r["payload_delta_bytes"],
                        "reason": "expert_triple_incomplete_runtime_requires_atomic_expert",
                    }
                )
        else:
            kept.append(r)
    stats = {
        "experts_demoted_for_incomplete_triple": n_experts_demoted,
        "organs_demoted_for_incomplete_triple": n_organs_demoted,
    }
    return kept, out_deferred, running, stats


def predicted_chain_survival(
    organs: Sequence[Mapping[str, Any]],
    *,
    n_layers: int = MODEL_LAYERS,
    cos_key: str = "output_cosine",
) -> float:
    """Product of per-layer mean output cosines — c^48 necessary-condition family."""
    buckets: list[list[float]] = [[] for _ in range(n_layers)]
    for o in organs:
        li = int(o["layer"])
        if 0 <= li < n_layers:
            buckets[li].append(float(o[cos_key]))
    observed = [i for i, xs in enumerate(buckets) if xs]
    if not observed:
        flat = [float(o[cos_key]) for o in organs]
        mean = sum(flat) / max(len(flat), 1) if flat else 0.0
        means = [mean] * n_layers
    else:
        means = [(sum(xs) / len(xs)) if xs else float("nan") for xs in buckets]
        fill = sum(means[i] for i in observed) / len(observed)
        means = [m if math.isfinite(m) else fill for m in means]
    clamped = [max(1e-12, min(1.0, float(c))) for c in means]
    return float(math.exp(sum(math.log(c) for c in clamped)))


def one_bit_ceiling_verdict(
    *,
    projected_bpw: float,
    payload_bytes: int,
    elements: int,
    manifest_reserve: int,
) -> dict[str, Any]:
    """Report whether projected complete BPW respects the one-bit law (≤ 1/1)."""
    legal = float(projected_bpw) <= ONE_BIT_CEILING_BPW + 1e-12
    return {
        "rule": "complete_artifact_bits / original_weight_count <= 1/1",
        "one_bit_ceiling_bpw": ONE_BIT_CEILING_BPW,
        "projected_complete_physical_bpw": float(projected_bpw),
        "payload_bytes": int(payload_bytes),
        "manifest_reserve_bytes": int(manifest_reserve),
        "source_weight_elements": int(elements),
        "legal": bool(legal),
        "verdict": "PASS" if legal else "FAIL_CLOSED_REBUDGET",
    }


def gravity_density_frontier_fields(
    *,
    budget_bpw: float,
    projected_bpw: float,
    survival: float,
    organs: Sequence[Mapping[str, Any]],
    ceiling_verdict: Mapping[str, Any],
) -> dict[str, Any]:
    """GRAVITY_DENSITY_FRONTIER fields for an evaluate result (single budget point)."""
    # Largest absolute payload contributors among selected (diagnostic only).
    ranked = sorted(
        organs,
        key=lambda r: -int(r.get("physical_payload_bytes") or 0),
    )[:8]
    contributors = [
        {
            "tensor_name": r["tensor_name"],
            "layer": r["layer"],
            "expert": r["expert"],
            "component": r["component"],
            "physical_payload_bytes": r["physical_payload_bytes"],
            "payload_delta_bytes": r["payload_delta_bytes"],
            "output_cosine": r["output_cosine"],
            "surplus_over_null": r["surplus_over_null"],
        }
        for r in ranked
    ]
    return {
        "schema": "hawking.GRAVITY_DENSITY_FRONTIER.v1",
        "GRAVITY_DENSITY_FRONTIER": {
            "lowest_tested_coherent_bpw": None,  # set by multi-point ladder
            "lowest_tested_executable_bpw": float(projected_bpw)
            if ceiling_verdict.get("legal")
            else None,
            "first_failed_lower_bpw": None,
            "capability_at_frontier": None,  # measured later; not claimed here
            "predicted_chain_survival": float(survival),
            "measured_generation_verdict": "UNEARNED_EVALUATE_ONLY",
            "largest_bpw_contributors": contributors,
            "largest_reclaimable_bit_buckets": [],
            "why_lower_candidate_failed": None,
            "next_representation_required": None,
        },
        "evaluate_budget_bpw": float(budget_bpw),
        "projected_complete_physical_bpw": float(projected_bpw),
        "one_bit_ceiling": dict(ceiling_verdict),
        "law": (
            "Seal the LOWEST VERIFIED EXECUTABLE FRONTIER, not the first "
            "passing artifact. 1.5 is an existential emergency ceiling, not a target."
        ),
    }


@dataclass
class ScoreIndex:
    """In-memory admitted organ metrics — load once, evaluate many budgets."""

    records: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    _sorted_records: list[dict[str, Any]] | None = field(default=None, repr=False)

    @property
    def n_organs(self) -> int:
        return len(self.records)

    def evaluate(
        self,
        *,
        budget_bpw: float,
        base_payload: int | None = None,
        elements: int | None = None,
        manifest_reserve: int | None = None,
        n_layers: int = MODEL_LAYERS,
    ) -> dict[str, Any]:
        """Pure CPU pack under budget. No I/O."""
        t0 = time.perf_counter()
        base = int(
            base_payload
            if base_payload is not None
            else self.meta.get("base_payload_bytes")
            or 0
        )
        elems = int(
            elements
            if elements is not None
            else self.meta.get("source_weight_elements")
            or 1
        )
        reserve = int(
            manifest_reserve
            if manifest_reserve is not None
            else self.meta.get("manifest_reserve_bytes")
            or 0
        )
        if base <= 0 or elems <= 0:
            raise ScoreIndexError(
                "evaluate requires base_payload_bytes and source_weight_elements "
                "(pass explicitly or embed in index meta)"
            )

        # Sort key cache: records are immutable after load; sort once, reuse.
        t_sort0 = time.perf_counter()
        if self._sorted_records is None:
            self._sorted_records = sorted(self.records, key=density_sort_key)
        scored = self._sorted_records
        t_sort = time.perf_counter() - t_sort0

        t_pack0 = time.perf_counter()
        organs, deferred, payload_running, projected = greedy_pack_under_ceiling(
            scored,
            budget_bpw=float(budget_bpw),
            base_payload=base,
            elements=elems,
            manifest_reserve=reserve,
            record_deferred=False,
        )
        t_pack = time.perf_counter() - t_pack0
        n_deferred_bpw = len(scored) - len(organs)

        t_atomic0 = time.perf_counter()
        organs, deferred, payload_running, atomic_stats = expert_atomic_enforce(
            organs, deferred, payload_running, record_deferred=False
        )
        # Recompute projected after demotions.
        projected = (payload_running + reserve) * 8.0 / max(elems, 1)
        t_atomic = time.perf_counter() - t_atomic0

        t_order0 = time.perf_counter()
        organs = sorted(
            organs, key=lambda r: (int(r["layer"]), int(r["expert"]), str(r["component"]))
        )
        t_order = time.perf_counter() - t_order0

        t_ceil0 = time.perf_counter()
        ceiling = one_bit_ceiling_verdict(
            projected_bpw=projected,
            payload_bytes=payload_running,
            elements=elems,
            manifest_reserve=reserve,
        )
        survival = predicted_chain_survival(organs, n_layers=n_layers)
        frontier = gravity_density_frontier_fields(
            budget_bpw=float(budget_bpw),
            projected_bpw=projected,
            survival=survival,
            organs=organs,
            ceiling_verdict=ceiling,
        )
        t_ceil = time.perf_counter() - t_ceil0

        # Selection identity for SEAL: organ set + budgets + payload shas.
        # Rows are already compact index records — reuse without rebuilding.
        selected_identity = list(organs)
        h = hashlib.sha256()
        for r in selected_identity:
            h.update(r["tensor_name"].encode("utf-8"))
            h.update(b"\0")
            h.update(str(r["physical_payload_sha256"]).encode("ascii"))
            h.update(b"\0")
            h.update(str(r.get("budget_label") or "").encode("utf-8"))
            h.update(b"\n")
        organ_set_sha = h.hexdigest()
        wall = time.perf_counter() - t0

        return {
            "schema": EVALUATE_SCHEMA,
            "status": "EVALUATED",
            "budget_bpw": float(budget_bpw),
            "complete_physical_bpw": float(projected),
            "predicted_chain_survival": float(survival),
            "ceiling_verdict": ceiling,
            "GRAVITY_DENSITY_FRONTIER": frontier,
            "selected_organs": selected_identity,
            "n_selected": len(selected_identity),
            "n_scored": len(self.records),
            "n_deferred": n_deferred_bpw
            + int(atomic_stats["organs_demoted_for_incomplete_triple"]),
            "deferred_organs": deferred[:200],
            "payload_bytes": int(payload_running),
            "base_payload_bytes": int(base),
            "manifest_reserve_bytes": int(reserve),
            "source_weight_elements": int(elems),
            "organ_set_sha256": organ_set_sha,
            "expert_atomic": atomic_stats,
            "timing_ms": {
                "sort": t_sort * 1000.0,
                "greedy_pack": t_pack * 1000.0,
                "expert_atomic": t_atomic * 1000.0,
                "order": t_order * 1000.0,
                "ceiling_and_frontier": t_ceil * 1000.0,
                "total": wall * 1000.0,
            },
            "index": {
                "path": str(self.path) if self.path else None,
                "n_organs": self.n_organs,
                "schema": self.meta.get("schema"),
                "capture_sha256": self.meta.get("capture_sha256"),
            },
        }


def build_index_document(
    records: Sequence[Mapping[str, Any]],
    *,
    meta: Mapping[str, Any] | None = None,
    min_surplus: float = 0.0,
) -> dict[str, Any]:
    admitted = filter_admitted(records, min_surplus=min_surplus)
    # Deterministic order for stable file hash (not the evaluate sort).
    admitted.sort(
        key=lambda r: (int(r["layer"]), int(r["expert"]), str(r["component"]), r["tensor_name"])
    )
    doc = {
        "schema": INDEX_SCHEMA,
        "n_organs": len(admitted),
        "min_surplus": float(min_surplus),
        "organs": admitted,
        **dict(meta or {}),
    }
    return doc


def save_score_index(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw + "\n", encoding="utf-8")
    tmp.replace(path)


def load_score_index(path: Path) -> ScoreIndex:
    path = Path(path)
    t0 = time.perf_counter()
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != INDEX_SCHEMA:
        raise ScoreIndexError(f"unexpected score index schema: {doc.get('schema')}")
    organs = [compact_organ_record(r) for r in doc["organs"]]
    meta = {k: v for k, v in doc.items() if k != "organs"}
    meta["load_ms"] = (time.perf_counter() - t0) * 1000.0
    return ScoreIndex(records=organs, meta=meta, path=path)


def index_from_selection_receipt(
    selection: Mapping[str, Any],
    *,
    baseline_ledger: Mapping[str, Any] | None = None,
    capture_sha256: str | None = None,
    min_surplus: float = 0.0,
    include_deferred: bool = False,
) -> dict[str, Any]:
    """Bootstrap a score index from a sealed selection receipt's organ rows.

    Selection receipts historically only retain *selected* organs. When the pack
    filled under a high ceiling, selected ≈ admitted and this is complete.
    """
    organs = list(selection.get("selected_representation", {}).get("organs") or [])
    if include_deferred:
        # Deferred rows in receipts are stubs (no full metrics) — skip.
        pass
    ledger = baseline_ledger or {}
    meta = {
        "source": "selection_receipt",
        "capture_sha256": capture_sha256
        or (selection.get("activation_capture") or {}).get("sha256"),
        "base_payload_bytes": int(
            ledger.get("tensor_payload_bytes")
            or selection.get("selection_method", {}).get("base_payload_bytes")
            or 0
        ),
        "source_weight_elements": int(
            ledger.get("source_weight_elements")
            or selection.get("selection_method", {}).get("source_weight_elements")
            or 0
        ),
        "manifest_reserve_bytes": int(
            ledger.get("manifest_bytes_billed")
            or selection.get("selection_method", {}).get("manifest_reserve_bytes")
            or 0
        ),
        "selection_method": selection.get("selection_method"),
    }
    # If ledger missing, recover base from first organ's projected path:
    if meta["base_payload_bytes"] <= 0 and organs:
        # base = physical - sum(delta) for selected-at-full is not safe; leave 0
        # and require caller to pass ledger.
        pass
    return build_index_document(organs, meta=meta, min_surplus=min_surplus)


def write_evaluate_receipt(path: Path, result: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Slim on-disk form: identity + timing, not full organ rows if huge — keep
    # full rows so SEAL can consume evaluate output without re-running.
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def selection_identity_from_evaluate(result: Mapping[str, Any]) -> dict[str, Any]:
    """Field-for-field comparison surface: organ set, budgets, payload shas."""
    organs = result.get("selected_organs") or []
    return {
        "budget_bpw": result.get("budget_bpw"),
        "complete_physical_bpw": result.get("complete_physical_bpw"),
        "n_selected": result.get("n_selected"),
        "organ_set_sha256": result.get("organ_set_sha256"),
        "organs": [
            {
                "tensor_name": r["tensor_name"],
                "budget_label": r.get("budget_label"),
                "rank": r.get("rank"),
                "bits": r.get("bits"),
                "physical_payload_sha256": r["physical_payload_sha256"],
                "physical_payload_bytes": r["physical_payload_bytes"],
            }
            for r in organs
        ],
    }


def selection_identity_from_sealed_manifest(
    manifest: Mapping[str, Any],
    *,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the same identity surface from a sealed candidate."""
    if selection is not None:
        organs = selection.get("selected_representation", {}).get("organs") or []
    else:
        # Fall back to changed organs in quality_repack_branch if present.
        branch = manifest.get("quality_repack_branch") or manifest.get(
            "activation_weighted_svd_branch"
        ) or {}
        names = set(branch.get("changed_organs") or [])
        organs = []
        for row in manifest.get("tensors") or []:
            if row.get("tensor_name") in names:
                organs.append(
                    {
                        "tensor_name": row["tensor_name"],
                        "budget_label": (row.get("layout") or {}).get("budget_label"),
                        "rank": (row.get("layout") or {}).get("rank"),
                        "bits": (row.get("layout") or {}).get("factor_bits"),
                        "physical_payload_sha256": row.get("artifact_sha256"),
                        "physical_payload_bytes": row.get("artifact_bytes"),
                    }
                )
    organs_sorted = sorted(organs, key=lambda r: r["tensor_name"])
    organ_set_sha = hashlib.sha256(
        json.dumps(
            [
                (
                    r["tensor_name"],
                    r.get("physical_payload_sha256") or r.get("artifact_sha256"),
                    r.get("budget_label"),
                )
                for r in organs_sorted
            ],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    ledger = manifest.get("complete_physical_bpw_ledger") or {}
    return {
        "budget_bpw": None,
        "complete_physical_bpw": ledger.get("complete_physical_bpw"),
        "n_selected": len(organs_sorted),
        "organ_set_sha256": organ_set_sha,
        "organs": [
            {
                "tensor_name": r["tensor_name"],
                "budget_label": r.get("budget_label"),
                "rank": r.get("rank"),
                "bits": r.get("bits") if r.get("bits") is not None else r.get("factor_bits"),
                "physical_payload_sha256": r.get("physical_payload_sha256")
                or r.get("artifact_sha256"),
                "physical_payload_bytes": r.get("physical_payload_bytes")
                or r.get("artifact_bytes"),
            }
            for r in organs_sorted
        ],
    }


def compare_evaluate_to_seal(
    evaluate_result: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Field-for-field proof that SEAL wrote exactly what EVALUATE predicted."""
    ev = selection_identity_from_evaluate(evaluate_result)
    se = sealed_identity
    ev_map = {r["tensor_name"]: r for r in ev["organs"]}
    se_map = {r["tensor_name"]: r for r in se["organs"]}
    only_ev = sorted(set(ev_map) - set(se_map))
    only_se = sorted(set(se_map) - set(ev_map))
    mismatches: list[dict[str, Any]] = []
    for name in sorted(set(ev_map) & set(se_map)):
        a, b = ev_map[name], se_map[name]
        for field in (
            "physical_payload_sha256",
            "physical_payload_bytes",
            "budget_label",
            "rank",
            "bits",
        ):
            av, bv = a.get(field), b.get(field)
            if av != bv and not (
                field in {"rank", "bits"} and av is None or bv is None and av == bv
            ):
                # Allow type coercion for rank/bits int/float.
                try:
                    if field in {"rank", "bits", "physical_payload_bytes"} and int(av) == int(
                        bv  # type: ignore[arg-type]
                    ):
                        continue
                except (TypeError, ValueError):
                    pass
                mismatches.append(
                    {"tensor_name": name, "field": field, "evaluate": av, "sealed": bv}
                )
    ok = not only_ev and not only_se and not mismatches
    return {
        "match": ok,
        "n_evaluate": ev["n_selected"],
        "n_sealed": se["n_selected"],
        "organ_set_sha_evaluate": ev["organ_set_sha256"],
        "organ_set_sha_sealed": se["organ_set_sha256"],
        "only_in_evaluate": only_ev[:20],
        "only_in_sealed": only_se[:20],
        "field_mismatches": mismatches[:50],
        "n_field_mismatches": len(mismatches),
    }
