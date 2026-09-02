"""HCLI's decision surface for "what does ModelLake acquire next, and why."

``acquire-next`` (tools/odyssey_ctl.py, wrapped read-only by hcli/odyssey.py)
picks list-order: the lowest-numbered ladder patient that is not on-disk, not
RETIRED, not ACQUIRING, not BLOCKED. On this box, right now, that is O010 --
it would propose GLM-4.5-Air, 205.76 GiB -- with no reasoning about whether
that is the best use of a disk and a network slot, and no idea that
GLM-4.5-Air (and five more of the fourteen ladder patients: O001, O003, O005,
O006, O007, O009, plus the 1.4 TiB O013 Kimi-K3) already sit as SEALED
specimens under /Volumes/corpdrive/hawking-modellake/specimens/, acquired
through the separate ModelLake watcher pipeline (tools/odyssey/modellake_*).
Left alone, list-order would have started a second, fully redundant download
of an already-complete payload into a *third* location
(~/.cache/huggingface/hub) -- the exact defect class this module exists to
close: acquisition without reconciling bytes already on disk.

Worse, the ladder's own bookkeeping cannot be trusted either. Right now
ODYSSEY_STATE.json still marks O000/O001/O002/O004/O005/O007/O008/O009 as
ACQUIRING against download pids from 2026-08-20 that are long dead, and the
HF cache directories those downloads once populated are simply gone --
~/.cache/huggingface is not protected from eviction. So this module never
trusts a stored ``state`` or ``ledger`` flag by itself: every "already have
it" verdict below is a live filesystem check, this call, against the actual
destination.

This module never downloads anything and never writes ODYSSEY_STATE.json,
HCLI_LEDGER.json, or partial/ -- it only reads three places an acquisition
can already exist (the ladder's own HF cache, the ModelLake watcher's
specimens/ and partial/, and any `hf download` process running right now)
and returns a ranked, reasoned ``propose()`` result for a confirm-gated
caller to act on. Starting a download stays that caller's decision -- e.g.
``hcli.odyssey.acquire_next(confirm=True)`` -- exactly as costly a step as
it already was.

Every number below comes from an existing producer, never invented here:
odyssey value = tools.odyssey_ctl.value() over its own SEED_WORK acquisition
items; role = ODYSSEY_MANIFEST.json's arch_objective/search_class/
info_budget (what question a specimen answers); disk headroom =
destination_disk_stat(), bound to the actual `hf download` destination, plus
the same DISK_RUN_GIB safety floor acquire_next() itself refuses under;
curriculum coverage = frontier_counts(). Direct function imports, not the
subprocess/text-parsing wrapper hcli/odyssey.py uses elsewhere in this
package, because ranking needs the structured intermediate values (bytes,
manifest fields), not a printed table.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.odyssey_ctl import (  # noqa: E402
    DISK_RUN_GIB,
    destination_disk_stat,
    ensure_state,
    frontier_counts,
    hf_cache_snapshot,
    overlay_manifest,
    patient_est_gib,
    patient_on_disk,
    pick_acquire_candidate,
    value as work_value,
)
from tools.odyssey import modellake_promote  # noqa: E402
from tools.odyssey.modellake_watch import process_rows, slug  # noqa: E402

SCHEMA = "hcli.acquisition.propose.v1"
HF_DOWNLOAD_NEEDLE = "hf download"  # same needle modellake_watch.matching_pids() uses


def _offline_hf_info_fn(repo: str) -> tuple[bool, Optional[dict], str]:
    """Never probe HF over the network from a read-only decision surface.

    A live gate-check belongs to acquire_next()'s own confirm-gated flow;
    this only needs pick_acquire_candidate's list-order pick for comparison.
    """
    return False, None, "not probed (acquisition.propose is offline)"


def _tag_expected_bytes(tag: str) -> int:
    """Total expected bytes for a ModelLake partial tag, read straight from
    its own resolved manifest -- the same file modellake_promote.verify()
    checks per-file sizes against. Never guessed from a directory name."""
    path = modellake_promote.MANIFEST_DIR / f"{tag}.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return sum(int(v) for v in (doc.get("sizes") or {}).values())
    except (OSError, ValueError, TypeError):
        return 0


def _ladder_verified_on_disk(repo: str) -> bool:
    """Live HF-cache truth for one repo, ignoring any stored on_disk/ledger
    flag. The same check finalize_acquisitions() uses to notice a landed
    download -- reused here so a stale flag can never stand in for bytes."""
    if not repo:
        return False
    return hf_cache_snapshot(repo) is not None


def _modellake_status(repo: str, revision: Optional[str]) -> dict[str, Any]:
    """Sealed / partial / absent, from ModelLake's own bytes-verified survey.

    survey() re-stats every file against the resolved manifest each call --
    it is the same reconciliation tools/odyssey/modellake_watch.py's
    reconcile() runs, just read here instead of re-derived.
    """
    if not repo or not revision:
        return {"tag": None, "sealed": False, "partial": None}
    tag = slug(repo, revision)
    sealed = (modellake_promote.SPECIMEN_ROOT / tag).is_dir()
    partial = None
    if not sealed:
        for row in modellake_promote.survey():
            if row["tag"] != tag or row["complete"]:
                continue
            on_disk = int(row["detail"].get("bytes") or 0)
            expected = _tag_expected_bytes(tag)
            partial = {
                "tag": tag,
                "bytes_on_disk": on_disk,
                "expected_bytes": expected,
                "pct": round(100 * on_disk / expected, 1) if expected else None,
                "note": (
                    "an incomplete payload for this exact repo+revision already "
                    "sits under ModelLake partial/; finishing it there outweighs "
                    "starting a separate ladder download"
                ),
            }
            break
    return {"tag": tag, "sealed": sealed, "partial": partial}


def propose(
    state: Optional[dict] = None,
    *,
    process_rows_fn: Optional[Callable[[], list]] = None,
) -> dict:
    """Rank the ladder's queued acquisitions HCLI could act on next.

    Returns a proposal for a confirm-gated caller -- never starts anything.
    ``process_rows_fn`` is a live-process-table override for tests only
    (default: a real `ps` scan, via tools.odyssey.modellake_watch).
    """
    state = state if state is not None else ensure_state()
    patients = {p["oxx"]: p for p in (state.get("patients") or []) if p.get("oxx")}
    proc_rows = (process_rows_fn or process_rows)()
    hf_rows = [(pid, cmd) for pid, cmd in proc_rows if HF_DOWNLOAD_NEEDLE in cmd]
    disk = destination_disk_stat()
    curriculum = frontier_counts()
    survey_rows = modellake_promote.survey()
    partials_in_progress = [
        {
            "tag": row["tag"],
            "bytes_on_disk": row["detail"].get("bytes"),
            "files": row["detail"].get("files"),
            "missing": row["detail"].get("missing"),
        }
        for row in survey_rows
        if not row["complete"]
    ]

    # list-order's own pick, for comparison -- on an isolated copy so its
    # in-place manifest-overlay writes (unconditional even with mutate=False)
    # never touch the caller's state.
    list_order_cand, _ = pick_acquire_candidate(
        copy.deepcopy(state), hf_info_fn=_offline_hf_info_fn, mutate=False
    )
    list_order_pick = (
        {"oxx": list_order_cand["oxx"], "repo": list_order_cand.get("canonical_source")
         or list_order_cand.get("source")}
        if list_order_cand
        else None
    )

    already_acquired: list[dict] = []
    blocked: list[dict] = []
    ranked: list[dict] = []

    for w in state.get("work") or []:
        if w.get("kind") != "acquisition" or w.get("status") != "READY":
            continue
        oxx = w.get("oxx")
        patient = patients.get(oxx)
        if patient is None:
            continue
        if patient.get("state") == "RETIRED":
            blocked.append({"oxx": oxx, "reason": "RETIRED"})
            continue

        man = overlay_manifest(patient, oxx)
        repo = man.get("canonical_source") or patient.get("canonical_source") or patient.get("source") or ""
        revision = man.get("canonical_revision")

        ladder_verified = _ladder_verified_on_disk(repo)
        claimed_on_disk = patient_on_disk(patient)
        stale_claim = None
        if claimed_on_disk and not ladder_verified:
            stale_claim = "state/ledger claims on-disk but no verified bytes in the HF cache right now"
        elif ladder_verified and not claimed_on_disk:
            stale_claim = "verified bytes on disk right now but ladder state does not reflect it"

        lake = _modellake_status(repo, revision)
        if lake["sealed"] or ladder_verified:
            already_acquired.append({
                "oxx": oxx,
                "model": patient.get("model"),
                "repo": repo,
                "where": "modellake specimens" if lake["sealed"] else "hf cache (ladder)",
                "tag": lake["tag"],
                "stale_claim": stale_claim,
                "_evidence": "MEASURED (bytes on disk)",
            })
            continue

        est_gib = patient_est_gib(oxx, patient, None)
        need_gib = est_gib + DISK_RUN_GIB
        fits_disk = disk["free_gib"] >= need_gib
        repo_in_flight = bool(repo) and any(repo in cmd for _, cmd in hf_rows)

        ranked.append({
            "oxx": oxx,
            "model": patient.get("model"),
            "class": patient.get("class"),
            "repo": repo,
            "canonical_revision": revision,
            "odyssey_value": round(work_value(w), 3),
            "role": {
                "arch_objective": man.get("arch_objective"),
                "search_class": man.get("search_class"),
                "info_budget": man.get("info_budget"),
                "reference_sibling": man.get("reference_sibling"),
                "notes": man.get("notes"),
            },
            "est_gib": round(est_gib, 2),
            "need_gib": round(need_gib, 2),
            "fits_disk": fits_disk,
            "partial_elsewhere": lake["partial"],
            "repo_in_flight": repo_in_flight,
            "stale_claim": stale_claim,
            "_evidence": "HYPOTHESIS (acquire plan) + MEASURED (disk, hf cache, modellake survey)",
        })

    # Finishing a same-repo partial beats starting fresh; of what's left,
    # prefer whatever fits the real disk headroom; break ties by Odyssey's
    # own info-value proxy. No new number is invented -- this only orders
    # signals odyssey_ctl.py and modellake_promote.py already compute.
    ranked.sort(key=lambda r: (
        0 if r["partial_elsewhere"] else 1,
        0 if r["fits_disk"] else 1,
        -r["odyssey_value"],
    ))

    recommended = ranked[0] if ranked else None
    list_order_would_redownload_sealed = bool(
        list_order_pick and any(a["oxx"] == list_order_pick["oxx"] for a in already_acquired)
    )
    if recommended is None:
        reason = (
            f"no eligible acquisition candidate ({len(already_acquired)} already acquired, "
            f"{len(blocked)} retired)"
        )
    elif recommended["partial_elsewhere"]:
        reason = (
            f"{recommended['oxx']} already has an incomplete payload at "
            f"{recommended['partial_elsewhere']['pct']}% under ModelLake partial/ "
            f"({recommended['partial_elsewhere']['tag']}); finishing that beats starting fresh"
        )
    elif not recommended["fits_disk"]:
        reason = (
            f"{recommended['oxx']} ranks highest by Odyssey value ({recommended['odyssey_value']}) "
            f"but needs {recommended['need_gib']} GiB against {disk['free_gib']} GiB free on "
            f"{disk['mount']}; nothing else ranked both eligible and fits"
        )
    else:
        reason = (
            f"{recommended['oxx']} ({recommended['model']}) ranks highest by Odyssey info-value "
            f"({recommended['odyssey_value']}) among {len(ranked)} eligible candidate(s), answers "
            f"'{recommended['role']['arch_objective']}', and fits headroom "
            f"({recommended['need_gib']} GiB needed vs {disk['free_gib']} GiB free on {disk['mount']})"
        )

    return {
        "schema": SCHEMA,
        "disk": disk,
        "disk_safety_headroom_gib": DISK_RUN_GIB,
        "network_slots_in_flight": {
            "count": len(hf_rows),
            "commands": [cmd[:200] for _, cmd in hf_rows],
        },
        "curriculum": curriculum,
        "partials_in_progress": partials_in_progress,
        "already_acquired": already_acquired,
        "blocked": blocked,
        "ranked": ranked,
        "recommended": recommended,
        "recommendation_reason": reason,
        "list_order_pick": list_order_pick,
        "list_order_would_redownload_sealed": list_order_would_redownload_sealed,
    }


if __name__ == "__main__":
    print(json.dumps(propose(), indent=2, default=str))
