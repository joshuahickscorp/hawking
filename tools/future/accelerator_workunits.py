"""ACCELERATOR_WORKUNITS — schedule Codex's profile→cut→AB→reprofile loop as HCLI species.

codex_behaviors.py already froze the thirty species and the GPU-host DAG.
This module does not redefine them. It is the sidecar scheduler: emit each
real loop stage as an HCLI unit, force every GPU-authority species SLEEPING
(this partition has no GPU and never will), refuse a species whose input
receipt is absent (named, never a default), refuse PROTECTED_AB as runnable
under LIGHT or worse contamination, and answer next_species() from the live
qualification queue.

A species that pretends it can run here is worse than an absent one.
next_species() never returns an empty list: when nothing is runnable it
returns a reason.

    python3 tools/future/accelerator_workunits.py --build
    python3 tools/future/accelerator_workunits.py --next
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, git, write_receipt
from tools.future import candidate_planner as cp
from tools.future import codex_behaviors as cb
from tools.future import contamination as C
from tools.future import protected_window as pw
from tools.future import qwen27_profile_schema as qps
from tools.future import workunit_species as ws

RECEIPT = "ACCELERATOR_WORKUNITS.json"
SCHEMA = "hawking.future.accelerator_workunits.v1"
RECORDED_BY = "tools/future/accelerator_workunits.py"

HANDOFF_REL = cb.HANDOFF_REL
QUEUE_REL = cp.QUEUE_REL.as_posix()
SCHEMA_REL = f"receipts/future/{qps.RECEIPT}"
STAGED_REL = f"receipts/future/{cp.RECEIPT}"
PREFLIGHT_REL = "receipts/future/STATIC_KERNEL_PREFLIGHT.json"
SCOREBOARD_REL = "receipts/headless/ACCELERATOR_SCOREBOARD.json"
LAW_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"
SCAR_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
ATTACK_REL = "receipts/future/ODYSSEY3_ADVERSARY.json"

# Closed loop the resident walks. Science for each id lives in codex_behaviors.
REQUIRED_SPECIES: tuple[str, ...] = (
    "PROFILE_COMPLETE_TOKEN",
    "PROFILE_REGION",
    "PROFILE_HOST_CEREMONY",
    "PROFILE_ACTIVE_BYTES",
    "PROFILE_DISPATCH",
    "PROFILE_SYNC",
    "FIND_TALLEST_COST",
    "GENERATE_KERNEL_CANDIDATE",
    "GENERATE_FUSION_CANDIDATE",
    "GENERATE_LAYOUT_CANDIDATE",
    "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
    "STATIC_KERNEL_VERIFY",
    "HOST_SHADER_ABI_VERIFY",
    "STRUCTURAL_COST_COMPARE",
    "DIAGNOSTIC_AB",
    "PROTECTED_AB",
    "REPROFILE_AFTER_WIN",
    "UPDATE_SCOREBOARD",
    "UPDATE_LAW",
    "UPDATE_SCAR",
    "TRANSFER_LAW",
    "ATTACK_LAW",
)

LOOP_WAVES: tuple[tuple[str, ...], ...] = (
    (
        "PROFILE_COMPLETE_TOKEN",
        "PROFILE_REGION",
        "PROFILE_HOST_CEREMONY",
        "PROFILE_ACTIVE_BYTES",
        "PROFILE_DISPATCH",
        "PROFILE_SYNC",
    ),
    ("FIND_TALLEST_COST",),
    (
        "GENERATE_KERNEL_CANDIDATE",
        "GENERATE_FUSION_CANDIDATE",
        "GENERATE_LAYOUT_CANDIDATE",
        "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
    ),
    ("STATIC_KERNEL_VERIFY", "HOST_SHADER_ABI_VERIFY", "STRUCTURAL_COST_COMPARE"),
    ("DIAGNOSTIC_AB",),
    ("PROTECTED_AB",),
    ("REPROFILE_AFTER_WIN",),
    ("UPDATE_SCOREBOARD", "UPDATE_LAW", "UPDATE_SCAR"),
    ("TRANSFER_LAW",),
    ("ATTACK_LAW",),
)

GPU_LANES = frozenset({"metal_gpu", "metal_compiler", "protected_lease", "diagnostic_ab", "flash_nx"})
GPU_RESOURCE = frozenset({"GPU_EXCLUSIVE", "GPU_DIRTY_OK", "GPU_DECODE"})
BLOCKED_CONTAMINATION = frozenset({"LIGHT", "HEAVY", "UNKNOWN"})
WIN_STATUSES = frozenset({"PROTECTED_PASS", "INTEGRATED"})

INPUT_RECEIPTS: dict[str, tuple[str, ...]] = {
    "PROFILE_COMPLETE_TOKEN": (HANDOFF_REL,),
    "PROFILE_REGION": (HANDOFF_REL,),
    "PROFILE_HOST_CEREMONY": (HANDOFF_REL,),
    "PROFILE_ACTIVE_BYTES": (HANDOFF_REL,),
    "PROFILE_DISPATCH": (HANDOFF_REL,),
    "PROFILE_SYNC": (HANDOFF_REL,),
    "FIND_TALLEST_COST": (HANDOFF_REL,),
    "GENERATE_KERNEL_CANDIDATE": (QUEUE_REL,),
    "GENERATE_FUSION_CANDIDATE": (QUEUE_REL,),
    "GENERATE_LAYOUT_CANDIDATE": (QUEUE_REL,),
    "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE": (QUEUE_REL,),
    "STATIC_KERNEL_VERIFY": (STAGED_REL,),
    "HOST_SHADER_ABI_VERIFY": (PREFLIGHT_REL,),
    "STRUCTURAL_COST_COMPARE": (HANDOFF_REL, STAGED_REL),
    "DIAGNOSTIC_AB": (QUEUE_REL,),
    "PROTECTED_AB": (QUEUE_REL,),
    "REPROFILE_AFTER_WIN": (QUEUE_REL,),
    "UPDATE_SCOREBOARD": (SCOREBOARD_REL,),
    "UPDATE_LAW": (LAW_REL,),
    "UPDATE_SCAR": (SCAR_REL,),
    "TRANSFER_LAW": (LAW_REL,),
    "ATTACK_LAW": (ATTACK_REL,),
}

OUTPUT_RECEIPTS: dict[str, str] = {
    "STATIC_KERNEL_VERIFY": PREFLIGHT_REL,
    "HOST_SHADER_ABI_VERIFY": "receipts/future/CLAUDE_SIDECAR_ABI_ADJUDICATION.json",
    "UPDATE_LAW": LAW_REL,
    "UPDATE_SCAR": SCAR_REL,
    "TRANSFER_LAW": LAW_REL,
    "ATTACK_LAW": ATTACK_REL,
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. GPU-authority species "
    "are emitted SLEEPING; this partition cannot take a lease or write a "
    "protected result. A missing input is a named refusal, never a pass."
)

SIDECAR_WAKE = (
    "SLEEPING until a distinct HCLI GPU_PROTECTED lane (not this sidecar) holds "
    "a proven lease on a Metal-capable GPU with the Metal compiler present. "
    "This sidecar partition has no GPU authority and never will. "
    f"protected_window.{pw.acquire_lease.__name__} raises rather than flock. "
    "A synthetic complete-token result is refused."
)


class InputRefused(ValueError):
    """A species was asked to emit without a named input receipt."""


class AcceleratorWorkunitError(ValueError):
    """Scheduler contract violation (GPU unit emitted runnable, empty next, …)."""


# ---------------------------------------------------------------------------
# Visibility. Missing in this sparse tree is not campaign-absence.
# ---------------------------------------------------------------------------


def _roots() -> list[Path]:
    return ws._checkout_roots()


def receipt_visible(rel: str, injected: Mapping[str, bool] | None = None) -> bool:
    """Disk/git presence, with an injected override so tests can watch a refusal."""
    if injected is not None and rel in injected:
        return bool(injected[rel])
    for root in _roots():
        path = root / rel
        if path.is_file():
            return True
    blob = git("show", f"HEAD:{rel}")
    return bool(blob)


def missing_inputs(species_id: str, injected: Mapping[str, bool] | None = None) -> list[str]:
    return [rel for rel in input_receipts_for(species_id) if not receipt_visible(rel, injected)]


def input_receipts_for(species_id: str) -> tuple[str, ...]:
    if species_id in INPUT_RECEIPTS:
        return INPUT_RECEIPTS[species_id]
    if species_id in cb.SPECIES_IDS:
        return (HANDOFF_REL,)
    raise InputRefused(f"{species_id} refuses: unknown species")


def output_receipt_for(species_id: str) -> str:
    return OUTPUT_RECEIPTS.get(species_id, f"receipts/future/{RECEIPT}")


# ---------------------------------------------------------------------------
# Catalog overlay. Science stays in codex_behaviors; we add the scheduler contract.
# ---------------------------------------------------------------------------


def needs_gpu_authority(spec: Mapping[str, Any]) -> bool:
    lane = str((spec.get("resources") or {}).get("lane") or spec.get("resource_lane") or "")
    rc = str(spec.get("resource_class") or "")
    return lane in GPU_LANES or rc in GPU_RESOURCE


def _fails_closed(species_id: str, *, gpu: bool) -> str:
    named = ", ".join(input_receipts_for(species_id))
    if species_id == "PROTECTED_AB":
        return (
            f"refuses if {QUEUE_REL} is absent (named); never pending under "
            "LIGHT/HEAVY/UNKNOWN contamination; never pending on this sidecar "
            f"even if QUIESCENT; {pw.acquire_lease.__name__} raises rather than flock"
        )
    if gpu:
        return (
            f"refuses if an input receipt is absent (named: {named}); "
            "emitted SLEEPING with a wake condition, never pending, never FAILED"
        )
    return (
        f"refuses if an input receipt is absent (named: {named}); "
        "does not invent a hardware number; does not promote"
    )


def contracts(*, handoff: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Scheduler contract per required species, grounded in the existing catalog."""
    doc = handoff
    if doc is None:
        doc, _src = cb.load_handoff()
    if doc is None:
        raise InputRefused(
            f"REQUIRED_SPECIES refuse: missing receipt {HANDOFF_REL}"
        )
    cat = cb.catalog_by_id(handoff=doc)
    missing = [sid for sid in REQUIRED_SPECIES if sid not in cat]
    if missing:
        raise AcceleratorWorkunitError(
            f"codex_behaviors catalog is missing required species {missing}"
        )
    out: dict[str, dict[str, Any]] = {}
    for sid in REQUIRED_SPECIES:
        spec = cat[sid]
        gpu = needs_gpu_authority(spec)
        lane = str((spec.get("resources") or {}).get("lane") or "static")
        out[sid] = {
            "id": sid,
            "title": spec.get("title"),
            "lane": lane,
            "resource_class": spec.get("resource_class"),
            "gpu_authority_required": gpu,
            "gpu_authority": False,
            "input_receipts": list(input_receipts_for(sid)),
            "output_receipt": output_receipt_for(sid),
            "verifier": spec.get("verifier"),
            "fails_closed": _fails_closed(sid, gpu=gpu),
            "profile_columns": list(qps.REQUIRED_METRICS) if sid.startswith("PROFILE_") or sid == "FIND_TALLEST_COST" else [],
            "extends": spec.get("extends_parent"),
            "loose_parent": spec.get("loose_parent"),
            "evidence_class": "STATIC_ONLY",
        }
    return out


# ---------------------------------------------------------------------------
# Emit. GPU species are SLEEPING on this sidecar, full stop.
# ---------------------------------------------------------------------------


def sidecar_wake(
    spec: Mapping[str, Any],
    *,
    contamination_class: str,
    blockers: Sequence[str],
) -> str:
    lane = str((spec.get("resources") or {}).get("lane") or "static")
    parts = [SIDECAR_WAKE]
    if blockers:
        parts.append(cb.wake_condition_for(lane, blockers))
    if str(spec.get("id")) == "PROTECTED_AB" and contamination_class in BLOCKED_CONTAMINATION:
        parts.append(
            f"PROTECTED_AB is not runnable under contamination_class={contamination_class} "
            "(LIGHT or worse); wake also requires QUIESCENT."
        )
    return " ".join(parts)


def _contamination_class(injected: str | None) -> tuple[str, str]:
    if injected is not None:
        klass = str(injected).strip().upper()
        if klass not in C.CONTAMINATION_CLASSES:
            raise InputRefused(
                f"PROTECTED_AB refuses: unknown contamination_class {injected!r}"
            )
        return klass, "injected"
    try:
        row = C.classify_contamination(C.snapshot())
        return str(row["contamination_class"]), str(row["contamination_reason"])
    except Exception as exc:
        return (
            "UNKNOWN",
            f"contamination snapshot failed ({type(exc).__name__}: {exc}); "
            "UNKNOWN never QUIESCENT",
        )


def emit_species(
    species_id: str,
    *,
    handoff: Mapping[str, Any] | None = None,
    receipts_visible: Mapping[str, bool] | None = None,
    contamination_class: str | None = None,
    blockers: Sequence[str] | None = None,
    cycle: int = 0,
) -> dict[str, Any]:
    """Emit one loop species as an HCLI unit. GPU authority → SLEEPING. Missing input → raise."""
    sid = str(species_id)
    if sid not in REQUIRED_SPECIES and sid not in cb.SPECIES_IDS:
        raise InputRefused(f"{sid} refuses: unknown species")

    doc = handoff
    src = "argument"
    if doc is None:
        doc, src = cb.load_handoff()
    if doc is None:
        raise InputRefused(f"{sid} refuses: missing receipt {HANDOFF_REL}")

    missing = missing_inputs(sid, receipts_visible)
    if missing:
        raise InputRefused(f"{sid} refuses: missing receipt {missing[0]}")

    klass, klass_src = _contamination_class(contamination_class)
    cat = cb.catalog_by_id(handoff=doc)
    if sid not in cat:
        raise InputRefused(f"{sid} refuses: unknown species")
    spec = cat[sid]
    gpu = needs_gpu_authority(spec)
    observed = list(blockers) if blockers is not None else cb.blockers_from_handoff(doc)

    unit = cb.emit_species_unit(
        spec,
        cycle=int(cycle),
        dependencies=[],
        blockers=observed,
    )
    # Sidecar host constraint: GPU species NEVER pending here, even if the
    # handoff blocker list is empty. codex_behaviors derives sleep from
    # blockers; this module derives it from the partition having no GPU.
    unit["gpu_authority"] = False
    unit["gpu_authority_required"] = gpu
    unit["input_receipts"] = list(input_receipts_for(sid))
    unit["output_receipt"] = output_receipt_for(sid)
    unit["fails_closed"] = _fails_closed(sid, gpu=gpu)
    unit["contamination_class"] = klass
    unit["contamination_source"] = klass_src
    unit["handoff_loaded_from"] = src
    unit["profile_columns"] = list(qps.REQUIRED_METRICS) if sid.startswith("PROFILE_") or sid == "FIND_TALLEST_COST" else []
    unit["claim_boundary"] = CLAIM_BOUNDARY
    unit["evidence_class"] = "STATIC_ONLY"
    unit["measurement_class"] = "STATIC_ONLY"
    unit["bench_state"] = "UNKNOWN"

    if gpu:
        wake = sidecar_wake(spec, contamination_class=klass, blockers=observed)
        unit["status"] = cb.STATUS_SLEEPING
        unit["classification"] = cb.CLASS_SLEEPING
        unit["wake_condition"] = wake
        unit["blocked_reason"] = wake
        unit["runnable"] = False
        unit["requires_quiescence"] = True
    else:
        unit["runnable"] = unit.get("status") == "pending"
        unit.setdefault("wake_condition", None)

    if sid == "PROTECTED_AB":
        # LIGHT or worse cannot be runnable. QUIESCENT still cannot: no GPU.
        unit["status"] = cb.STATUS_SLEEPING
        unit["classification"] = cb.CLASS_SLEEPING
        unit["runnable"] = False
        if not unit.get("wake_condition"):
            unit["wake_condition"] = sidecar_wake(spec, contamination_class=klass, blockers=observed)
            unit["blocked_reason"] = unit["wake_condition"]

    if gpu and unit.get("status") != cb.STATUS_SLEEPING:
        raise AcceleratorWorkunitError(
            f"{sid} needs GPU authority but was emitted status={unit.get('status')!r}"
        )
    if unit.get("runnable") is True and gpu:
        raise AcceleratorWorkunitError(f"{sid} was emitted runnable; sidecar has no GPU")
    if str(unit.get("status") or "").lower() in {"failed", "skipped"}:
        raise AcceleratorWorkunitError(f"{sid} emitted {unit.get('status')}; must be SLEEPING or pending")
    ws.validate_emitted_unit(unit)
    return unit


def emit_loop(
    *,
    handoff: Mapping[str, Any] | None = None,
    receipts_visible: Mapping[str, bool] | None = None,
    contamination_class: str | None = None,
    blockers: Sequence[str] | None = None,
    cycle: int = 0,
) -> dict[str, Any]:
    """Emit every required loop species. Per-species input refusal is recorded, not rounded into a pass."""
    units: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for sid in REQUIRED_SPECIES:
        try:
            units.append(
                emit_species(
                    sid,
                    handoff=handoff,
                    receipts_visible=receipts_visible,
                    contamination_class=contamination_class,
                    blockers=blockers,
                    cycle=cycle,
                )
            )
        except InputRefused as exc:
            msg = str(exc)
            named = msg.split("missing receipt ", 1)[-1] if "missing receipt " in msg else None
            refusals.append(
                {
                    "species": sid,
                    "reason": msg,
                    "missing_receipt": named,
                    "runnable": False,
                }
            )
    gpu_runnable = [
        u["species"]
        for u in units
        if u.get("gpu_authority_required") and u.get("runnable") is True
    ]
    if gpu_runnable:
        raise AcceleratorWorkunitError(f"GPU species emitted runnable: {gpu_runnable}")
    return {
        "units": units,
        "refusals": refusals,
        "n_emitted": len(units),
        "n_refused": len(refusals),
        "n_sleeping": sum(1 for u in units if u.get("status") == cb.STATUS_SLEEPING),
        "n_pending": sum(1 for u in units if u.get("status") == "pending"),
        "n_gpu_sleeping": sum(
            1 for u in units if u.get("gpu_authority_required") and u.get("status") == cb.STATUS_SLEEPING
        ),
    }


# ---------------------------------------------------------------------------
# next_species — derived from the real candidate queue, never an empty list.
# ---------------------------------------------------------------------------


def load_queue_or_summary() -> tuple[dict[str, Any] | None, str]:
    """Full physical queue if visible; else the handoff's current_queue summary."""
    try:
        queue = cp.load_queue()
    except cp.QueueNotFoundError as exc:
        not_found = str(exc)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{QUEUE_REL}:{exc}"
    else:
        return queue, str(queue.get("_loaded_from") or QUEUE_REL)
    handoff, src = cb.load_handoff()
    summary = (handoff or {}).get("current_queue") if isinstance(handoff, dict) else None
    if isinstance(summary, dict):
        row = dict(summary)
        row["_loaded_from"] = f"{src}:current_queue"
        return row, row["_loaded_from"]
    return None, f"unseen_in_this_checkout after QueueNotFoundError: {not_found}"


def _ids_with_status(queue: Mapping[str, Any], status: str) -> list[str]:
    key = {
        "READY_PROTECTED": "ready_candidate_ids",
        "READY_DIAGNOSTIC": "ready_diagnostic_ids",
        "STATIC_ONLY": "static_only_candidate_ids",
        "BLOCKED": "blocked_candidate_ids",
    }.get(status)
    named = [str(x) for x in (queue.get(key) or [])] if key else []
    rows = queue.get("candidates")
    if isinstance(rows, list):
        from_rows = [
            cp.cid(r) for r in rows if str(r.get("status") or "") == status and r.get("candidate_id")
        ]
        return from_rows or named
    return named


def queue_census(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and identity sets. Empty is a first-class answer, not a missing key."""
    raw_counts = queue.get("status_counts")
    counts: dict[str, int] = {}
    if isinstance(raw_counts, Mapping):
        counts = {str(k): int(v) for k, v in raw_counts.items()}
    rows = queue.get("candidates")
    if isinstance(rows, list):
        derived: dict[str, int] = {}
        for row in rows:
            derived[str(row.get("status") or "")] = derived.get(str(row.get("status") or ""), 0) + 1
        if derived:
            counts = derived
        n = len(rows)
    elif "total_candidates" in queue:
        n = int(queue["total_candidates"] or 0)
    else:
        n = sum(counts.values())
    ready_p = _ids_with_status(queue, "READY_PROTECTED")
    ready_d = _ids_with_status(queue, "READY_DIAGNOSTIC")
    static = _ids_with_status(queue, "STATIC_ONLY")
    blocked = _ids_with_status(queue, "BLOCKED")
    wins = _ids_with_status(queue, "PROTECTED_PASS") + _ids_with_status(queue, "INTEGRATED")
    if not wins:
        wins = []
        if isinstance(rows, list):
            wins = [cp.cid(r) for r in rows if str(r.get("status") or "") in WIN_STATUSES]
    return {
        "n_candidates": int(n),
        "status_counts": dict(sorted(counts.items())),
        "ready_protected_ids": ready_p,
        "ready_diagnostic_ids": ready_d,
        "static_only_ids": static,
        "blocked_ids": blocked,
        "win_ids": wins,
        "n_ready_protected": len(ready_p) if ready_p else int(counts.get("READY_PROTECTED") or 0),
        "n_ready_diagnostic": len(ready_d) if ready_d else int(counts.get("READY_DIAGNOSTIC") or 0),
        "n_static_only": len(static) if static else int(counts.get("STATIC_ONLY") or 0),
        "n_blocked": len(blocked) if blocked else int(counts.get("BLOCKED") or 0),
        "n_wins": len(wins) if wins else int(counts.get("PROTECTED_PASS") or 0) + int(counts.get("INTEGRATED") or 0),
        "loaded_from": queue.get("_loaded_from"),
    }


def _generate_species_for(candidate_id: str) -> str:
    name = candidate_id.lower()
    if "fusion" in name:
        return "GENERATE_FUSION_CANDIDATE"
    if "pipeline" in name or "commit-timing" in name or "encoder-label" in name:
        return "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE"
    if "splitk" in name or "vecgroup" in name or name.endswith("-vec"):
        return "GENERATE_LAYOUT_CANDIDATE"
    return "GENERATE_KERNEL_CANDIDATE"


def _answer(
    *,
    species: str | None,
    runnable: bool,
    status: str,
    reason: str,
    census: Mapping[str, Any] | None,
    contamination_class: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(reason or "").strip():
        raise AcceleratorWorkunitError("next_species produced no reason")
    body: dict[str, Any] = {
        "species": species,
        "runnable": bool(runnable) and species is not None,
        "status": status,
        "reason": str(reason),
        "gpu_authority": False,
        "contamination_class": contamination_class,
        "queue": dict(census) if census else None,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if extra:
        body.update(dict(extra))
    return body


def next_species(
    queue: Mapping[str, Any] | None = None,
    *,
    contamination_class: str | None = None,
    receipts_visible: Mapping[str, bool] | None = None,
    handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What the resident would do next on the Accelerator right now.

    Always a dict with a reason. Never []. An empty queue is a named refusal
    to invent work, not a silent no-op.
    """
    klass, klass_src = _contamination_class(contamination_class)
    src = "argument"
    loaded = queue
    if loaded is None:
        loaded, src = load_queue_or_summary()
    if loaded is None:
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason=(
                f"nothing runnable: missing receipt {QUEUE_REL} "
                f"(looked at checkout roots and git HEAD; loaded_from={src})"
            ),
            census=None,
            contamination_class=klass,
            extra={"contamination_source": klass_src, "missing_receipt": QUEUE_REL},
        )
    if not isinstance(loaded, Mapping):
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason="nothing runnable: qualification queue is not a mapping",
            census=None,
            contamination_class=klass,
        )

    census = queue_census(loaded)
    census["loaded_from"] = loaded.get("_loaded_from") or src
    n = int(census["n_candidates"])
    if n <= 0:
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason=(
                "nothing runnable: qualification queue has 0 candidates; "
                "the Accelerator loop has no patient. An empty list of species "
                "would hide this."
            ),
            census=census,
            contamination_class=klass,
            extra={"contamination_source": klass_src},
        )

    n_win = int(census["n_wins"])
    n_rp = int(census["n_ready_protected"])
    n_rd = int(census["n_ready_diagnostic"])
    n_st = int(census["n_static_only"])
    n_bl = int(census["n_blocked"])

    def _sleeping(sid: str, reason: str) -> dict[str, Any]:
        missing = missing_inputs(sid, receipts_visible)
        if missing:
            return _answer(
                species=sid,
                runnable=False,
                status="refused",
                reason=f"{sid} refuses: missing receipt {missing[0]}",
                census=census,
                contamination_class=klass,
                extra={"missing_receipt": missing[0], "contamination_source": klass_src},
            )
        return _answer(
            species=sid,
            runnable=False,
            status=cb.STATUS_SLEEPING,
            reason=reason,
            census=census,
            contamination_class=klass,
            extra={
                "wake_condition": SIDECAR_WAKE,
                "gpu_authority_required": True,
                "contamination_source": klass_src,
            },
        )

    def _cpu(sid: str, reason: str) -> dict[str, Any]:
        missing = missing_inputs(sid, receipts_visible)
        if missing:
            return _answer(
                species=sid,
                runnable=False,
                status="refused",
                reason=f"{sid} refuses: missing receipt {missing[0]}",
                census=census,
                contamination_class=klass,
                extra={"missing_receipt": missing[0], "contamination_source": klass_src},
            )
        return _answer(
            species=sid,
            runnable=True,
            status="pending",
            reason=reason,
            census=census,
            contamination_class=klass,
            extra={"gpu_authority_required": False, "contamination_source": klass_src},
        )

    if n_win > 0:
        return _sleeping(
            "REPROFILE_AFTER_WIN",
            (
                f"{n_win} PROTECTED_PASS/INTEGRATED identit"
                f"{'y' if n_win == 1 else 'ies'} on the queue; the loop must "
                "reprofile the new incumbent. REPROFILE_AFTER_WIN needs GPU "
                "authority this sidecar does not have."
            ),
        )
    if n_rp > 0:
        why = (
            f"{n_rp} READY_PROTECTED candidate(s) wait on a protected complete-token "
            f"AB. PROTECTED_AB is the Accelerator next step and is not runnable "
            f"here (gpu_authority=false"
        )
        if klass in BLOCKED_CONTAMINATION:
            why += f", contamination_class={klass} is LIGHT or worse"
        why += ")."
        return _sleeping("PROTECTED_AB", why)
    if n_rd > 0:
        return _sleeping(
            "DIAGNOSTIC_AB",
            (
                f"{n_rd} READY_DIAGNOSTIC candidate(s) wait on a diagnostic AB. "
                "DIAGNOSTIC_AB needs a GPU window this sidecar does not have; "
                "a diagnostic pass would not promote."
            ),
        )
    if n_st > 0:
        ident = (census["static_only_ids"] or ["static-only"])[0]
        sid = _generate_species_for(str(ident))
        return _cpu(
            sid,
            (
                f"{n_st} STATIC_ONLY candidate(s) are not in a GPU rung; "
                f"next CPU species is {sid} for {ident}. Spec only; no timing claim."
            ),
        )
    if n_bl > 0 and n_bl == n:
        sample = (census["blocked_ids"] or ["(unnamed)"])[0]
        rows = loaded.get("candidates") if isinstance(loaded.get("candidates"), list) else []
        blocked_reason = None
        for row in rows or []:
            if str(row.get("candidate_id")) == str(sample):
                blocked_reason = row.get("blocked_reason")
                break
        extra = f" sample blocked_reason={blocked_reason!r}" if blocked_reason else ""
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason=(
                f"nothing runnable: all {n} candidates are BLOCKED "
                f"(sample={sample}).{extra} The sidecar will not invent a GPU "
                "cell for a candidate the queue itself refuses to run."
            ),
            census=census,
            contamination_class=klass,
            extra={"contamination_source": klass_src},
        )

    return _cpu(
        "FIND_TALLEST_COST",
        (
            f"queue has {n} candidate(s) and no READY_PROTECTED / READY_DIAGNOSTIC / "
            "STATIC_ONLY cursor; FIND_TALLEST_COST can rank AKB columns without "
            "inventing ns (UNKNOWN is a legal stop)."
        ),
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    handoff, handoff_src = cb.load_handoff()
    if handoff is None:
        raise InputRefused(
            f"build refuses: missing receipt {HANDOFF_REL} "
            "(the loop cannot be scheduled without the training trace)"
        )
    klass, klass_src = _contamination_class(None)
    loop = emit_loop(handoff=handoff, contamination_class=klass)
    nxt = next_species(handoff=handoff, contamination_class=klass)
    table = contracts(handoff=handoff)
    gpu_ids = [sid for sid, row in table.items() if row["gpu_authority_required"]]
    gpu_emitted = [u for u in loop["units"] if u.get("gpu_authority_required")]
    if any(u.get("runnable") is True for u in gpu_emitted):
        raise AcceleratorWorkunitError("GPU species emitted runnable")
    if any(u.get("status") != cb.STATUS_SLEEPING for u in gpu_emitted):
        raise AcceleratorWorkunitError("GPU species emitted not SLEEPING")

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Make Codex's profile → tallest cost → why → remove (information / "
            "bytes / FLOPs / intermediate / dispatch / sync / copy / ceremony) → "
            "A/B → integrate → reprofile loop schedulable by HCLI, so the next "
            "latency seam is found without a human asking for it."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "handoff": {
            "path": HANDOFF_REL,
            "loaded_from": handoff_src,
            "present": True,
        },
        "loop": {
            "waves": [list(w) for w in LOOP_WAVES],
            "required_species": list(REQUIRED_SPECIES),
            "rule": (
                "profile, rank the tallest denominator, ask why the cost exists, "
                "generate a candidate that eliminates work, verify statically, "
                "AB, ledger, transfer, attack; a PHYSICAL_WIN enqueues REPROFILE_AFTER_WIN. "
                "GPU stages sleep on this sidecar."
            ),
            "profile_columns": list(qps.REQUIRED_METRICS),
            "protected_window_stages": list(pw.WINDOW_STAGES),
        },
        "species": table,
        "emitted": {
            "n_emitted": loop["n_emitted"],
            "n_refused": loop["n_refused"],
            "n_sleeping": loop["n_sleeping"],
            "n_pending": loop["n_pending"],
            "n_gpu_sleeping": loop["n_gpu_sleeping"],
            "units": [
                {
                    "id": u["id"],
                    "species": u.get("species"),
                    "status": u.get("status"),
                    "runnable": u.get("runnable"),
                    "gpu_authority_required": u.get("gpu_authority_required"),
                    "lane": (u.get("resources") or {}).get("lane"),
                    "verifier": u.get("verifier"),
                    "input_receipts": u.get("input_receipts"),
                    "output_receipt": u.get("output_receipt"),
                    "wake_condition": u.get("wake_condition"),
                    "fails_closed": u.get("fails_closed"),
                    "contamination_class": u.get("contamination_class"),
                }
                for u in loop["units"]
            ],
            "refusals": loop["refusals"],
        },
        "next_species": nxt,
        "contamination": {
            "class": klass,
            "source": klass_src,
            "protected_ab_runnable": False,
            "rule": (
                "PROTECTED_AB is never runnable under LIGHT/HEAVY/UNKNOWN, and is "
                "never runnable on this sidecar even if QUIESCENT."
            ),
        },
        "gpu_species": gpu_ids,
        "recovered_implementation": [
            "tools/future/codex_behaviors.py — thirty grounded species, emit_species_unit, "
            "emit_cycle, wake_condition_for; this module does not fork the catalog",
            "tools/future/workunit_species.py — HCLI field set, define_species, emit_hcli_workunit",
            "tools/future/candidate_planner.py — live qualification queue + staged factorial plan",
            "tools/future/contamination.py — QUIESCENT/LIGHT/HEAVY/UNKNOWN; assert_promotable",
            "tools/future/protected_window.py — eviction envelope; acquire_lease raises",
            "tools/future/qwen27_profile_schema.py — REQUIRED_METRICS columns FIND_TALLEST ranks",
            f"{HANDOFF_REL} — training trace loaded from {handoff_src}",
        ],
        "gaps_closed": [
            "next_species() answers the Accelerator cursor from real queue state and "
            "returns a reason when nothing is runnable (never an empty list)",
            "GPU-authority species are emitted SLEEPING on this sidecar even if the "
            "handoff blocker list is empty (codex_behaviors derives sleep from blockers)",
            "a species whose input receipt is absent refuses, naming the receipt",
            "PROTECTED_AB is never emitted runnable under LIGHT or worse contamination",
        ],
        "negative_findings": [
            "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
            "READY_PROTECTED candidates wait on a GPU window this partition must not seize",
            f"contamination_class={klass} ({klass_src}); PROTECTED_AB stays SLEEPING",
            "codex_behaviors.UPDATE_SCOREBOARD and ATTACK_LAW remain UNGROUNDED_FROM_HANDOFF; "
            "this module still emits them as species with their input-receipt gate",
            "orchestration.py BINDINGS is not writable from this lane; the module is "
            "resident-callable via --build/--next regardless",
        ],
        "resident_callable": {
            "entry_point": "tools.future.accelerator_workunits.next_species()",
            "workunit": (
                "one CPU_ANALYSIS unit for next_species / emit_loop; GPU species "
                "are emitted SLEEPING and are not a work source on this host"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "InputRefused names the missing receipt; GPU units cannot be pending; "
                "next_species on an empty or absent queue returns a reason, not []"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--next", action="store_true")
    a = ap.parse_args()
    if a.next:
        print(json.dumps(next_species(), indent=1, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
