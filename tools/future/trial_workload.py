"""TRIAL WORKLOAD — compose mixed real frontier work so an autonomy trial cannot pass by idling.

A resident that survives six hours doing nothing useful FAILS. The frozen
autonomy trials must run current frontier work, mixed enough that a static
executor cannot fake the behaviours. This module is the composer: `mix`
declares the proportions and why each is there; `compose` selects the unit
set from live frontier items and live orchestration bindings.

It refuses a 3h/6h set with no replan pair (naming what is missing), a unit
not bound to a real frontier item, a duplicate work identity (unique ids do
not make work distinct), and an unknown trial id. Units that would require
GPU authority are emitted SLEEPING, never as a synthetic completion.

This does not invoke the units, does not take a GPU lease, and does not
measure hardware. Declared capability is not executed capability: compose
selects and binds; the autonomy driver is what runs. A cancelled trial is
the test of the mix — every unit has to be work that would have been worth
doing anyway. Padding is refused, not scheduled.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import REPO, write_receipt
from tools.future import adaptive_verification as av
from tools.future import autonomy_trial as at
from tools.future import frontiers as fr
from tools.future import odyssey_launch as ol
from tools.future import orchestration as orch
from tools.future import phase_listeners as pl
from tools.future import specimen_verify as sv

RECEIPT = "TRIAL_WORKLOAD.json"
SCHEMA = "hawking.future.trial_workload.v1"

FALCON_NEEDLE = "falcon-h1"
# Documented identity in specimen_verify's CLI and the Odyssey I curriculum.
# Used only to recognise the cheap school, never as a substitute for a listing.
FALCON_CANONICAL = "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb"

GPU_RESOURCE = at.GPU_RESOURCE
LONGER_TRIALS = ("3h", "6h")

ROLE_FAST = "FAST_SPECIMEN_SCIENCE"
ROLE_LONG = "LONG_SPECIMEN_VERIFY"
ROLE_NEGATIVE = "NEGATIVE_SCIENCE"
ROLE_SCREEN = "MULTI_FIDELITY_SCREEN"
ROLE_HCLI = "HCLI_SELF_OPTIMIZE"
ROLE_O2 = "ODYSSEY_II_TRANSFER"
ROLE_O3 = "ODYSSEY_III_ATTACK"
ROLE_SUPPORT = "SUPPORTING_REPLAN"

# Why each role is in a long trial. Proportions are unit counts, not a clock.
ROLE_WHY: dict[str, str] = {
    ROLE_FAST: (
        "Falcon-H1-7B is the cheap Odyssey I procedural school and is already "
        "whole-tree verified; CPU science against it is real and finishes in "
        "the short window"
    ),
    ROLE_LONG: (
        "whole-tree specimen verification is real disk work that takes real "
        "minutes; a completed specimen stands if the trial is cancelled"
    ),
    ROLE_NEGATIVE: (
        "refuse_if_dead must actually be able to kill a proposal; a trial that "
        "never refuses is not using the scar index"
    ),
    ROLE_SCREEN: (
        "a cheap falsifier that kills must refuse later meta-funnel gates so "
        "expensive children never launch"
    ),
    ROLE_HCLI: (
        "Hawking that cannot treat its own machinery as a candidate will keep "
        "paying the same costs; hcli_self_profile ranks and prunes"
    ),
    ROLE_O2: (
        "Odyssey II asks what Hawking already learned; a trial that never "
        "proposes a transfer hypothesis cannot compound"
    ),
    ROLE_O3: (
        "a law with no attack is refused; Odyssey III must ride along with II, "
        "not wait for II to finish"
    ),
}

_RECEIPT_LIT = re.compile(r'^RECEIPT\s*=\s*"([^"]+\.json)"', re.M)


class WorkloadRefused(ValueError):
    """A workload that would look like a trial mix without being one."""

    def __init__(self, message: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.missing = [str(x) for x in missing]


# ---------------------------------------------------------------------------
# Specimens. Absence is a refusal, never a substitute parent.
# ---------------------------------------------------------------------------


def _falcon_hits(names: Iterable[str]) -> list[str]:
    hits = []
    for name in names:
        blob = str(name).lower().replace("_", "-")
        if FALCON_NEEDLE in blob:
            hits.append(str(name))
    if FALCON_CANONICAL in hits:
        return [FALCON_CANONICAL] + [n for n in hits if n != FALCON_CANONICAL]
    return sorted(hits)


def select_falcon(names: Sequence[str] | None = None) -> dict[str, Any]:
    """The cheap procedural school. Another specimen is not Falcon."""
    listing = list(names) if names is not None else sv.list_specimens()
    hits = _falcon_hits(listing)
    if not hits:
        avail = sv.available()
        raise WorkloadRefused(
            "FAST_SPECIMEN_SCIENCE missing: Falcon-H1-7B is not in the specimen "
            f"listing (lake_mounted={avail.get('mounted')}); refusing to "
            "substitute another model",
            missing=[ROLE_FAST],
        )
    name = hits[0]
    present = name in listing
    nbytes: int | None = None
    if present:
        try:
            nbytes = sum(p.stat().st_size for p in sv.specimen_files(name))
        except (OSError, sv.SpecimenError):
            nbytes = None
    return {
        "name": name,
        "present": present,
        "specimen_bytes": nbytes,
        "role": ROLE_FAST,
        "why": ROLE_WHY[ROLE_FAST],
    }


def select_long(names: Sequence[str] | None = None, *, exclude: str | None = None) -> dict[str, Any]:
    """Genuinely long whole-tree work: the largest non-Falcon specimen.

    Falcon is the cheap school and cannot impersonate the long unit. An empty
    non-Falcon listing is a refusal, not a default back onto Falcon.
    """
    listing = list(names) if names is not None else sv.list_specimens()
    skip = {exclude} if exclude else set()
    rows: list[tuple[int, str]] = []
    unread: list[str] = []
    for name in listing:
        if name in skip or _falcon_hits([name]):
            continue
        try:
            nbytes = sum(p.stat().st_size for p in sv.specimen_files(name))
        except (OSError, sv.SpecimenError):
            unread.append(name)
            continue
        rows.append((nbytes, name))
    if not rows:
        why = (
            "LONG_SPECIMEN_VERIFY missing: no non-Falcon specimen could be sized "
            "on disk"
        )
        if unread:
            why += f" (unreadable: {unread[:4]})"
        if not listing:
            why += "; specimen listing is empty"
        raise WorkloadRefused(why, missing=[ROLE_LONG])
    rows.sort()
    nbytes, name = rows[-1]
    return {
        "name": name,
        "present": True,
        "specimen_bytes": nbytes,
        "n_larger_than_falcon": len(rows),
        "role": ROLE_LONG,
        "why": ROLE_WHY[ROLE_LONG],
    }


# ---------------------------------------------------------------------------
# Bindings, catalog, GPU park
# ---------------------------------------------------------------------------


def _item_by_id(book: fr.FrontierBook, fid: str) -> dict[str, Any] | None:
    for item in book.items:
        if item.get("id") == fid:
            return item
    return None


def _receipt_literal(module: str) -> str | None:
    path = REPO / "tools" / "future" / module
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _RECEIPT_LIT.search(text)
    return m.group(1) if m else None


def requires_gpu_authority(unit: Mapping[str, Any], book: fr.FrontierBook | None = None) -> bool:
    """True iff launching this unit would need a GPU the sidecar does not have."""
    rc = str(unit.get("resource_class") or "")
    if rc in GPU_RESOURCE:
        return True
    lanes = {str(x) for x in (unit.get("required_lanes") or [])}
    if lanes & set(fr.HARDWARE_LANES):
        return True
    fid = str(unit.get("frontier_id") or "")
    current = book
    if current is None and fid:
        current = load_book()
    if current is not None and fid:
        item = _item_by_id(current, fid)
        if item is not None:
            if str(item.get("resource_class") or "") in GPU_RESOURCE:
                return True
            if set(item.get("required_lanes") or []) & set(fr.HARDWARE_LANES):
                return True
    return False


def load_book() -> fr.FrontierBook:
    return fr.load_book()


# ---------------------------------------------------------------------------
# Replan edges. Derived from the catalog and recovered APIs, not staged.
# ---------------------------------------------------------------------------


def _priority_how(cause_module: str, effect: Mapping[str, Any]) -> str | None:
    """The priority-change verb, or None if the edge is merely informational."""
    evidence = {Path(str(e)).name for e in (effect.get("evidence") or []) if e}
    verifier = str(effect.get("verifier") or "")
    eid = str(effect.get("id") or "")
    if cause_module == "negative_index.py":
        if (
            "NEGATIVE_SCIENCE_INDEX.json" in evidence
            or "refuse_if_dead" in verifier
            or "negative_index" in verifier
        ):
            return (
                "refuse_if_dead can kill the effect unit at admission; a live "
                "family is scheduled, a scar-dead one is not"
            )
    if cause_module == "odyssey2_law_store.py":
        if eid.startswith("FT.ODYSSEY_ADVERSARY") or "odyssey_iii" in verifier:
            return pl.LISTEN_RULE
    if cause_module == "specimen_verify.py":
        if eid == "FT.ODYSSEY_TRANSFER.re-earn" or "odyssey" in verifier:
            return (
                "odyssey_launch refuses on specimen_curriculum_ready; "
                "WHOLE_TREE_VERIFIED can raise that criterion, PARTIAL keeps "
                "launch refused"
            )
    return None


def catalog_replan_edges(book: fr.FrontierBook) -> list[dict[str, Any]]:
    """Causal edges where A's receipt is evidence for B AND A's result can
    change B's priority. Walked from the frontier catalog + BINDINGS receipts,
    not a hand-staged pair table.
    """
    writers: dict[str, list[tuple[str, str, str]]] = {}
    for module, (fid, species) in orch.BINDINGS.items():
        rec = _receipt_literal(module)
        if not rec:
            continue
        writers.setdefault(rec, []).append((fid, module, species))

    seen: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    for item in book.items:
        eid = str(item.get("id") or "")
        if not eid:
            continue
        for ev in item.get("evidence") or []:
            base = Path(str(ev)).name
            for cause_fid, module, species in writers.get(base, []):
                if cause_fid == eid:
                    continue
                how = _priority_how(module, item)
                if not how:
                    continue
                key = (cause_fid, eid)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "cause_frontier_id": cause_fid,
                        "effect_frontier_id": eid,
                        "cause_module": module,
                        "effect_species": species,
                        "how": how,
                        "evidence_receipt": base,
                        "derived_from": "frontier_catalog_evidence",
                    }
                )
    return edges


def recovered_mechanism_edges(book: fr.FrontierBook) -> list[dict[str, Any]]:
    """Priority edges that live in recovered APIs rather than catalog evidence.

    Adaptive verification's receipt is not listed on the meta-gates item;
    the cheap-kill → saved funnel-child relationship is the module. Specimen
    verification is not listed on re-earn; odyssey_launch.CRITERION_IDS is.
    An API that is missing does not mint an edge.
    """
    ids = {str(i.get("id") or "") for i in book.items}
    edges: list[dict[str, Any]] = []
    children = av.funnel_child_workunits()
    if (
        "FT.VERIFICATION.repro" in ids
        and "FT.MODEL_REPRESENTATION.meta-gates-3-9" in ids
        and children
    ):
        edges.append(
            {
                "cause_frontier_id": "FT.VERIFICATION.repro",
                "effect_frontier_id": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
                "cause_module": "adaptive_verification.py",
                "effect_module": "meta_ready.py",
                "how": (
                    "adaptive_verification.saved() names the meta-funnel gates "
                    "that are not launched after a cheap kill; those gates are "
                    "the remaining work of FT.MODEL_REPRESENTATION.meta-gates-3-9"
                ),
                "named_children": [c["workunit"] for c in children],
                "derived_from": "recovered_mechanism:adaptive_verification.funnel_child_workunits",
            }
        )
    if (
        "FT.MODEL_CAPABILITY.hard-gates" in ids
        and "FT.ODYSSEY_TRANSFER.re-earn" in ids
        and "specimen_curriculum_ready" in ol.CRITERION_IDS
    ):
        edges.append(
            {
                "cause_frontier_id": "FT.MODEL_CAPABILITY.hard-gates",
                "effect_frontier_id": "FT.ODYSSEY_TRANSFER.re-earn",
                "cause_module": "specimen_verify.py",
                "effect_module": "odyssey_launch.py",
                "how": (
                    "odyssey_launch refuses on specimen_curriculum_ready; a "
                    "WHOLE_TREE_VERIFIED specimen can raise that criterion, "
                    "PARTIAL keeps launch refused"
                ),
                "derived_from": "recovered_mechanism:odyssey_launch.CRITERION_IDS",
            }
        )
    if (
        "FT.ODYSSEY_TRANSFER.flash-qwen27" in ids
        and "FT.ODYSSEY_ADVERSARY.attacks" in ids
    ):
        edges.append(
            {
                "cause_frontier_id": "FT.ODYSSEY_TRANSFER.flash-qwen27",
                "effect_frontier_id": "FT.ODYSSEY_ADVERSARY.attacks",
                "cause_module": "odyssey2_law_store.py",
                "effect_module": "odyssey3_adversary.py",
                "how": pl.LISTEN_RULE,
                "derived_from": "recovered_mechanism:phase_listeners.LISTEN_RULE",
            }
        )
    return edges


def replan_edges(book: fr.FrontierBook) -> list[dict[str, Any]]:
    """Union of catalog-derived and recovered-mechanism edges, deduped."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in catalog_replan_edges(book) + recovered_mechanism_edges(book):
        key = (str(edge["cause_frontier_id"]), str(edge["effect_frontier_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def replan_pairs(
    units: Sequence[Mapping[str, Any]],
    book: fr.FrontierBook,
) -> list[dict[str, Any]]:
    """Unit pairs whose first result can change the second's priority."""
    by_fid: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        fid = str(unit.get("frontier_id") or "")
        if fid:
            by_fid.setdefault(fid, []).append(unit)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in replan_edges(book):
        causes = by_fid.get(str(edge["cause_frontier_id"])) or []
        effects = by_fid.get(str(edge["effect_frontier_id"])) or []
        for cause in causes:
            for effect in effects:
                if cause is effect:
                    continue
                if str(cause.get("id")) == str(effect.get("id")):
                    continue
                key = (str(cause.get("id")), str(effect.get("id")))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "cause_id": cause.get("id"),
                        "effect_id": effect.get("id"),
                        "cause_frontier_id": edge["cause_frontier_id"],
                        "effect_frontier_id": edge["effect_frontier_id"],
                        "cause_module": cause.get("module") or edge.get("cause_module"),
                        "effect_module": effect.get("module") or edge.get("effect_module"),
                        "how": edge["how"],
                        "derived_from": edge["derived_from"],
                    }
                )
    return pairs


# ---------------------------------------------------------------------------
# Mix
# ---------------------------------------------------------------------------


def _known_trial(trial_id: str) -> None:
    if trial_id not in at.TRIAL_IDS:
        raise WorkloadRefused(
            f"unknown trial_id {trial_id!r}; known {at.TRIAL_IDS}",
            missing=["trial_id"],
        )


def mix(trial_id: str) -> dict[str, Any]:
    """Required proportions for one trial, and why each role is there."""
    _known_trial(trial_id)
    longer = trial_id in LONGER_TRIALS
    if trial_id == "15m":
        required = (ROLE_FAST, ROLE_NEGATIVE, ROLE_HCLI)
    elif trial_id == "1h":
        required = (ROLE_FAST, ROLE_NEGATIVE, ROLE_SCREEN, ROLE_HCLI, ROLE_O2, ROLE_O3)
    else:
        required = (
            ROLE_FAST,
            ROLE_LONG,
            ROLE_NEGATIVE,
            ROLE_SCREEN,
            ROLE_HCLI,
            ROLE_O2,
            ROLE_O3,
        )
    proportions = {
        role: {
            "min_units": 1,
            "max_fraction": 0.5,
            "why": ROLE_WHY[role],
        }
        for role in required
    }
    return {
        "trial_id": trial_id,
        "duration_s": at.TRIAL_DURATION_S[trial_id],
        "required_roles": list(required),
        "proportions": proportions,
        "replan_required": longer,
        "replan_why": (
            "a trial that cannot demonstrate replanning cannot pass the 3h bar; "
            "the pair must be derived from the live frontier, not staged"
            if longer
            else "shorter trials may carry a replan pair but are not refused without one"
        ),
        "gpu_rule": "no unit may require GPU authority; units that would are emitted SLEEPING",
        "padding_rule": (
            "if the trial were cancelled halfway, already-done work must have "
            "been worth doing; a unit that fails that test is padding"
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Unit construction and admission
# ---------------------------------------------------------------------------


def _park(unit: dict[str, Any]) -> dict[str, Any]:
    unit["status"] = "blocked"
    unit["classification"] = "SLEEPING"
    unit["disposition"] = "SLEEPING"
    unit["gpu_authority"] = False
    unit.setdefault(
        "blocked_reason",
        "GPU authority is required and this sidecar has none; parked SLEEPING",
    )
    return unit


def make_unit(
    module: str,
    *,
    description: str,
    mix_role: str,
    book: fr.FrontierBook,
    specimen: str | None = None,
    why_worth_doing: str,
    resource_class: str | None = None,
) -> dict[str, Any]:
    """One bound unit. GPU-bound work is parked, never completed."""
    if module not in orch.BINDINGS:
        raise WorkloadRefused(
            f"module {module!r} is not in orchestration.BINDINGS; a unit that "
            "is not bound to a real frontier item is rejected",
            missing=["binding"],
        )
    fid, species = orch.BINDINGS[module]
    item = _item_by_id(book, fid)
    if item is None:
        raise WorkloadRefused(
            f"unit is not bound to a real frontier item: {fid} (module {module})",
            missing=["frontier_item"],
        )
    desc = str(description).strip()
    if not desc or at.is_low_information(
        {"description": desc, "verifier": item.get("verifier"), "frontier_id": fid}
    ):
        raise WorkloadRefused(
            f"padding refused: {module} description {desc!r} would not be worth "
            "doing if the trial were cancelled halfway",
            missing=["worth_doing_anyway"],
        )
    item_rc = str(item.get("resource_class") or "STATIC_ANALYSIS")
    rc = resource_class or item_rc
    gpu = rc in GPU_RESOURCE or bool(set(item.get("required_lanes") or []) & set(fr.HARDWARE_LANES))
    slug = module.removesuffix(".py")
    unit_id = f"WU.TRIAL.{mix_role}.{slug}"
    if specimen:
        unit_id = f"{unit_id}.{specimen.split('@')[0].split('--')[-1]}"
    verifier = str(item.get("verifier") or f"future.{slug}")
    if gpu:
        unit = at.sleeping_hardware_unit(
            unit_id,
            blocker_id="no_gpu_authority",
            reason=(
                f"{module} informs {fid} but required lanes "
                f"{item.get('required_lanes')} need GPU authority this sidecar "
                "does not have"
            ),
            frontier_id=fid,
        )
        unit = _park(unit)
        unit["species"] = species
    else:
        unit = at.cpu_workunit(
            unit_id,
            frontier_id=fid,
            description=desc,
            verifier=verifier,
        )
        unit["species"] = species
        unit["resource_class"] = rc
        unit["status"] = "pending"
        unit["classification"] = "STATIC_ONLY"
    unit["module"] = module
    unit["capability"] = module
    unit["frontier_id"] = fid
    unit["mix_role"] = mix_role
    unit["worth_doing_anyway"] = why_worth_doing
    unit["gpu_authority"] = False
    unit["evidence_class"] = "STATIC_ONLY"
    unit["required_lanes"] = list(item.get("required_lanes") or [])
    if specimen:
        unit["specimen"] = specimen
    return unit


def admit_unit(
    unit: Mapping[str, Any],
    *,
    book: fr.FrontierBook,
    queued: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Refuse unbound, mismatched, duplicate, or padding units. Park GPU."""
    row = dict(unit)
    fid = str(row.get("frontier_id") or "")
    if not fid or _item_by_id(book, fid) is None:
        raise WorkloadRefused(
            f"unit is not bound to a real frontier item: {fid or '<missing>'}",
            missing=["frontier_item"],
        )
    module = str(row.get("module") or row.get("capability") or "")
    if module not in orch.BINDINGS:
        raise WorkloadRefused(
            f"module {module!r} is not in orchestration.BINDINGS; a unit that "
            "is not bound to a real frontier item is rejected",
            missing=["binding"],
        )
    bound_fid, bound_species = orch.BINDINGS[module]
    if bound_fid != fid:
        raise WorkloadRefused(
            f"{module} is bound to {bound_fid}, not {fid}; a mismatched "
            "binding is fiction",
            missing=["binding_match"],
        )
    row.setdefault("species", bound_species)
    ident = at.work_identity(row)
    for prev in queued:
        if at.work_identity(prev) == ident:
            raise WorkloadRefused(
                f"duplicate work identity {ident}; unique ids do not make "
                "work distinct",
                missing=["distinct_work"],
            )
    if at.is_low_information(row):
        raise WorkloadRefused(
            f"padding refused for {row.get('id')}: low-information unit would "
            "not be worth doing if the trial were cancelled halfway",
            missing=["worth_doing_anyway"],
        )
    if requires_gpu_authority(row, book):
        row = _park(row)
    row["gpu_authority"] = False
    return row


def admit_workload(
    units: Sequence[Mapping[str, Any]],
    trial_id: str,
    *,
    book: fr.FrontierBook | None = None,
) -> dict[str, Any]:
    """Admit a composed set. 3h/6h without a replan pair is refused by name."""
    spec = mix(trial_id)
    current = book or load_book()
    admitted: list[dict[str, Any]] = []
    for raw in units:
        admitted.append(admit_unit(raw, book=current, queued=admitted))
    roles = {str(u.get("mix_role") or "") for u in admitted}
    missing_roles = [r for r in spec["required_roles"] if r not in roles]
    if missing_roles:
        raise WorkloadRefused(
            f"trial {trial_id} missing required mix roles {missing_roles}",
            missing=missing_roles,
        )
    runnable = [u for u in admitted if str(u.get("status") or "") != "blocked"]
    n = len(runnable) or len(admitted)
    for role, prop in spec["proportions"].items():
        n_role = sum(1 for u in admitted if u.get("mix_role") == role)
        frac = n_role / n if n else 0.0
        if n_role < int(prop["min_units"]):
            raise WorkloadRefused(
                f"trial {trial_id} role {role} has {n_role} units; need "
                f"{prop['min_units']}: {prop['why']}",
                missing=[role],
            )
        if frac - float(prop["max_fraction"]) > 1e-9 and n_role > int(prop["min_units"]):
            raise WorkloadRefused(
                f"trial {trial_id} role {role} is {frac:.2f} of the mix "
                f"(max {prop['max_fraction']}); a flood of one behaviour is padding",
                missing=["mix_balance"],
            )
    pairs = replan_pairs(admitted, current)
    if spec["replan_required"] and not pairs:
        raise WorkloadRefused(
            f"trial {trial_id} has no replan pair: no unit's result can change "
            "another unit's priority, derived from the live frontier. The 3h "
            "bar cannot be demonstrated without one.",
            missing=["replan_pair"],
        )
    sleeping = [u for u in admitted if str(u.get("classification") or "") == "SLEEPING"]
    pending_gpu = [
        u["id"] for u in admitted
        if requires_gpu_authority(u, current) and str(u.get("status") or "") == "pending"
    ]
    if pending_gpu:
        raise WorkloadRefused(
            f"GPU units leaked as pending rather than SLEEPING: {pending_gpu}",
            missing=["gpu_park"],
        )
    return {
        "admitted": True,
        "trial_id": trial_id,
        "units": admitted,
        "n_units": len(admitted),
        "sleeping": sleeping,
        "n_sleeping": len(sleeping),
        "replan_pairs": pairs,
        "n_replan_pairs": len(pairs),
        "mix": spec,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def _specimen_description(name: str) -> str:
    # Same work identity the autonomy driver already queues per specimen.
    return (
        f"recompute every published digest for specimen {name} "
        "and decide whole-tree verification offline"
    )


def _plan(trial_id: str, book: fr.FrontierBook) -> list[dict[str, Any]]:
    """The unit set for one trial. Every row is work that stands alone."""
    spec = mix(trial_id)
    falcon = select_falcon()
    units: list[dict[str, Any]] = [
        make_unit(
            "specimen_verify.py",
            description=_specimen_description(falcon["name"]),
            mix_role=ROLE_FAST,
            book=book,
            specimen=falcon["name"],
            why_worth_doing=(
                "a completed Falcon whole-tree receipt is Odyssey I curriculum "
                "integrity and stands if the trial dies"
            ),
            resource_class="IO_HEAVY",
        ),
        make_unit(
            "negative_index.py",
            description=(
                "rebuild the scar index that prunes work before it is scheduled"
            ),
            mix_role=ROLE_NEGATIVE,
            book=book,
            why_worth_doing=(
                "a current scar index is the campaign's own next work; "
                "rediscovery is not free"
            ),
        ),
        make_unit(
            "hcli_self_profile.py",
            description=(
                "rank HCLI process-wall as a search space and prune; "
                "SELF_MEASURED_DIRTY decides nothing"
            ),
            mix_role=ROLE_HCLI,
            book=book,
            why_worth_doing=(
                "the git-status lock incident is already the worked example; "
                "a ranked cost list stands without a GPU"
            ),
        ),
    ]
    if ROLE_SCREEN in spec["required_roles"]:
        units.append(
            make_unit(
                "adaptive_verification.py",
                description=(
                    "run the cheapest-first multi-fidelity screen so a cheap "
                    "falsifier can refuse later meta-funnel gates"
                ),
                mix_role=ROLE_SCREEN,
                book=book,
                why_worth_doing=(
                    "a kill receipt names the funnel work that was therefore "
                    "not done; that decision stands"
                ),
            )
        )
    if ROLE_O2 in spec["required_roles"]:
        units.append(
            make_unit(
                "odyssey2_law_store.py",
                description=(
                    "list Flash↔Qwen27 transfer hypotheses still unevidenced at "
                    "ARCHITECTURE_FAMILY; transfer is a PROPOSAL not Law.promote()"
                ),
                mix_role=ROLE_O2,
                book=book,
                why_worth_doing=(
                    "scoped laws are Odyssey II's product; a sealed store is "
                    "usable even if Phase III never starts"
                ),
            )
        )
    if ROLE_O3 in spec["required_roles"]:
        units.append(
            make_unit(
                "odyssey3_adversary.py",
                description=(
                    "generate Odyssey III attack specs against every current law; "
                    "a law with no attack is refused"
                ),
                mix_role=ROLE_O3,
                book=book,
                why_worth_doing=(
                    "attack specs are STATIC_ONLY and worth having before any "
                    "measurement; a law that emits none is a bug"
                ),
            )
        )
    if ROLE_LONG in spec["required_roles"]:
        long = select_long(exclude=falcon["name"])
        units.append(
            make_unit(
                "specimen_verify.py",
                description=_specimen_description(long["name"]),
                mix_role=ROLE_LONG,
                book=book,
                specimen=long["name"],
                why_worth_doing=(
                    "a completed large-specimen receipt is the work Odyssey I "
                    "still needs; it stands if the clock runs out"
                ),
                resource_class="IO_HEAVY",
            )
        )
    if spec["replan_required"]:
        # Supporting units complete frontier-derived pairs already required
        # by the mix (negative→ngram, screen→meta-gates, specimen→launch).
        units.append(
            make_unit(
                "ngram_school.py",
                description=(
                    "generate n-gram-school representation candidates below Q4 "
                    "without fitting weights, scored against the negative index"
                ),
                mix_role=ROLE_SUPPORT,
                book=book,
                why_worth_doing=(
                    "a fresh candidate set scored against the scar index is "
                    "representation work the campaign already queued"
                ),
            )
        )
        units.append(
            make_unit(
                "meta_ready.py",
                description=(
                    "prepare meta funnel gates 3-9 so they can start the moment "
                    "teacher rows arrive, without fabricating teacher rows"
                ),
                mix_role=ROLE_SUPPORT,
                book=book,
                why_worth_doing=(
                    "wiring gates 3-9 to the corpus-arrival contract is F019 "
                    "CPU work; a cheap screen kill is what must be able to "
                    "stop it"
                ),
            )
        )
        units.append(
            make_unit(
                "odyssey_launch.py",
                description="re-evaluate every Odyssey I launch criterion",
                mix_role=ROLE_SUPPORT,
                book=book,
                why_worth_doing=(
                    "the launch gate currently refuses; keeping the refusal "
                    "current is the work, and specimen results change it"
                ),
            )
        )
    return units


def compose(trial_id: str, *, book: fr.FrontierBook | None = None) -> dict[str, Any]:
    """The unit set for a trial, drawn from real frontier items and bindings."""
    _known_trial(trial_id)
    current = book or load_book()
    planned = _plan(trial_id, current)
    admitted = admit_workload(planned, trial_id, book=current)
    admitted["purpose"] = (
        "mixed real frontier work for a frozen autonomy trial; not a toy checklist"
    )
    admitted["declared_not_executed"] = (
        "compose selects and binds; it does not invoke. Invocation is the "
        "autonomy driver's job. Declared capability is not executed capability."
    )
    admitted["available_lanes"] = list(fr.THIS_HOST_LANES)
    admitted["blocked_lanes"] = list(fr.HARDWARE_LANES)
    return admitted


def compose_all(book: fr.FrontierBook | None = None) -> dict[str, Any]:
    current = book or load_book()
    by_trial: dict[str, Any] = {}
    refusals: dict[str, str] = {}
    for tid in at.TRIAL_IDS:
        try:
            by_trial[tid] = compose(tid, book=current)
        except WorkloadRefused as exc:
            refusals[tid] = str(exc)
            by_trial[tid] = {
                "admitted": False,
                "trial_id": tid,
                "refused": True,
                "why": str(exc),
                "missing": list(exc.missing),
            }
    return {"by_trial": by_trial, "refusals": refusals, "book": current}


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    current = load_book()
    packed = compose_all(current)
    edges = replan_edges(current)
    catalog = catalog_replan_edges(current)
    mechanisms = recovered_mechanism_edges(current)
    public_trials = {}
    for tid, doc in packed["by_trial"].items():
        if not doc.get("admitted"):
            public_trials[tid] = {
                "admitted": False,
                "missing": doc.get("missing"),
                "why": doc.get("why"),
            }
            continue
        public_trials[tid] = {
            "admitted": True,
            "n_units": doc["n_units"],
            "n_sleeping": doc["n_sleeping"],
            "n_replan_pairs": doc["n_replan_pairs"],
            "required_roles": doc["mix"]["required_roles"],
            "replan_required": doc["mix"]["replan_required"],
            "units": [
                {
                    "id": u.get("id"),
                    "module": u.get("module"),
                    "frontier_id": u.get("frontier_id"),
                    "species": u.get("species"),
                    "resource_class": u.get("resource_class"),
                    "mix_role": u.get("mix_role"),
                    "status": u.get("status"),
                    "specimen": u.get("specimen"),
                    "worth_doing_anyway": u.get("worth_doing_anyway"),
                    "description": u.get("description"),
                }
                for u in doc["units"]
            ],
            "replan_pairs": doc["replan_pairs"],
        }
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Compose mixed real frontier work for the frozen autonomy trials "
            "so a resident cannot pass by surviving a clock while doing nothing "
            "useful."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "trial_ids": list(at.TRIAL_IDS),
        "longer_trials": list(LONGER_TRIALS),
        "mix_by_trial": {tid: mix(tid) for tid in at.TRIAL_IDS},
        "composed": public_trials,
        "n_catalog_replan_edges": len(catalog),
        "n_mechanism_replan_edges": len(mechanisms),
        "n_replan_edges": len(edges),
        "replan_edges": edges,
        "compose_refusals": packed["refusals"],
        "recovered_implementation": [
            "tools/future/autonomy_run.py queue construction (SAFE_CAPABILITIES, "
            "specimen_verify subprocess, identity = species+frontier+resource+description)",
            "tools/future/autonomy_trial.py TRIAL_IDS, work_identity, cpu_workunit, "
            "sleeping_hardware_unit, is_low_information, GPU_RESOURCE",
            "tools/future/orchestration.py BINDINGS / emit_workunit / invoke",
            "tools/future/frontiers.py catalog items, THIS_HOST_LANES, HARDWARE_LANES, load_book",
            "tools/future/specimen_verify.py list_specimens / verify_specimen / Falcon identity",
            "tools/future/adaptive_verification.py screen / saved / funnel_child_workunits",
            "tools/future/phase_listeners.py LISTEN_RULE (II law spawns III; empty store emits zero)",
            "tools/future/odyssey_launch.py CRITERION_IDS.specimen_curriculum_ready",
            "tools/future/hcli_self_profile.py rank_attributed_costs / as_actionable",
            "tools/future/negative_index.py refuse_if_dead",
            "tools/future/trial_freeze.py freeze(trial_id) known-id refusal shape",
        ],
        "gaps_closed": [
            "no composer produced a mixed real-frontier unit set for 15m/1h/3h/6h",
            "3h/6h without a frontier-derived replan pair is now a named refusal",
            "duplicate work identity is refused even when ids differ",
            "unbound / mismatched bindings are refused rather than defaulted",
            "GPU-bound units park SLEEPING instead of leaking as pending",
        ],
        "negative_findings": [
            "this module is not in orchestration.BINDINGS (that table is outside this lane's WRITE list)",
            "compose does not invoke; a unit listed here is declared, not executed",
            "resident_model_cognition is not exercised",
            "a catalog evidence edge that cannot change priority is not a replan pair",
        ],
        "resident_callable": {
            "entry_point": "tools.future.trial_workload.compose(trial_id)",
            "workunit": (
                "one CPU_ANALYSIS unit; compose the mixed real-frontier workload "
                "for a named autonomy trial and refuse a 3h/6h set with no replan pair"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.CHILD_RESIDENT.launch",
            "fails_closed": (
                "unknown trial_id raises WorkloadRefused; unbound / mismatched / "
                "duplicate / padding units are refused; 3h/6h without a replan "
                "pair names the missing pair; Falcon cannot be substituted; a "
                "GPU unit is SLEEPING not pending"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/trial_workload.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--compose", metavar="TRIAL")
    ap.add_argument("--mix", metavar="TRIAL")
    a = ap.parse_args()
    if a.mix:
        import json
        print(json.dumps(mix(a.mix), indent=1, sort_keys=True))
        return 0
    if a.compose:
        import json
        print(json.dumps(compose(a.compose), indent=1, sort_keys=True, default=str))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
