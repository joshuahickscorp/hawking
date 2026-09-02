"""Ingest local Hawking receipts as §19.12 HAWKING SELF-BOUNTY work.

Intake reads LOCAL artifacts only. Unknown schemas refuse rather than guess.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from tools.theia.bounty import (
    Bounty,
    BountyClass,
    make_internal_bounty,
)
from tools.theia.intake import IntakeRefused, VerificationFailed, local_artifact
from tools.theia.labs import LabKind, SelfBountyKind
from tools.theia.value import DeclaredFactor, ValueInputs, ValueRefused
from tools.future._common import REPO


SCHEMA_KIND: dict[str, SelfBountyKind] = {
    "hawking.future.autonomy_scars.v1": SelfBountyKind.NEGATIVE_SCIENCE,
    "hawking.future.campaign_scars.v1": SelfBountyKind.NEGATIVE_SCIENCE,
    "hawking.future.complete_ebpw.v1": SelfBountyKind.REPRESENTATION_WIN,
    "hawking.future.device_compiler.v1": SelfBountyKind.NEW_COMPILER_PASS,
    "hawking.future.repro_science.v1": SelfBountyKind.REGRESSIONS,
    "hawking.future.kernel_geometry.v1": SelfBountyKind.KERNEL_WIN,
}

KIND_RECEIPTS: dict[SelfBountyKind, str] = {
    SelfBountyKind.NEGATIVE_SCIENCE: "AUTONOMY_SCARS.json",
    SelfBountyKind.AUTONOMY_RECOVERY_PROOF: "AUTONOMY_SCARS.json",
    SelfBountyKind.REPRESENTATION_WIN: "COMPLETE_EBPW.json",
    SelfBountyKind.NEW_COMPILER_PASS: "DEVICE_COMPILER.json",
    SelfBountyKind.REGRESSIONS: "REPRO_SCIENCE.json",
    SelfBountyKind.KERNEL_WIN: "KERNEL_GEOMETRY.json",
}

SCHEMA_SECONDARY: dict[str, tuple[SelfBountyKind, ...]] = {
    "hawking.future.autonomy_scars.v1": (SelfBountyKind.AUTONOMY_RECOVERY_PROOF,),
}

VERIFIER = "tools.theia.intake.verify_receipt"

UNIT_SOURCE = (
    "declared unit baseline so the H.1 denominator is defined; STATIC_ONLY; "
    "not a hardware measurement and not complete_ebpw (wrong axes)"
)


def classify(
    doc: Mapping[str, Any], kind: SelfBountyKind | None = None
) -> SelfBountyKind:
    schema = doc.get("schema")
    if not isinstance(schema, str) or schema not in SCHEMA_KIND:
        raise IntakeRefused(
            f"no self-bounty mapping for schema {schema!r}; refusing rather than guessing"
        )
    primary = SCHEMA_KIND[schema]
    secondary = SCHEMA_SECONDARY.get(schema, ())
    if kind is None:
        return primary
    if kind is not primary and kind not in secondary:
        raise IntakeRefused(
            f"kind {kind.value!r} is not valid for schema {schema!r} "
            f"(primary {primary.value!r}, secondary {[k.value for k in secondary]})"
        )
    return kind


def receipt_for_kind(kind: SelfBountyKind) -> Path:
    name = KIND_RECEIPTS.get(kind)
    if not name:
        raise IntakeRefused(f"no default receipt for self-bounty kind {kind.value!r}")
    path = REPO / "receipts" / "future" / name
    if not path.is_file():
        raise IntakeRefused(f"self-bounty receipt missing: {path}")
    return path


def _unit(name: str, source: str) -> DeclaredFactor:
    return DeclaredFactor(value=Fraction(1), name=name, source=source)


def _schedule(
    path: Path,
    *,
    kind: SelfBountyKind,
    schema: str,
    information_gain: int,
    information_source: str,
    transfer_value: int,
    transfer_source: str,
    reward_note: str,
) -> ValueInputs:
    if information_gain < 1:
        raise ValueRefused(f"{kind.value}: information_gain grounded at {information_gain}")
    if transfer_value < 1:
        raise ValueRefused(f"{kind.value}: transfer_value grounded at {transfer_value}")
    return ValueInputs(
        verified_reward=_unit(
            "verified_reward",
            f"H.1: verified_reward may include {reward_note}; schema {schema}",
        ),
        probability_of_success=_unit(
            "probability_of_success",
            f"artifact already on disk at {path}; intake is local parse, not a search",
        ),
        information_gain=DeclaredFactor(
            value=Fraction(information_gain),
            name="information_gain",
            source=information_source,
        ),
        transfer_value=DeclaredFactor(
            value=Fraction(transfer_value),
            name="transfer_value",
            source=transfer_source,
        ),
        strategic_relevance=_unit(
            "strategic_relevance",
            f"HAWKING SELF-BOUNTY laboratory, §19.12: {kind.value}",
        ),
        wall_time=_unit("wall_time", UNIT_SOURCE),
        compute_cost=_unit("compute_cost", UNIT_SOURCE),
        human_cost=_unit("human_cost", UNIT_SOURCE),
        risk=_unit(
            "risk",
            "local readonly receipt; no network, no ACTIVE_TEST, no credentials",
        ),
        opportunity_cost=_unit("opportunity_cost", UNIT_SOURCE),
    )


def _scar_counts(doc: Mapping[str, Any], schema: str) -> tuple[int, int]:
    scars = doc.get("scars")
    if not isinstance(scars, list) or not scars:
        raise ValueRefused(f"{schema} receipt has no scars to ground information_gain")
    n = doc.get("n_scars")
    if n != len(scars):
        raise ValueRefused(f"n_scars {n} != len(scars) {len(scars)}")
    n_laws = sum(1 for s in scars if s.get("law"))
    if n_laws == 0 and doc.get("general_law"):
        n_laws = 1
    if n_laws == 0:
        raise ValueRefused("no law fields to ground transfer_value")
    return int(n), int(n_laws)


def value_inputs_from_receipt(
    path: Path,
    doc: Mapping[str, Any],
    kind: SelfBountyKind | None = None,
) -> ValueInputs:
    schema = doc.get("schema")
    kind = classify(doc, kind)
    if kind is SelfBountyKind.AUTONOMY_RECOVERY_PROOF:
        recovered = doc.get("recovered_implementation")
        if not isinstance(recovered, list) or not recovered:
            raise ValueRefused("no recovered_implementation list to ground recovery proof")
        n_laws = 1 if doc.get("general_law") else 0
        if n_laws == 0:
            raise ValueRefused("no general_law to ground transfer_value for recovery proof")
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=len(recovered),
            information_source="receipt recovered_implementation entries",
            transfer_value=n_laws,
            transfer_source="receipt general_law",
            reward_note="autonomy/recovery proof",
        )
    if schema == "hawking.future.autonomy_scars.v1":
        n, n_laws = _scar_counts(doc, str(schema))
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=n,
            information_source="receipt n_scars (count of recorded defects)",
            transfer_value=n_laws,
            transfer_source="count of scars that state a law, else general_law",
            reward_note="negative science",
        )
    if schema == "hawking.future.campaign_scars.v1":
        n, n_laws = _scar_counts(doc, str(schema))
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=n,
            information_source="receipt n_scars (campaign defects)",
            transfer_value=n_laws,
            transfer_source="receipt general_law",
            reward_note="negative science",
        )
    if schema == "hawking.future.complete_ebpw.v1":
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=1,
            information_source="one complete-executable-BPW calculator law",
            transfer_value=1,
            transfer_source="refuse-missing-input doctrine reusable as a cost law",
            reward_note="a new compiler law / representation",
        )
    if schema == "hawking.future.device_compiler.v1":
        lowering = doc.get("lowering")
        if not isinstance(lowering, Mapping):
            raise ValueRefused("device_compiler receipt has no lowering object")
        n_compiled = int(lowering.get("n_compiled") or 0)
        if n_compiled < 1:
            raise ValueRefused("device_compiler lowering.n_compiled is not positive")
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=n_compiled,
            information_source="lowering.n_compiled (organs with compiled identity)",
            transfer_value=1,
            transfer_source="placeholder compiled identity is refused, not recorded as COMPILED",
            reward_note="new compiler pass",
        )
    if schema == "hawking.future.repro_science.v1":
        canaries = doc.get("mutation_canaries")
        if not isinstance(canaries, Mapping) or not canaries:
            raise ValueRefused("repro_science receipt has no mutation_canaries")
        trans = (doc.get("claim_downgrade") or {}).get("transitivity_holds")
        if trans is not True:
            raise ValueRefused("repro_science claim_downgrade.transitivity_holds is not True")
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=len(canaries),
            information_source="mutation_canaries keys (fail-closed regression probes)",
            transfer_value=1,
            transfer_source="claim_downgrade.transitivity_holds",
            reward_note="regression / fail-closed reproduction",
        )
    if schema == "hawking.future.kernel_geometry.v1":
        occ = doc.get("occupancy_class")
        organs = doc.get("organs")
        if not isinstance(occ, Mapping):
            raise ValueRefused("kernel_geometry receipt has no occupancy_class")
        if not isinstance(organs, list) or not organs:
            raise ValueRefused("kernel_geometry receipt has no organs")
        tpr = int(occ["threads_per_row"])
        rpt = int(occ["rows_per_tg"])
        tg = int(occ["threadgroup"])
        if tpr * rpt != tg:
            raise ValueRefused(
                f"occupancy identity broken: threads_per_row {tpr} * rows_per_tg "
                f"{rpt} != threadgroup {tg}"
            )
        stride = int(occ["col_stride"])
        covered = 0
        for organ in organs:
            cov = organ.get("coverage") if isinstance(organ, Mapping) else None
            if not isinstance(cov, Mapping):
                continue
            if int(cov.get("trips") or 0) * stride == int(organ.get("cols") or -1):
                if int(cov.get("dropped") or 0) == 0:
                    covered += 1
        if covered < 1:
            raise ValueRefused("no organ has trips*col_stride==cols with dropped==0")
        return _schedule(
            path,
            kind=kind,
            schema=str(schema),
            information_gain=covered,
            information_source=(
                "organs whose coverage.trips * occupancy_class.col_stride == cols "
                "and dropped==0; STATIC arithmetic, not a hardware occupancy counter"
            ),
            transfer_value=1,
            transfer_source=(
                "occupancy_class.threadgroup == threads_per_row * rows_per_tg"
            ),
            reward_note="kernel win (static geometry identity)",
        )
    raise ValueRefused(f"no grounded H.1 mapping for schema {schema!r}")


def bounty_from_receipt(
    path: Path, kind: SelfBountyKind | None = None
) -> tuple[Bounty, SelfBountyKind, dict[str, Any]]:
    artifact = local_artifact(str(path))
    doc = json.loads(artifact.read_text())
    if not isinstance(doc, dict):
        raise IntakeRefused(f"{artifact} is not a JSON object")
    resolved = classify(doc, kind)
    schema = str(doc.get("schema"))
    question = (
        doc.get("purpose")
        or doc.get("question")
        or doc.get("obligation")
        or doc.get("angle")
        or artifact.name
    )
    suffix = f":{resolved.value}" if kind is not None and kind is not SCHEMA_KIND.get(schema) else ""
    bounty = make_internal_bounty(
        id=f"self:{artifact.stem}:{schema}{suffix}",
        source=str(artifact.resolve()),
        domain="hawking",
        question_or_target=str(question),
        nonmonetary_value=resolved.value,
        bounty_class=BountyClass.HAWKING_INTERNAL_SELF_BOUNTY,
        lab=LabKind.HAWKING_SELF_BOUNTY.value,
    )
    return bounty, resolved, doc


def independent_self_bounty_checks(path: Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Call the producer module's own symbols. Import is not a check."""
    schema = doc.get("schema")
    if schema == "hawking.future.autonomy_scars.v1":
        return _verify_autonomy_scars(doc)
    if schema == "hawking.future.campaign_scars.v1":
        return _verify_campaign_scars(doc)
    if schema == "hawking.future.complete_ebpw.v1":
        return _verify_complete_ebpw(doc)
    if schema == "hawking.future.device_compiler.v1":
        return _verify_device_compiler(doc)
    if schema == "hawking.future.repro_science.v1":
        return _verify_repro_science(doc)
    if schema == "hawking.future.kernel_geometry.v1":
        return _verify_kernel_geometry(doc)
    del path
    return {}


def _verify_autonomy_scars(doc: Mapping[str, Any]) -> dict[str, Any]:
    scars = doc.get("scars")
    if not isinstance(scars, list):
        raise VerificationFailed("autonomy_scars receipt has no scars list")
    n = doc.get("n_scars")
    if n != len(scars):
        raise VerificationFailed(f"n_scars {n} != len(scars) {len(scars)}")
    receipt_ids = [s.get("id") for s in scars]
    from tools.future import autonomy_scars as asc

    module_ids = [s["id"] for s in asc.scars()]
    if receipt_ids != module_ids:
        raise VerificationFailed(
            "receipt scar ids disagree with tools.future.autonomy_scars.scars()"
        )
    return {
        "n_scars": n,
        "scar_ids": receipt_ids,
        "independent_module": "tools.future.autonomy_scars.scars",
    }


def _verify_campaign_scars(doc: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.campaign_scars import scar_ids as module_scar_ids, scars as module_scars

    receipt_ids = list(doc.get("scar_ids") or [s.get("id") for s in doc.get("scars") or []])
    live = module_scar_ids()
    if receipt_ids != live:
        raise VerificationFailed(
            "campaign_scars scar_ids disagree with tools.future.campaign_scars.scar_ids()"
        )
    if len(module_scars()) != int(doc.get("n_scars") or -1):
        raise VerificationFailed("campaign_scars n_scars disagrees with scars()")
    return {
        "n_scars": doc.get("n_scars"),
        "scar_ids": live,
        "independent_module": "tools.future.campaign_scars.scar_ids",
    }


def _verify_complete_ebpw(doc: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.complete_ebpw import cost, incumbent_candidate

    incumbent = doc.get("incumbent")
    if not isinstance(incumbent, Mapping):
        raise VerificationFailed("complete_ebpw receipt has no incumbent")
    parts = incumbent.get("parts")
    stored = incumbent.get("stored_bytes")
    if not isinstance(parts, list) or stored is None:
        raise VerificationFailed("incumbent missing parts/stored_bytes")
    parts_bytes = sum(int(p["bytes"]) for p in parts)
    if parts_bytes != int(stored):
        raise VerificationFailed(
            f"incumbent parts sum {parts_bytes} != stored_bytes {stored}"
        )
    live = cost(incumbent_candidate())
    claimed = float(incumbent["complete_ebpw"])
    got = float(live["complete_ebpw"])
    if abs(claimed - got) > 1e-9:
        raise VerificationFailed(
            f"complete_ebpw.cost() {got} != receipt incumbent {claimed}"
        )
    if live.get("reconciled") is not True:
        raise VerificationFailed("complete_ebpw.cost() did not reconcile")
    return {
        "complete_ebpw": got,
        "parts_bytes": parts_bytes,
        "independent_module": "tools.future.complete_ebpw.cost",
    }


def _verify_device_compiler(doc: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.device_compiler import (
        PlaceholderCompiledIdentity,
        is_genuine_compiled_identity,
        qwen3_dense_gguf_blocker,
        refuse_placeholder,
    )

    lowering = doc.get("lowering") or {}
    plan = lowering.get("plan") or []
    genuine = 0
    for slot in plan:
        identity = (slot or {}).get("compiled_identity")
        if identity is None:
            continue
        if not is_genuine_compiled_identity(identity):
            raise VerificationFailed(
                f"compiled_identity for organ {slot.get('organ')!r} failed "
                "is_genuine_compiled_identity"
            )
        genuine += 1
    if genuine < 1:
        raise VerificationFailed("no genuine compiled_identity on the device_compiler plan")
    refused = False
    try:
        refuse_placeholder({"kind": "PLACEHOLDER", "shader_hash": "0" * 64})
    except PlaceholderCompiledIdentity:
        refused = True
    if not refused:
        raise VerificationFailed("refuse_placeholder accepted a PLACEHOLDER identity")
    blocker = doc.get("qwen3_dense_gguf_blocker") or {}
    live_blocker = qwen3_dense_gguf_blocker(
        blocker.get("architectures"),
        family=blocker.get("family"),
        model_type=blocker.get("model_type"),
    )
    if live_blocker.get("id") != blocker.get("id"):
        raise VerificationFailed("qwen3_dense_gguf_blocker id disagrees with the module")
    if live_blocker.get("includes_qwen3_dense") is not False:
        raise VerificationFailed("qwen3 dense arm is not supposed to be present")
    return {
        "genuine_compiled_identities": genuine,
        "placeholder_refused": True,
        "independent_module": "tools.future.device_compiler.is_genuine_compiled_identity",
        "blocker_symbol": "tools.future.device_compiler.qwen3_dense_gguf_blocker",
        "refuse_symbol": "tools.future.device_compiler.refuse_placeholder",
    }


def _verify_repro_science(doc: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.repro_science import transitive_downgrade_proof

    live = transitive_downgrade_proof()
    claimed = (doc.get("claim_downgrade") or {}).get("after")
    if live.get("transitivity_holds") is not True:
        raise VerificationFailed("transitive_downgrade_proof().transitivity_holds is not True")
    if claimed != live.get("after"):
        raise VerificationFailed(
            "repro_science claim_downgrade.after disagrees with "
            "tools.future.repro_science.transitive_downgrade_proof"
        )
    return {
        "transitivity_holds": True,
        "after": live.get("after"),
        "independent_module": "tools.future.repro_science.transitive_downgrade_proof",
    }


def _verify_kernel_geometry(doc: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.static_kernel_verify import APPLE_MAX_THREADS_PER_THREADGROUP

    occ = doc.get("occupancy_class") or {}
    tpr = int(occ["threads_per_row"])
    rpt = int(occ["rows_per_tg"])
    tg = int(occ["threadgroup"])
    if tpr * rpt != tg:
        raise VerificationFailed(
            f"occupancy identity: {tpr}*{rpt} != threadgroup {tg}"
        )
    if tg > APPLE_MAX_THREADS_PER_THREADGROUP:
        raise VerificationFailed(
            f"threadgroup {tg} exceeds APPLE_MAX_THREADS_PER_THREADGROUP "
            f"{APPLE_MAX_THREADS_PER_THREADGROUP}"
        )
    stride = int(occ["col_stride"])
    held = []
    for organ in doc.get("organs") or []:
        cov = organ.get("coverage") or {}
        cols = int(organ["cols"])
        trips = int(cov["trips"])
        if trips * stride != cols:
            raise VerificationFailed(
                f"{organ.get('role')}: trips {trips} * col_stride {stride} != cols {cols}"
            )
        if int(cov.get("dropped") or 0) != 0:
            raise VerificationFailed(f"{organ.get('role')}: coverage.dropped is not 0")
        held.append(organ.get("role"))
    return {
        "occupancy_identity": f"{tpr}*{rpt}=={tg}",
        "organs_covered": held,
        "apple_ceiling": APPLE_MAX_THREADS_PER_THREADGROUP,
        "independent_module": "tools.future.static_kernel_verify.APPLE_MAX_THREADS_PER_THREADGROUP",
        "evidence_tier": "STATIC",
    }
