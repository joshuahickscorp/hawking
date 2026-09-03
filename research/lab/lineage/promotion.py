"""External, mechanical Genesis promotion gate.

A child is promoted only if every clause holds. Parent and child are
refused if they invoke the gate on themselves. Missing evidence is
PENDING (never a fabricated ACCEPT). Present-but-wrong evidence is
REJECT. A missing required field on present evidence is FAIL, never a
skip. Corrupt the input of any clause and it goes red.

The gate hashes artifact preimages and state-transfer payloads. It never
trusts an asserted checksum, an asserted child_tps, a lowered parent
contract, or a paperwork-only PASS string.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from lab.lineage.canon import bpw_key, require_mapping, utc_now
from lab.lineage.identity import (
    EXTERNAL_PRINCIPALS,
    GenesisInstance,
    Invoker,
    SELF_PRINCIPALS,
    as_instance,
    as_invoker,
)
from lab.lineage.transfer import TransferError, payload_checksum
from lab.receipts import seal

if TYPE_CHECKING:
    from lab.lineage.state import LineageState

PROMOTION_SCHEMA = "hawking.lineage.promotion_gate.v1"

# 1% of the parent complete-token wall. A 1 ns "win" is not material.
MATERIAL_COMPLETE_TOKEN_FRACTION = 0.01
MIN_PAIRED_REPS = 3
MIN_GREEDY_PROMPTS = 3
NS_PER_SECOND = 1_000_000_000.0

CLAUSE_CAPABILITY = "capability_ge_parent_contract"
CLAUSE_NO_NEW_SILENT_FALLBACK = "no_new_silent_fallback"
CLAUSE_COMPLETE_TOKEN_MATERIAL = "complete_token_wall_improves_materially"
CLAUSE_ARTIFACT_IDENTITY = "artifact_identity_exact"
CLAUSE_REPRESENTATION_BPW = "representation_bpw_exact"
CLAUSE_RUNTIME_GENOME = "runtime_and_kernel_genome_exact"
CLAUSE_PROTECTED_TESTS = "protected_tests_pass"
CLAUSE_STATE_TRANSFER = "state_transfer_test_passes"
CLAUSE_ROLLBACK_ARTIFACT = "rollback_artifact_exists"
CLAUSE_TPS_UP_CAP_DOWN = "reject_tps_up_capability_down"
CLAUSE_BPW_UP_TOKEN_DOWN = "reject_bpw_improved_token_worse"
CLAUSE_BENCHMARK_UNCHANGED = "reject_benchmark_changed"
CLAUSE_GENERATION = "generation_strictly_increases"
CLAUSE_MODEL_IDENTITY = "identity_model_matches_parent"
CLAUSE_GREEDY_TOKEN_IDS = "greedy_token_ids_agree"

ALL_CLAUSES: tuple[str, ...] = (
    CLAUSE_CAPABILITY,
    CLAUSE_NO_NEW_SILENT_FALLBACK,
    CLAUSE_COMPLETE_TOKEN_MATERIAL,
    CLAUSE_ARTIFACT_IDENTITY,
    CLAUSE_REPRESENTATION_BPW,
    CLAUSE_RUNTIME_GENOME,
    CLAUSE_PROTECTED_TESTS,
    CLAUSE_STATE_TRANSFER,
    CLAUSE_ROLLBACK_ARTIFACT,
    CLAUSE_TPS_UP_CAP_DOWN,
    CLAUSE_BPW_UP_TOKEN_DOWN,
    CLAUSE_BENCHMARK_UNCHANGED,
    CLAUSE_GENERATION,
    CLAUSE_MODEL_IDENTITY,
    CLAUSE_GREEDY_TOKEN_IDS,
)

REQUIRED_PROTECTED_TESTS: tuple[str, ...] = (
    "coherence_greedy_ids",
    "complete_token_ledger_closed",
    "no_silent_fallback",
)

_SKIP_STATUSES = frozenset({"SKIP", "SKIPPED", "XFAIL", "XPASS"})


class PromotionError(ValueError):
    """Promotion gate cannot evaluate the request."""


class SelfCertificationRefused(PermissionError):
    """Parent or child tried to invoke the gate on itself."""


def refuse_self_certification(
    invoker: Invoker | Mapping[str, Any],
    parent: GenesisInstance | Mapping[str, Any],
    child: GenesisInstance | Mapping[str, Any],
) -> Invoker:
    inv = as_invoker(invoker)
    parent_i = as_instance(parent, "parent")
    child_i = as_instance(child, "child")
    forbidden_ids = {
        parent_i.instance_id,
        child_i.instance_id,
        "CURRENT",
        "CANDIDATE",
        "parent",
        "child",
    }
    if inv.principal in SELF_PRINCIPALS:
        raise SelfCertificationRefused(
            f"self-certification refused: principal={inv.principal!r} "
            "is not an external promotion authority"
        )
    if inv.principal not in EXTERNAL_PRINCIPALS:
        raise SelfCertificationRefused(
            f"self-certification refused: principal={inv.principal!r} "
            f"is not one of {sorted(EXTERNAL_PRINCIPALS)}"
        )
    if inv.identity in forbidden_ids:
        raise SelfCertificationRefused(
            f"self-certification refused: invoker identity {inv.identity!r} "
            "is the parent or the child"
        )
    acting = (inv.acting_as or "external").lower()
    if acting in SELF_PRINCIPALS or acting in {"parent", "child", "current", "candidate"}:
        raise SelfCertificationRefused(
            f"self-certification refused: acting_as={inv.acting_as!r}"
        )
    return inv


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _capability_losses(
    *,
    contract: Mapping[str, float],
    parent_cap: Mapping[str, float],
    child_cap: Mapping[str, float],
) -> tuple[list[str], list[str], list[str]]:
    missing = [axis for axis in contract if axis not in child_cap]
    below_contract = [
        axis
        for axis, floor in contract.items()
        if axis in child_cap and float(child_cap[axis]) < float(floor)
    ]
    below_parent = [
        axis
        for axis, parent_v in parent_cap.items()
        if axis in child_cap and float(child_cap[axis]) < float(parent_v)
    ]
    return missing, below_contract, below_parent


def _lowered_contract_axes(
    parent_cap: Mapping[str, float],
    evidence_contract: Mapping[str, Any] | None,
) -> list[str]:
    """Axes whose evidence floor was dropped or set below the parent's measured value."""
    if not isinstance(evidence_contract, Mapping):
        return []
    lowered: list[str] = []
    for axis, parent_v in parent_cap.items():
        if axis not in evidence_contract:
            lowered.append(axis)
            continue
        try:
            floor = float(evidence_contract[axis])
        except (TypeError, ValueError):
            lowered.append(axis)
            continue
        if floor < float(parent_v):
            lowered.append(axis)
    return lowered


def _preimage_bytes(value: object) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str) and value:
        return value.encode("utf-8")
    return None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _computed_artifact_sha(receipt: Mapping[str, Any]) -> str | None:
    preimage = receipt.get("preimage")
    if preimage is None:
        preimage = receipt.get("bytes")
    data = _preimage_bytes(preimage)
    if data is None:
        return None
    return _sha256_hex(data)


def _tps_from_wall_ns(wall_ns: float) -> float:
    return NS_PER_SECOND / float(wall_ns)


def _authority_text(value: object) -> str:
    return str(value or "").strip()


def _is_cpu_wait_proxy(text: str) -> bool:
    low = text.lower()
    return "cpu" in low and "proxy" in low


def _int_token_ids(value: object, name: str) -> list[int] | str:
    if not isinstance(value, (list, tuple)) or not value:
        return f"{name} must be a non-empty sequence of token ids"
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return f"{name} must be integer token ids, got {item!r}"
        out.append(item)
    return out


def _greedy_rows(evidence: Mapping[str, Any], tests: object) -> list[Any] | None:
    raw = evidence.get("greedy_token_ids")
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(tests, (list, tuple)):
        for row in tests:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("name")) != "coherence_greedy_ids":
                continue
            for key in ("prompts", "pairs", "token_ids", "greedy_token_ids"):
                if isinstance(row.get(key), (list, tuple)):
                    return list(row[key])
    return None


def evaluate_promotion(
    *,
    parent: GenesisInstance | Mapping[str, Any],
    child: GenesisInstance | Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    invoker: Invoker | Mapping[str, Any],
    lineage: LineageState | None = None,
) -> dict[str, Any]:
    """Mechanical gate. Raises on self-cert. Never fabricates ACCEPT."""
    parent_i = as_instance(parent, "parent")
    child_i = as_instance(child, "child")
    inv = refuse_self_certification(invoker, parent_i, child_i)
    ev = dict(evidence or {})
    contract_block = ev.get("parent_contract") if isinstance(ev.get("parent_contract"), Mapping) else {}
    evidence_cap = (
        contract_block.get("capability")
        if isinstance(contract_block.get("capability"), Mapping)
        else None
    )
    # Child is compared against the PARENT. Evidence may not lower that floor.
    missing, below_contract, below_parent = _capability_losses(
        contract=parent_i.capability,
        parent_cap=parent_i.capability,
        child_cap=child_i.capability,
    )
    lowered_floor = _lowered_contract_axes(parent_i.capability, evidence_cap)
    checks: list[dict[str, str]] = []

    if lowered_floor:
        checks.append(
            _check(
                CLAUSE_CAPABILITY,
                "FAIL",
                f"parent contract floor lowered below parent measured capability: {lowered_floor}",
            )
        )
    elif missing or below_parent:
        checks.append(
            _check(
                CLAUSE_CAPABILITY,
                "FAIL",
                f"capability below parent: missing={missing} below={below_parent}",
            )
        )
    else:
        checks.append(
            _check(CLAUSE_CAPABILITY, "PASS", "capability >= parent on every required axis")
        )

    parent_fb = set(parent_i.silent_fallback_ids)
    child_fb = set(child_i.silent_fallback_ids)
    declared_new = ev.get("new_silent_fallbacks") or []
    if not isinstance(declared_new, (list, tuple)):
        declared_new = [declared_new]
    new_fb = sorted(set(child_fb - parent_fb) | {str(x) for x in declared_new if str(x)})
    if new_fb:
        checks.append(
            _check(
                CLAUSE_NO_NEW_SILENT_FALLBACK,
                "FAIL",
                f"new silent fallback(s): {new_fb}",
            )
        )
    else:
        checks.append(_check(CLAUSE_NO_NEW_SILENT_FALLBACK, "PASS", "no new silent fallback"))

    meas = ev.get("measurement") if isinstance(ev.get("measurement"), Mapping) else None
    child_wall: float | None = None
    parent_wall: float | None = None
    if meas is None:
        checks.append(
            _check(
                CLAUSE_COMPLETE_TOKEN_MATERIAL,
                "PENDING",
                "complete-token measurement not submitted (PENDING, never fabricate ACCEPT)",
            )
        )
    else:
        reps_c = meas.get("complete_token_ns_reps")
        reps_p = meas.get("parent_complete_token_ns_reps")
        regime = meas.get("regime")
        parent_auth = _authority_text(meas.get("parent_timing_authority"))
        child_auth = _authority_text(
            meas.get("child_timing_authority") or meas.get("timing_authority")
        )
        offered_auths = [
            _authority_text(meas.get(key))
            for key in ("timing_authority", "child_timing_authority", "parent_timing_authority")
        ]
        if not isinstance(reps_c, (list, tuple)) or len(reps_c) < MIN_PAIRED_REPS:
            checks.append(
                _check(
                    CLAUSE_COMPLETE_TOKEN_MATERIAL,
                    "FAIL",
                    f"paired measurement requires >= {MIN_PAIRED_REPS} child reps; got {reps_c!r}",
                )
            )
        elif not isinstance(reps_p, (list, tuple)) or len(reps_p) < MIN_PAIRED_REPS:
            checks.append(
                _check(
                    CLAUSE_COMPLETE_TOKEN_MATERIAL,
                    "FAIL",
                    f"paired measurement requires >= {MIN_PAIRED_REPS} parent reps; got {reps_p!r}",
                )
            )
        elif not regime:
            checks.append(
                _check(
                    CLAUSE_COMPLETE_TOKEN_MATERIAL,
                    "FAIL",
                    "measurement.regime (warm/cold) must be stated",
                )
            )
        elif any(_is_cpu_wait_proxy(text) for text in offered_auths if text):
            checks.append(
                _check(
                    CLAUSE_COMPLETE_TOKEN_MATERIAL,
                    "FAIL",
                    "GPU timing authority must be MTLCommandBuffer GPUStartTime/GPUEndTime "
                    "after wait; CPU-wait proxy refused",
                )
            )
        elif not parent_auth or not child_auth:
            checks.append(
                _check(
                    CLAUSE_COMPLETE_TOKEN_MATERIAL,
                    "FAIL",
                    "timing_authority required on both sides; missing is FAIL, never skip",
                )
            )
        elif parent_auth != child_auth:
            checks.append(
                _check(
                    CLAUSE_COMPLETE_TOKEN_MATERIAL,
                    "FAIL",
                    f"cross-authority comparison refused: parent={parent_auth!r} child={child_auth!r}",
                )
            )
        else:
            try:
                child_vals = [float(x) for x in reps_c]
                parent_vals = [float(x) for x in reps_p]
            except (TypeError, ValueError):
                checks.append(
                    _check(CLAUSE_COMPLETE_TOKEN_MATERIAL, "FAIL", "complete-token reps must be numeric")
                )
            else:
                child_wall = _mean(child_vals)
                parent_wall = _mean(parent_vals)
                need = parent_wall * MATERIAL_COMPLETE_TOKEN_FRACTION
                delta = parent_wall - child_wall
                spread = {
                    "parent": {"n": len(parent_vals), "mean": parent_wall, "min": min(parent_vals), "max": max(parent_vals)},
                    "child": {"n": len(child_vals), "mean": child_wall, "min": min(child_vals), "max": max(child_vals)},
                    "delta_ns": delta,
                    "material_ns": need,
                    "regime": regime,
                    "timing_authority": child_auth,
                }
                if child_wall >= parent_wall:
                    checks.append(
                        _check(
                            CLAUSE_COMPLETE_TOKEN_MATERIAL,
                            "FAIL",
                            f"complete-token wall did not improve: child_mean={child_wall} "
                            f"parent_mean={parent_wall} spread={spread}",
                        )
                    )
                elif delta < need:
                    checks.append(
                        _check(
                            CLAUSE_COMPLETE_TOKEN_MATERIAL,
                            "FAIL",
                            f"complete-token improvement is not material: delta={delta} "
                            f"< {need} ({MATERIAL_COMPLETE_TOKEN_FRACTION:.0%} of parent) spread={spread}",
                        )
                    )
                else:
                    checks.append(
                        _check(
                            CLAUSE_COMPLETE_TOKEN_MATERIAL,
                            "PASS",
                            f"complete-token improved materially: delta={delta} ns spread={spread}",
                        )
                    )

    receipt = ev.get("artifact_receipt") if isinstance(ev.get("artifact_receipt"), Mapping) else None
    art = receipt.get("sha") if receipt else None
    meas_art = meas.get("artifact_sha") if meas else None
    computed_art = _computed_artifact_sha(receipt) if receipt else None
    if meas is None and receipt is None:
        checks.append(
            _check(
                CLAUSE_ARTIFACT_IDENTITY,
                "PENDING",
                "artifact identity requires child.artifact_sha == measurement.artifact_sha "
                "== artifact_receipt.sha",
            )
        )
    elif not art or not meas_art:
        checks.append(
            _check(
                CLAUSE_ARTIFACT_IDENTITY,
                "FAIL",
                "artifact identity requires child.artifact_sha == measurement.artifact_sha "
                "== artifact_receipt.sha",
            )
        )
    elif computed_art is None:
        checks.append(
            _check(
                CLAUSE_ARTIFACT_IDENTITY,
                "FAIL",
                "artifact hash must be computed from a preimage, not asserted",
            )
        )
    elif (
        computed_art != art
        or computed_art != child_i.artifact_sha
        or meas_art != child_i.artifact_sha
    ):
        checks.append(
            _check(
                CLAUSE_ARTIFACT_IDENTITY,
                "FAIL",
                f"artifact identity mismatch: child={child_i.artifact_sha} "
                f"measurement={meas_art} receipt={art} computed={computed_art}",
            )
        )
    else:
        checks.append(_check(CLAUSE_ARTIFACT_IDENTITY, "PASS", "artifact identity exact (sha256 computed)"))

    rep = ev.get("representation") if isinstance(ev.get("representation"), Mapping) else None
    if rep is None or rep.get("bpw") is None:
        checks.append(_check(CLAUSE_REPRESENTATION_BPW, "PENDING", "representation/BPW receipt not submitted"))
    else:
        try:
            same = bpw_key(rep.get("bpw")) == bpw_key(child_i.representation_bpw)
        except ValueError as exc:
            checks.append(_check(CLAUSE_REPRESENTATION_BPW, "FAIL", str(exc)))
        else:
            if not same:
                checks.append(
                    _check(
                        CLAUSE_REPRESENTATION_BPW,
                        "FAIL",
                        f"representation/BPW mismatch: child={child_i.representation_bpw} "
                        f"receipt={rep.get('bpw')}",
                    )
                )
            else:
                checks.append(_check(CLAUSE_REPRESENTATION_BPW, "PASS", "representation/BPW exact"))

    genome = ev.get("genome") if isinstance(ev.get("genome"), Mapping) else None
    if genome is None or not genome.get("runtime_sha") or not genome.get("kernel_genome_sha"):
        checks.append(
            _check(CLAUSE_RUNTIME_GENOME, "PENDING", "runtime/kernel genome receipt not submitted")
        )
    elif (
        genome.get("runtime_sha") != child_i.runtime_sha
        or genome.get("kernel_genome_sha") != child_i.kernel_genome_sha
    ):
        checks.append(
            _check(
                CLAUSE_RUNTIME_GENOME,
                "FAIL",
                f"runtime/kernel genome mismatch: child runtime={child_i.runtime_sha} "
                f"kernel={child_i.kernel_genome_sha} receipt={dict(genome)}",
            )
        )
    else:
        checks.append(_check(CLAUSE_RUNTIME_GENOME, "PASS", "runtime and kernel genome exact"))

    tests = ev.get("protected_tests")
    if not isinstance(tests, (list, tuple)) or not tests:
        checks.append(_check(CLAUSE_PROTECTED_TESTS, "PENDING", "protected test results not submitted"))
    else:
        names = {str(row.get("name")) for row in tests if isinstance(row, Mapping)}
        missing_tests = [name for name in REQUIRED_PROTECTED_TESTS if name not in names]
        statuses = [
            str(row.get("status") or "").upper()
            for row in tests
            if isinstance(row, Mapping)
        ]
        skipped = [s for s in statuses if s in _SKIP_STATUSES]
        failed = [s for s in statuses if s == "FAIL"]
        if skipped:
            checks.append(
                _check(
                    CLAUSE_PROTECTED_TESTS,
                    "FAIL",
                    f"skipped protected test is a FAILURE, never a pass: {skipped}",
                )
            )
        elif failed:
            checks.append(_check(CLAUSE_PROTECTED_TESTS, "FAIL", "one or more protected tests FAILED"))
        elif missing_tests:
            checks.append(
                _check(
                    CLAUSE_PROTECTED_TESTS,
                    "FAIL",
                    f"required protected tests missing: {missing_tests}",
                )
            )
        elif statuses and all(s == "PASS" for s in statuses):
            checks.append(_check(CLAUSE_PROTECTED_TESTS, "PASS", "protected tests passed"))
        else:
            checks.append(
                _check(CLAUSE_PROTECTED_TESTS, "FAIL", f"protected test statuses not all PASS: {statuses}")
            )

    st = ev.get("state_transfer") if isinstance(ev.get("state_transfer"), Mapping) else None
    if st is None:
        checks.append(_check(CLAUSE_STATE_TRANSFER, "PENDING", "state-transfer test not submitted"))
    else:
        payload = st.get("payload")
        claimed = st.get("checksum_sha256")
        if not isinstance(payload, Mapping) or not claimed:
            checks.append(
                _check(
                    CLAUSE_STATE_TRANSFER,
                    "FAIL",
                    "state-transfer checksum must be computed from a payload; "
                    "checksum_verified assertions are untrusted",
                )
            )
        else:
            try:
                computed_xfer = payload_checksum(payload)
            except (TransferError, ValueError, TypeError) as exc:
                checks.append(
                    _check(
                        CLAUSE_STATE_TRANSFER,
                        "FAIL",
                        f"state-transfer payload refused: {exc}",
                    )
                )
            else:
                if computed_xfer != claimed:
                    checks.append(
                        _check(
                            CLAUSE_STATE_TRANSFER,
                            "FAIL",
                            f"state-transfer checksum mismatch: claimed={claimed!r} "
                            f"computed={computed_xfer}",
                        )
                    )
                else:
                    checks.append(
                        _check(
                            CLAUSE_STATE_TRANSFER,
                            "PASS",
                            "state-transfer checksum recomputed and matched the payload",
                        )
                    )

    rollback_ok = False
    rollback_detail = "no rollback artifact"
    if lineage is not None:
        lkg = lineage.last_known_good
        if lkg is not None and lkg.valid:
            rollback_ok = True
            rollback_detail = f"LAST_KNOWN_GOOD={lkg.instance_id}"
        else:
            rollback_detail = "lineage LAST_KNOWN_GOOD missing or invalid"
    elif isinstance(ev.get("rollback_artifact"), Mapping):
        if ev["rollback_artifact"].get("valid") is True and ev["rollback_artifact"].get("instance_id"):
            rollback_ok = True
            rollback_detail = f"rollback_artifact={ev['rollback_artifact'].get('instance_id')}"
        else:
            rollback_detail = f"rollback_artifact present but not valid: {dict(ev['rollback_artifact'])}"
    checks.append(
        _check(
            CLAUSE_ROLLBACK_ARTIFACT,
            "PASS" if rollback_ok else "FAIL",
            rollback_detail,
        )
    )

    # child_tps is derived from the wall reps (or the instance complete-token
    # wall). Evidence child_tps and parent_contract.tps are never trusted.
    parent_tps = _tps_from_wall_ns(parent_wall) if parent_wall is not None else parent_i.tps
    child_tps = _tps_from_wall_ns(child_wall) if child_wall is not None else child_i.tps
    cap_lost = bool(missing or below_parent)
    if child_tps > parent_tps and cap_lost:
        checks.append(
            _check(
                CLAUSE_TPS_UP_CAP_DOWN,
                "FAIL",
                f"child improves TPS ({child_tps:.4f} > {parent_tps:.4f}) but loses required "
                f"capability (missing={missing} below_contract={below_contract} "
                f"below_parent={below_parent})",
            )
        )
    else:
        checks.append(
            _check(
                CLAUSE_TPS_UP_CAP_DOWN,
                "PASS",
                "TPS-up / capability-down reject rule did not fire",
            )
        )

    declared_child_wall = float(child_wall) if child_wall is not None else float(child_i.complete_token_ns)
    declared_parent_wall = float(parent_wall) if parent_wall is not None else float(parent_i.complete_token_ns)
    if child_i.representation_bpw < parent_i.representation_bpw and declared_child_wall >= declared_parent_wall:
        checks.append(
            _check(
                CLAUSE_BPW_UP_TOKEN_DOWN,
                "FAIL",
                f"child improves BPW ({child_i.representation_bpw} < {parent_i.representation_bpw}) "
                f"but adds complete-token latency ({declared_child_wall} >= {declared_parent_wall})",
            )
        )
    else:
        checks.append(
            _check(
                CLAUSE_BPW_UP_TOKEN_DOWN,
                "PASS",
                "BPW-improved / token-worse reject rule did not fire",
            )
        )

    ev_fp = meas.get("benchmark_fingerprint") if meas else None
    parent_fp = contract_block.get("benchmark_fingerprint") or parent_i.benchmark_fingerprint
    child_fp = child_i.benchmark_fingerprint
    changed = bool(ev.get("benchmark_changed"))
    if changed or (ev_fp and ev_fp != parent_fp) or child_fp != parent_fp:
        checks.append(
            _check(
                CLAUSE_BENCHMARK_UNCHANGED,
                "FAIL",
                f"child changed the benchmark instead of improving itself: "
                f"parent={parent_fp} child={child_fp} measurement={ev_fp}",
            )
        )
    else:
        checks.append(_check(CLAUSE_BENCHMARK_UNCHANGED, "PASS", "benchmark fingerprint unchanged"))

    if child_i.generation <= parent_i.generation:
        checks.append(
            _check(
                CLAUSE_GENERATION,
                "FAIL",
                f"generation must strictly increase: parent={parent_i.generation} "
                f"child={child_i.generation}",
            )
        )
    else:
        checks.append(
            _check(
                CLAUSE_GENERATION,
                "PASS",
                f"generation increased {parent_i.generation} -> {child_i.generation}",
            )
        )

    parent_model = str((parent_i.identity or {}).get("model") or "").strip()
    child_model = str((child_i.identity or {}).get("model") or "").strip()
    if not parent_model or not child_model:
        checks.append(
            _check(
                CLAUSE_MODEL_IDENTITY,
                "FAIL",
                "identity.model required on parent and child; missing is FAIL",
            )
        )
    elif child_model != parent_model:
        checks.append(
            _check(
                CLAUSE_MODEL_IDENTITY,
                "FAIL",
                f"identity.model mismatch: parent={parent_model!r} child={child_model!r}",
            )
        )
    else:
        checks.append(
            _check(CLAUSE_MODEL_IDENTITY, "PASS", f"identity.model matches parent ({parent_model})")
        )

    tests_block = ev.get("protected_tests")
    greedy = _greedy_rows(ev, tests_block)
    protected_present = isinstance(tests_block, (list, tuple)) and bool(tests_block)
    if greedy is None:
        if protected_present:
            checks.append(
                _check(
                    CLAUSE_GREEDY_TOKEN_IDS,
                    "FAIL",
                    f"greedy token id agreement on >= {MIN_GREEDY_PROMPTS} prompts "
                    "is required evidence, not optional",
                )
            )
        else:
            checks.append(
                _check(
                    CLAUSE_GREEDY_TOKEN_IDS,
                    "PENDING",
                    "greedy token id pairs not submitted",
                )
            )
    else:
        if len(greedy) < MIN_GREEDY_PROMPTS:
            checks.append(
                _check(
                    CLAUSE_GREEDY_TOKEN_IDS,
                    "FAIL",
                    f"greedy token id agreement requires >= {MIN_GREEDY_PROMPTS} prompts; "
                    f"got {len(greedy)}",
                )
            )
        else:
            prompts: list[str] = []
            mismatch = None
            invalid = None
            for idx, row in enumerate(greedy):
                if not isinstance(row, Mapping):
                    invalid = f"greedy row {idx} is not a mapping"
                    break
                prompt = str(row.get("prompt") or "").strip()
                if not prompt:
                    invalid = f"greedy row {idx} missing prompt"
                    break
                if prompt in prompts:
                    invalid = f"greedy prompts must be distinct; repeated {prompt!r}"
                    break
                parent_ids = _int_token_ids(row.get("parent_ids"), f"greedy[{idx}].parent_ids")
                child_ids = _int_token_ids(row.get("child_ids"), f"greedy[{idx}].child_ids")
                if isinstance(parent_ids, str):
                    invalid = parent_ids
                    break
                if isinstance(child_ids, str):
                    invalid = child_ids
                    break
                if parent_ids != child_ids:
                    mismatch = (
                        f"greedy token ids disagree on prompt {prompt!r}: "
                        f"parent={parent_ids} child={child_ids}"
                    )
                    break
                prompts.append(prompt)
            if invalid:
                checks.append(_check(CLAUSE_GREEDY_TOKEN_IDS, "FAIL", invalid))
            elif mismatch:
                checks.append(_check(CLAUSE_GREEDY_TOKEN_IDS, "FAIL", mismatch))
            else:
                checks.append(
                    _check(
                        CLAUSE_GREEDY_TOKEN_IDS,
                        "PASS",
                        f"greedy token ids agree on {len(prompts)} prompts",
                    )
                )

    present = {c["name"] for c in checks}
    for name in ALL_CLAUSES:
        if name not in present:
            raise PromotionError(f"internal error: clause {name} was not evaluated")

    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        overall = "REJECT"
        reason = "one or more promotion-gate clauses FAILED"
        authority = "authoritative"
    elif statuses <= {"PASS"}:
        overall = "ACCEPT"
        reason = "all promotion-gate clauses PASSED"
        authority = "authoritative"
    else:
        overall = "PENDING"
        reason = "promotion evidence incomplete; PENDING honestly, never a fabricated ACCEPT"
        authority = "pending"

    document = {
        "schema": PROMOTION_SCHEMA,
        "authority_level": authority,
        "verdict": overall,
        "reason": reason,
        "fabricated_accept": False,
        "invoker": inv.to_dict(),
        "parent_id": parent_i.instance_id,
        "child_id": child_i.instance_id,
        "checks": checks,
        "clauses": list(ALL_CLAUSES),
        "material_complete_token_fraction": MATERIAL_COMPLETE_TOKEN_FRACTION,
        "recorded_at": utc_now(),
    }
    return seal(document)


def clause_status(verdict: Mapping[str, Any], name: str) -> str:
    require_mapping(verdict, "verdict")
    for row in verdict.get("checks") or []:
        if isinstance(row, Mapping) and row.get("name") == name:
            return str(row.get("status") or "")
    raise PromotionError(f"clause {name!r} missing from verdict")
