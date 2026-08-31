"""ODYSSEY II — transfer a scoped campaign law onto a different sealed specimen.

Odyssey I does not have to finish. The moment specimen A has yielded a scoped
law and specimen B exists, this module tests transfer. It consumes the Odyssey
II law-store field set (ols.Law, promote, transfer_candidates, the sequential
lattice) rather than minting a second authority. A failed transfer narrows
scope; it does not delete the law. A similarity score is not a transfer.

No GPU lease. evidence_class STATIC_ONLY. Hardware numbers are copied from
named campaign receipts or recorded UNMEASURED with the experiment that would
settle them.

    python3 tools/future/odyssey2_transfer.py --build
    python3 tools/future/odyssey2_transfer.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import replace
from typing import Any, Mapping

from tools.future._common import REPO, load_json, write_receipt, _assert_no_hardware_claims
from tools.future import odyssey2_law_store as ols
from tools.future import phase_listeners as pl


RECEIPT = "ODYSSEY2_TRANSFER.json"
SCHEMA = "hawking.future.odyssey2_transfer.v1"
VERSION = 1
RECORDED_BY = "tools/future/odyssey2_transfer.py"

TRANSFER_HELD = "TRANSFER_HELD"
TRANSFER_FAILED = "TRANSFER_FAILED"
UNMEASURED = "UNMEASURED"
VERDICTS = (TRANSFER_HELD, TRANSFER_FAILED, UNMEASURED)

SOURCE_SPECIMEN = "qwen3.8-27b-abliterated-bf16@local"
SOURCE_DISPLAY = "qwen3.8-27b sealed-3.14"
SOURCE_MODEL = "Qwen3.8-27B"
SOURCE_FAMILY = "dense_hybrid_transformer"

# Campaign receipts this sidecar is allowed to copy numbers from. Absence of a
# field is UNMEASURED, never an invented substitute.
ALU_REL = "receipts/future/MLP_ALU_ROOFLINE.json"
STREAM_REL = "receipts/future/MLP_STREAM_COUNT.json"
ISSUE_REL = "receipts/future/MLP_ISSUE_RATE_LADDER.json"
AUX_NATIVE_REL = "receipts/future/AUX_U8_NATIVE.json"
AUX_LUT_REL = "receipts/future/AUX_U8_LUT.json"
ECON_REL = "receipts/future/ECONOMICS_CALIBRATION.json"
STRUCT_REL = "receipts/future/MLP_STRUCTURED_OPERATOR.json"
WIDEN_REL = "receipts/future/DELTANET_WIDEN_AB.json"
FOLD_REL = "receipts/future/FOLD_ADDQX_AB.json"
ROOF_REL = "receipts/future/ROOF_ANCHOR.json"
VERIFY_REL = "receipts/future/SPECIMEN_VERIFICATION.json"
XFER_VERIFIED_REL = "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json"
LAW_STORE_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"

# Experiment that would settle a METHOD transfer of L1 onto a new specimen.
L1_METHOD_EXPERIMENT = (
    "crates/hawking-core/examples/alu_roofline_organs.rs ARM A (production "
    "access pattern and byte count, decode+dequant+FMA replaced by XOR/add "
    "sink) vs production on the target specimen's MLP, same process, loads "
    "proven live (stripped time above the zero-load floor and stripped-half "
    "time dropping with bytes). Settle: ARM A jumps vs production on the "
    "same unique payload. Do not copy 497.4 / 329.6 / 83.56 MB."
)


class TransferError(ValueError):
    """Transfer protocol violation."""


class ReplicatingSpecimenRequired(ols.ScopeViolation):
    """Widening a law requires a second specimen that replicated the statement.

    Being sealed is not replication. Being listed as a transfer_candidate is
    not replication. UNMEASURED is not replication.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("reason", "need_replicating_specimen")
        super().__init__(message, **kwargs)


class NotAMeasurementError(TransferError):
    """A similarity score, or any non-measurement, is not a transfer verdict."""


class IdentityTransferError(pl.NotATransferError):
    """A target identical to the law's origin is not a transfer."""


# ---------------------------------------------------------------------------
# Cite: copy a field from a named receipt, or UNMEASURED. Never invent.
# ---------------------------------------------------------------------------


def _dig(node: Any, dotted: str) -> Any:
    cur = node
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def load_named(rel: str) -> tuple[dict[str, Any] | None, str | None]:
    """Worktree, then git HEAD (ols.try_load). Absence is not an error."""
    path = REPO / rel
    if path.is_file():
        try:
            return load_json(path), None
        except (OSError, json.JSONDecodeError) as e:
            return None, f"unreadable {rel}: {e}"
    return ols.try_load(rel)


def cite(rel: str, field: str) -> dict[str, Any]:
    """Copy `field` from `rel`. Missing file/field -> UNMEASURED, never a guess."""
    doc, err = load_named(rel)
    if doc is None:
        return {
            "value": None,
            "source_receipt": rel,
            "source_field": field,
            "copied": False,
            "measurement_state": UNMEASURED,
            "why": err or f"{rel} not in working tree or git HEAD",
        }
    try:
        value = _dig(doc, field)
    except KeyError:
        return {
            "value": None,
            "source_receipt": rel,
            "source_field": field,
            "copied": False,
            "measurement_state": UNMEASURED,
            "why": f"{rel} has no field {field}",
        }
    return {
        "value": value,
        "source_receipt": rel,
        "source_field": field,
        "copied": True,
        "measurement_state": "COPIED_FROM_NAMED_RECEIPT",
        "why": "copied; this sidecar did not measure it",
    }


def cites_ok(*rows: dict[str, Any]) -> bool:
    return all(r.get("copied") for r in rows)


def cite_roof(roof_id: str, field: str) -> dict[str, Any]:
    """Copy a field from ROOF_ANCHOR.json registry[id=roof_id]."""
    doc, err = load_named(ROOF_REL)
    dotted = f"registry[id={roof_id}].{field}"
    if doc is None:
        return {
            "value": None,
            "source_receipt": ROOF_REL,
            "source_field": dotted,
            "copied": False,
            "measurement_state": UNMEASURED,
            "why": err or f"{ROOF_REL} missing",
        }
    row = None
    for item in doc.get("registry") or []:
        if isinstance(item, dict) and item.get("id") == roof_id:
            row = item
            break
    if row is None:
        return {
            "value": None,
            "source_receipt": ROOF_REL,
            "source_field": dotted,
            "copied": False,
            "measurement_state": UNMEASURED,
            "why": f"registry has no id {roof_id!r}",
        }
    try:
        value = _dig(row, field)
    except KeyError:
        return {
            "value": None,
            "source_receipt": ROOF_REL,
            "source_field": dotted,
            "copied": False,
            "measurement_state": UNMEASURED,
            "why": f"{roof_id} has no field {field}",
        }
    return {
        "value": value,
        "source_receipt": ROOF_REL,
        "source_field": dotted,
        "copied": True,
        "measurement_state": "COPIED_FROM_NAMED_RECEIPT",
        "why": "copied; this sidecar did not measure it",
    }


# ---------------------------------------------------------------------------
# Sealed B-side specimens. Curriculum + verification are the authority.
# ---------------------------------------------------------------------------


def _verification_rows() -> dict[str, dict[str, Any]]:
    doc, _ = load_named(VERIFY_REL)
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(doc, dict):
        return out
    for row in doc.get("results") or []:
        if isinstance(row, dict) and row.get("specimen"):
            out[str(row["specimen"])] = row
    return out


def sealed_specimens() -> dict[str, dict[str, Any]]:
    """The five WHOLE_TREE_VERIFIED first-wave specimens. Not invented."""
    verified = _verification_rows()
    roles = {
        "very_small_dense_procedural_speed": "Qwen3-0.6B",
        "small_dense_alternate_architecture_transfer": "Falcon-H1-7B-Instruct",
        "mid_size_dense_compiler": "Mistral-Small-3.1-24B",
        "qwen27_mature_physical": "qwen27",
        "flash_heterogeneous_frontier": "Qwen3.8-Flash-Next",
    }
    by_role: dict[str, dict[str, Any]] = {}
    curriculum_roles: list[dict[str, Any]] = []
    rec, _ = load_named("receipts/future/SPECIMEN_CURRICULUM.json")
    if isinstance(rec, dict):
        block = rec.get("curriculum") if isinstance(rec.get("curriculum"), dict) else rec
        curriculum_roles = [r for r in (block.get("roles") or []) if isinstance(r, dict)]

    for row in curriculum_roles:
        role = str(row.get("role") or "")
        alias = roles.get(role)
        if not alias:
            continue
        vs = row.get("verified_specimen") if isinstance(row.get("verified_specimen"), dict) else {}
        specimen_id = None
        path = vs.get("specimen_path") or row.get("specimen_path")
        if isinstance(path, str) and path:
            specimen_id = path.rstrip("/").split("/")[-1]
            if specimen_id == "qwen3.8-27b-abliterated-bf16":
                specimen_id = SOURCE_SPECIMEN
        if specimen_id is None:
            # Fall back to verification rows by repo.
            repo = str(row.get("repo") or "")
            for name, vrow in verified.items():
                if repo and repo.split("/")[-1].lower() in name.lower():
                    specimen_id = name
                    break
                if alias.lower().replace(".", "") in name.lower().replace(".", ""):
                    specimen_id = name
                    break
        vrow = verified.get(specimen_id or "") or {}
        whole = bool(
            (vs.get("whole_tree_verified") if vs else False)
            or vrow.get("status") == "WHOLE_TREE_VERIFIED"
            or vrow.get("whole_tree_verified")
        )
        by_role[alias] = {
            "alias": alias,
            "role": role,
            "repo": row.get("repo"),
            "revision": row.get("revision") or vs.get("revision") or vrow.get("specimen"),
            "architecture_family": row.get("architecture_family"),
            "specimen_id": specimen_id,
            "specimen_path": vs.get("specimen_path") or vrow.get("specimen_path"),
            "whole_tree_verified": whole,
            "bytes_hashed": vs.get("bytes_hashed") or vrow.get("bytes_hashed"),
            "n_files": vs.get("n_files") or vrow.get("n_files"),
            "ready": bool(row.get("ready")),
            "source_of_campaign_laws": alias == "qwen27",
        }
    # Verification receipt is sufficient if curriculum is thin in this checkout.
    if "qwen27" not in by_role and SOURCE_SPECIMEN in verified:
        by_role["qwen27"] = {
            "alias": "qwen27",
            "role": "qwen27_mature_physical",
            "repo": SOURCE_MODEL,
            "revision": SOURCE_SPECIMEN,
            "architecture_family": SOURCE_FAMILY,
            "specimen_id": SOURCE_SPECIMEN,
            "whole_tree_verified": verified[SOURCE_SPECIMEN].get("status") == "WHOLE_TREE_VERIFIED",
            "bytes_hashed": verified[SOURCE_SPECIMEN].get("bytes_hashed"),
            "n_files": verified[SOURCE_SPECIMEN].get("n_files"),
            "ready": True,
            "source_of_campaign_laws": True,
        }
    aliases = {
        "Qwen3-0.6B": ("Qwen--Qwen3-0.6B@", "dense_transformer", "Qwen/Qwen3-0.6B"),
        "Falcon-H1-7B-Instruct": ("tiiuae--Falcon-H1-7B-Instruct@", "falcon_h1", "tiiuae/Falcon-H1-7B-Instruct"),
        "Mistral-Small-3.1-24B": ("mistralai--Mistral-Small-3.1-24B", "dense_transformer", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
        "Qwen3.8-Flash-Next": ("Qwen--Qwen3.8-Flash-Next@", "qwen4_exp", "Qwen/Qwen3.8-Flash-Next"),
    }
    for alias, (prefix, family, repo) in aliases.items():
        if alias in by_role:
            continue
        match = next((n for n in verified if n.startswith(prefix) or prefix.rstrip("@") in n), None)
        if not match:
            continue
        row = verified[match]
        by_role[alias] = {
            "alias": alias,
            "role": None,
            "repo": repo,
            "revision": match,
            "architecture_family": family,
            "specimen_id": match,
            "specimen_path": row.get("specimen_path"),
            "whole_tree_verified": row.get("status") == "WHOLE_TREE_VERIFIED",
            "bytes_hashed": row.get("bytes_hashed"),
            "n_files": row.get("n_files"),
            "ready": row.get("status") == "WHOLE_TREE_VERIFIED",
            "source_of_campaign_laws": False,
        }
    return by_role


def b_side_specimens() -> dict[str, dict[str, Any]]:
    return {k: v for k, v in sealed_specimens().items() if not v.get("source_of_campaign_laws")}


def require_sealed(alias: str) -> dict[str, Any]:
    specs = sealed_specimens()
    if alias not in specs:
        raise TransferError(f"specimen {alias!r} is not in the sealed first-wave set {sorted(specs)}")
    row = specs[alias]
    if not row.get("whole_tree_verified"):
        raise TransferError(f"specimen {alias!r} is not WHOLE_TREE_VERIFIED")
    return row


# ---------------------------------------------------------------------------
# Campaign laws L1–L5 as ols.Law. MODEL_LOCAL / ORGAN-scoped. Not invented.
# ---------------------------------------------------------------------------


def _tc(*aliases: str) -> tuple[dict[str, Any], ...]:
    specs = sealed_specimens()
    out: list[dict[str, Any]] = []
    for alias in aliases:
        row = specs.get(alias)
        if not row:
            continue
        out.append(
            {
                "target_school": alias,
                "target_model": row.get("repo") or alias,
                "target_architecture_family": row.get("architecture_family") or "UNKNOWN",
                "confidence": 0.25,
                "confidence_basis": (
                    "MODEL_LOCAL campaign law; values do not transfer; "
                    "method transfer is a measurement, not a similarity score"
                ),
                "counterexample_requirement": (
                    "a measurement of the same statement on this specimen that fails"
                ),
                "source_school": "Qwen27",
                "source_model": SOURCE_MODEL,
            }
        )
    return tuple(out)


def campaign_laws() -> list[ols.Law]:
    """The five named laws this campaign actually measured on sealed-3.14."""
    laws = [
        ols._law(
            law_id="LAW-L1-MLP-ARITHMETIC-SENSITIVE",
            statement=(
                "On qwen3.8-27b sealed-3.14 the MLP is arithmetic-sensitive, not "
                "addressing-limited. ARM A (stripped) at 497.4 GB/s against "
                "production 329.6 on the SAME 83.56 MB with loads proven live "
                "(MLP_ALU_ROOFLINE). Stream count is MIXED at fixed bytes/thread "
                "(MLP_STREAM_COUNT). Not a dependency chain (ILP 8/1 = 1.062), "
                "not register pressure (ws32/ws0 = 1.078), not occupancy — raising "
                "it is worse (MLP_ISSUE_RATE_LADDER)."
            ),
            source_model=SOURCE_MODEL,
            architecture_family=SOURCE_FAMILY,
            organ_class="mlp",
            evidence_strength="DIAGNOSTIC_RELATIVE",
            evidence_refs=[ALU_REL, STREAM_REL, ISSUE_REL],
            scope="MODEL_LOCAL",
            counterexample_requirement=(
                "ARM A vs production on a second specimen's MLP, loads live, "
                "same unique payload in both arms, where ARM A does not jump "
                "relative to production. Copying 497.4 GB/s is not a counterexample "
                "and is not a transfer."
            ),
            source_device="APPLE_GPU_0",
            backend="Metal",
            expected_saved_experiments=3,
            actual_saved_experiments=None,
            n_models=1,
            n_families=1,
        ),
        ols._law(
            law_id="LAW-L2-BROADCAST-AUX-NOT-CRITICAL-PATH",
            statement=(
                "On qwen3.8-27b sealed-3.14 broadcast-aux bytes are not on the "
                "critical path: shrinking the aux stream made the kernel SLOWER, "
                "and the calibrated economics now prices them at ~0 "
                "(AUX_U8_NATIVE, AUX_U8_LUT, ECONOMICS_CALIBRATION)."
            ),
            source_model=SOURCE_MODEL,
            architecture_family=SOURCE_FAMILY,
            organ_class="mlp.broadcast_aux",
            evidence_strength="DIAGNOSTIC_RELATIVE",
            evidence_refs=[AUX_NATIVE_REL, AUX_LUT_REL, ECON_REL],
            scope="MODEL_LOCAL",
            counterexample_requirement=(
                "a specimen whose broadcast-aux (or equivalent scale/bias stream) "
                "drop of the same unique bytes is billed on the critical path "
                "outside the calibration noise floor"
            ),
            source_device="APPLE_GPU_0",
            backend="Metal",
            n_models=1,
            n_families=1,
        ),
        ols._law(
            law_id="LAW-L3-MLP-FUNCTION-REPLACEMENT-CLOSED",
            statement=(
                "MLP function replacement is closed for this parent: the distilled "
                "control beats the mean predictor (0.442 vs 0.859) and still cannot "
                "carry F under 5,347,795,776 bytes (MLP_STRUCTURED_OPERATOR)."
            ),
            source_model=SOURCE_MODEL,
            architecture_family=SOURCE_FAMILY,
            organ_class="mlp",
            evidence_strength="DIAGNOSTIC_RELATIVE",
            evidence_refs=[STRUCT_REL],
            scope="MODEL_LOCAL",
            counterexample_requirement=(
                "an under-incumbent full-width operator on this parent whose "
                "held-out relative L2 is <= 0.25 (the kill). Beating the mean "
                "predictor is not function replacement."
            ),
            source_device="APPLE_GPU_0",
            backend="Metal",
            n_models=1,
            n_families=1,
        ),
        ols._law(
            law_id="LAW-L4-PROBE-UNDERSELLS-TOKEN",
            statement=(
                "A probe undersells the token on this parent: widen_f4 isolated "
                "fair-cut 0.7046 ms became 1.0245 ms on the complete token "
                "(DELTANET_WIDEN_AB); fold_addqx 1.745 ms projected became "
                "3.9833 ms measured (FOLD_ADDQX_AB)."
            ),
            source_model=SOURCE_MODEL,
            architecture_family=SOURCE_FAMILY,
            organ_class="complete_token",
            evidence_strength="DIAGNOSTIC_RELATIVE",
            evidence_refs=[WIDEN_REL, FOLD_REL],
            scope="MODEL_LOCAL",
            counterexample_requirement=(
                "a probe vs complete-token pair on this parent where the probe "
                "over-sells the saving (probe_ms > complete_token_ms) by more "
                "than the materiality bar, or a third independent pair that "
                "fails to undersell"
            ),
            source_device="APPLE_GPU_0",
            backend="Metal",
            n_models=1,
            n_families=1,
        ),
        ols._law(
            law_id="LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD",
            statement=(
                "The production-shaped roof WITH the activation load is 497.4 GB/s "
                "on two independent legs (MLP ARM A stripped and LM head "
                "production); 703.5 is a probe that never loads the activation "
                "(ROOF_ANCHOR, MLP_ALU_ROOFLINE)."
            ),
            source_model=SOURCE_MODEL,
            architecture_family=SOURCE_FAMILY,
            organ_class="production_shaped_activation_load",
            evidence_strength="DIAGNOSTIC_RELATIVE",
            evidence_refs=[ALU_REL, ROOF_REL],
            scope="MODEL_LOCAL",
            counterexample_requirement=(
                "a third production-shaped organ on this parent that loads the "
                "activation and whose cited GB/s disagrees with 497.4 by more "
                "than the catalog-full sibling (~2%), or a second specimen's "
                "same-shape measurement that disagrees. 703.5 without an "
                "activation load is not a counterexample — the law already "
                "names it as the wrong shape."
            ),
            source_device="APPLE_GPU_0",
            backend="Metal",
            n_models=1,
            n_families=1,
        ),
    ]
    attached: list[ols.Law] = []
    targets = (
        "Falcon-H1-7B-Instruct",
        "Qwen3-0.6B",
        "Mistral-Small-3.1-24B",
        "Qwen3.8-Flash-Next",
    )
    extra = _tc(*targets)
    for law in laws:
        attached.append(replace(law, transfer_candidates=extra))
    return attached


def campaign_law(law_id: str) -> ols.Law:
    for law in campaign_laws():
        if law.law_id == law_id:
            return law
    raise TransferError(f"unknown campaign law {law_id!r}")


def campaign_citations(law_id: str) -> dict[str, Any]:
    """Every hardware-shaped number this module is willing to mention, cited."""
    if law_id == "LAW-L1-MLP-ARITHMETIC-SENSITIVE":
        return {
            "arm_a_gb_s": cite(ALU_REL, "mlp.arm_a_stripped.effective_gb_s"),
            "production_gb_s": cite(ALU_REL, "mlp.production.effective_gb_s"),
            "weight_bytes": cite(ALU_REL, "mlp.arm_a_stripped.weight_bytes"),
            "loads_survived": cite(ALU_REL, "mlp.judgement.loads_survived.survived"),
            "arm_a_over_production": cite(ALU_REL, "mlp.judgement.arm_a_over_production"),
            "lm_head_gb_s": cite(ALU_REL, "lm_head_gb_s"),
            "ilp_8_over_1": cite(ISSUE_REL, "judgement.ilp.ratio_8_over_1"),
            "register_ws32_over_ws0": cite(ISSUE_REL, "judgement.register_pressure.ratio_ws32_over_ws0"),
            "stream_verdict": cite(STREAM_REL, "verdict"),
        }
    if law_id == "LAW-L2-BROADCAST-AUX-NOT-CRITICAL-PATH":
        return {
            "native_slower_us": cite(AUX_NATIVE_REL, "gpu_ab.delta.gpu_us"),
            "bytes_removed_this_layer": cite(AUX_NATIVE_REL, "gpu_ab.bytes_removed_this_layer"),
            "bytes_removed_organ": cite(AUX_NATIVE_REL, "economics.bytes_only_screen_style.bytes_removed"),
            "predicted_ms_saved": cite(AUX_NATIVE_REL, "economics.bytes_only_screen_style.predicted_ms_saved"),
            "aux_stream_on_critical_path": cite(
                AUX_NATIVE_REL,
                "economics.bytes_only_screen_style.assumptions.stream_on_critical_path",
            ),
            "lut_faster_than_incumbent": cite(AUX_LUT_REL, "gpu_ab.speedup.lut_faster_than_incumbent"),
        }
    if law_id == "LAW-L3-MLP-FUNCTION-REPLACEMENT-CLOSED":
        return {
            "distilled_held_out_relative_l2": cite(STRUCT_REL, "distilled_control.best_held_out_relative_l2"),
            "mean_held_out_relative_l2": cite(STRUCT_REL, "distilled_control.mean_held_out_relative_l2"),
            "incumbent_mlp_bytes": cite(STRUCT_REL, "incumbent_mlp_bytes"),
            "function_replacement_closed": cite(STRUCT_REL, "campaign.function_replacement_closed"),
            "n_survivors_under_incumbent": cite(STRUCT_REL, "campaign.n_survivors_under_incumbent"),
        }
    if law_id == "LAW-L4-PROBE-UNDERSELLS-TOKEN":
        return {
            "widen_isolated_ms": cite(WIDEN_REL, "saving.isolated_fair_cut_ms"),
            "widen_complete_ms": cite(WIDEN_REL, "saving.complete_token_saving_ms"),
            "fold_projection_ms": cite(FOLD_REL, "cited_diagnostic.projection_token_ms"),
            "fold_incumbent_ms": cite(FOLD_REL, "complete_token.incumbent_ms"),
            "fold_addqx_ms": cite(FOLD_REL, "complete_token.fold_addqx_ms"),
            "fold_isolated_incumbent_ms": cite(FOLD_REL, "isolated_mlp.mlp_full_incumbent_ms"),
            "fold_isolated_addqx_ms": cite(FOLD_REL, "isolated_mlp.mlp_full_fold_addqx_ms"),
        }
    if law_id == "LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD":
        return {
            "mlp_arm_a_gb_s": cite(ALU_REL, "mlp.arm_a_stripped.effective_gb_s"),
            "lm_head_gb_s": cite(ALU_REL, "lm_head_gb_s"),
            "clean_gemv_gb_s": cite(ALU_REL, "clean_gemv_gb_s"),
            "deltanet_arm_a_gb_s": cite(ALU_REL, "deltanet.arm_a_stripped.effective_gb_s"),
            "mlp_arm_a_activation_loaded": cite_roof("mlp_arm_a_stripped_497p4", "what_was_measured.activation_loaded"),
            "lm_head_activation_loaded": cite_roof("lm_head_production_497p4", "what_was_measured.activation_loaded"),
            "addr_probe_activation_loaded": cite_roof("q4_single_gemv_addr_13p6gb_max", "what_was_measured.activation_loaded"),
            "addr_probe_usable": cite_roof("q4_single_gemv_addr_13p6gb_max", "usable_as_production_streaming_roof"),
        }
    raise TransferError(f"no citations registered for {law_id!r}")


# ---------------------------------------------------------------------------
# Store-backed scars consulted before new search (§65).
# ---------------------------------------------------------------------------


def store_law_ids() -> list[str]:
    doc, _ = load_named(LAW_STORE_REL)
    if not isinstance(doc, dict):
        laws, _ = ols.seed_store()
        return [l.law_id for l in laws]
    return [str(x["law_id"]) for x in (doc.get("laws") or []) if isinstance(x, dict) and x.get("law_id")]


def consulted_store_laws() -> dict[str, Any]:
    """§65: laws and scars first. These already-stored laws constrain transfer."""
    wanted = {
        "LAW-VALUES-DO-NOT-TRANSFER-METHODS-DO": (
            "values (497.4, 329.6, 83.56 MB, 5.35e9 bytes) do not copy; "
            "a method may still be worth measuring"
        ),
        "LAW-FALCON-H1-HAS-NO-EXPERT-TENSORS": (
            "Falcon-H1 is not an MoE transfer target; L1 is a dense MLP law, "
            "so this scar blocks expert-tensor copies, not the MLP METHOD"
        ),
        "LAW-KERNEL-REUSE-FOLLOWS-STORAGE-LAYOUT": (
            "kernel identity is a storage claim; architecture-family similarity "
            "is not evidence the 497.4 kernel exists on B"
        ),
    }
    have = set(store_law_ids())
    return {
        lid: {
            "in_store": lid in have,
            "how_it_constrains_transfer": why,
        }
        for lid, why in wanted.items()
    }


def falcon_layout() -> dict[str, Any]:
    doc, err = load_named(XFER_VERIFIED_REL)
    if doc is None:
        return {
            "measurement_state": UNMEASURED,
            "why": err or f"{XFER_VERIFIED_REL} missing",
            "experiment_that_would_settle": (
                "re-run the layout survey in ACCELERATOR_TRANSFER_VERIFIED on "
                "tiiuae/Falcon-H1-7B-Instruct"
            ),
        }
    layout = ((doc.get("result") or {}).get("layout_survey") or {}).get("Falcon-H1-7B")
    if not isinstance(layout, dict):
        return {
            "measurement_state": UNMEASURED,
            "why": "layout_survey.Falcon-H1-7B missing",
            "experiment_that_would_settle": (
                "re-run the layout survey in ACCELERATOR_TRANSFER_VERIFIED"
            ),
        }
    return {
        "measurement_state": "COPIED_FROM_NAMED_RECEIPT",
        "source_receipt": XFER_VERIFIED_REL,
        "source_field": "result.layout_survey.Falcon-H1-7B",
        "storage": layout.get("storage"),
        "probe_tensor": layout.get("probe_tensor"),
        "msl_matches_model2_from_disk": layout.get("msl_matches_model2_from_disk"),
        "copied": True,
    }


# ---------------------------------------------------------------------------
# No Odyssey I barrier. Concurrent with I and III.
# ---------------------------------------------------------------------------


def odyssey_i_barrier() -> None:
    """There is no global Phase-I barrier. Returning None is the deliverable."""
    return None


def may_transfer(law: ols.Law | None, specimen_b: Mapping[str, Any] | None) -> bool:
    """True as soon as a scoped law and a different sealed B exist."""
    if odyssey_i_barrier() is not None:
        return False
    if law is None or specimen_b is None:
        return False
    if not specimen_b.get("whole_tree_verified"):
        return False
    if law.scope not in ols.SCOPES:
        return False
    return not _same_origin(law, specimen_b)


def _same_origin(law: ols.Law, specimen_b: Mapping[str, Any]) -> bool:
    sid = str(specimen_b.get("specimen_id") or "")
    repo = str(specimen_b.get("repo") or "")
    if sid == SOURCE_SPECIMEN or specimen_b.get("source_of_campaign_laws"):
        return True
    return bool(pl._same_model(law.source_model, repo) or pl._same_model(law.source_model, sid))


# ---------------------------------------------------------------------------
# Widen guard: replicating specimen, then the store's sequential lattice.
# ---------------------------------------------------------------------------


def replicating_specimens(law: ols.Law, evidence: Mapping[str, Any]) -> list[str]:
    """Second specimens whose measurement of THIS statement HOLDS.

    Sealed, listed, or UNMEASURED does not count.
    """
    out: list[str] = []
    for row in evidence.get("replications") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("verdict") or "") != "HOLDS":
            continue
        if str(row.get("measurement_state") or "") == UNMEASURED:
            continue
        if not row.get("measurement"):
            continue
        spec = str(row.get("specimen") or "").strip()
        if not spec:
            continue
        if pl._same_model(spec, law.source_model) or spec == SOURCE_SPECIMEN:
            continue
        stmt = str(row.get("law_id") or row.get("statement_id") or "")
        if stmt and stmt != law.law_id:
            continue
        if spec not in out:
            out.append(spec)
    return out


def _require_replicating_specimen(law: ols.Law, evidence: Mapping[str, Any]) -> list[str]:
    reps = replicating_specimens(law, evidence)
    if not reps:
        raise ReplicatingSpecimenRequired(
            f"{law.law_id}: refusing to widen scope without a replicating specimen. "
            f"A sealed B-side is not a replication. UNMEASURED is not a replication. "
            f"Got replications={list(evidence.get('replications') or [])}",
            law_id=law.law_id,
            from_scope=law.scope,
            reason="need_replicating_specimen",
        )
    return reps


def widen(law: ols.Law, target_scope: str, evidence: Mapping[str, Any]) -> ols.Law:
    """Widen one lattice step only if a second specimen replicated the statement.

    Delegates the lattice itself to ols.promote. Does not fork a second lattice.
    """
    ols.validate_law(law)
    _require_replicating_specimen(law, evidence)
    return ols.promote(law, target_scope, dict(evidence))


def record_narrowing(law: ols.Law, *, target: str, reason: str) -> dict[str, Any]:
    """A failed transfer measures scope. The law is kept, not deleted."""
    return {
        "law_id": law.law_id,
        "deleted": False,
        "scope_before": law.scope,
        "scope_after": law.scope,  # II lattice bottoms at MODEL_LOCAL
        "narrowed_to": {
            "source_model": law.source_model,
            "organ_class": law.organ_class,
            "does_not_carry_values_to": target,
        },
        "reason": reason,
        "note": (
            "MODEL_LOCAL is already the bottom of the Odyssey II promotion "
            "lattice; the measured scope is recorded on the transfer, not by "
            "deleting the law or inventing an ORGAN_LOCAL rung here (that rung "
            "lives on the Odyssey III adversary ladder)."
        ),
    }


# ---------------------------------------------------------------------------
# Transfer trials. Measurement or UNMEASURED. Never a similarity score.
# ---------------------------------------------------------------------------


def _refuse_similarity(evidence: Mapping[str, Any]) -> None:
    if "similarity" in evidence or "similarity_score" in evidence or "cosine_to_source" in evidence:
        raise NotAMeasurementError(
            "a similarity score is not a transfer verdict; report the "
            "measurement (or UNMEASURED with the experiment that would settle it)"
        )


def transfer_values_l1_to_falcon() -> dict[str, Any]:
    """L1 VALUES (497.4 / 329.6 / 83.56 MB) onto Falcon-H1.

    Those numbers are an organ-instance measurement on sealed-3.14. Falcon-H1
    is a different architecture family (curriculum: falcon_h1; store scar:
    no expert tensors). Copying the GB/s would fabricate a hardware number.
    """
    law = campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    target = require_sealed("Falcon-H1-7B-Instruct")
    if not may_transfer(law, target):
        raise TransferError("L1 -> Falcon is not runnable (barrier or identity)")
    citations = campaign_citations(law.law_id)
    layout = falcon_layout()
    consulted = consulted_store_laws()
    weight_bytes = citations["weight_bytes"]
    reason = (
        "VALUE transfer failed: 83.56 MB / 497.4 GB/s / 329.6 GB/s are the "
        "unique payload and measured-under-load rates of one sealed-3.14 MLP "
        f"layer (copied from {ALU_REL}). Falcon-H1-7B-Instruct is architecture "
        f"family {target.get('architecture_family')!r}, specimen "
        f"{target.get('specimen_id')!r}, WHOLE_TREE_VERIFIED. Layout survey "
        f"storage={layout.get('storage')!r}. The stored scar "
        "LAW-VALUES-DO-NOT-TRANSFER-METHODS-DO forbids copying the numbers. "
        "No Falcon MLP ARM A measurement exists in this campaign."
    )
    narrowing = record_narrowing(
        law,
        target=str(target.get("specimen_id")),
        reason=reason,
    )
    bought = {
        "experiments_saved": 0,
        "expected_if_values_had_held": None,
        "expected_if_method_holds": law.expected_saved_experiments,
        "precise_reason_the_law_did_not_carry": reason,
        "section_65": (
            "consulted LAW-VALUES-DO-NOT-TRANSFER-METHODS-DO and "
            "LAW-FALCON-H1-HAS-NO-EXPERT-TENSORS before any new search; "
            "did not launch an ALU/stream/issue ladder on Falcon expecting 497.4"
        ),
        "new_search_only_where_transfer_fails": (
            "METHOD (arithmetic-sensitive vs addressing-limited) is UNMEASURED "
            "on Falcon; that is the remaining experiment, not a value copy"
        ),
    }
    return {
        "trial_id": "L1-VALUES-TO-FALCON-H1",
        "law_id": law.law_id,
        "named_law": law.law_id,
        "statement": law.statement,
        "kind": "values",
        "source_specimen": SOURCE_SPECIMEN,
        "source_display": SOURCE_DISPLAY,
        "target_specimen": target.get("specimen_id"),
        "target_alias": target.get("alias"),
        "target_repo": target.get("repo"),
        "target_architecture_family": target.get("architecture_family"),
        "target_whole_tree_verified": target.get("whole_tree_verified"),
        "verdict": TRANSFER_FAILED,
        "similarity_score": None,
        "citations": citations,
        "falcon_layout": layout,
        "consulted_store_laws": consulted,
        "scope_before": law.scope,
        "scope_after": narrowing["scope_after"],
        "narrowing": narrowing,
        "what_transfer_bought": bought,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def transfer_method_l1_to_falcon() -> dict[str, Any]:
    """L1 METHOD (arithmetic-sensitive, not addressing-limited) onto Falcon-H1.

    No ARM A vs production measurement exists on Falcon. UNMEASURED, with the
    experiment that would settle it. Not a null run and not a similarity score.
    """
    law = campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    target = require_sealed("Falcon-H1-7B-Instruct")
    return {
        "trial_id": "L1-METHOD-TO-FALCON-H1",
        "law_id": law.law_id,
        "named_law": law.law_id,
        "statement": law.statement,
        "kind": "method",
        "source_specimen": SOURCE_SPECIMEN,
        "source_display": SOURCE_DISPLAY,
        "target_specimen": target.get("specimen_id"),
        "target_alias": target.get("alias"),
        "target_repo": target.get("repo"),
        "target_architecture_family": target.get("architecture_family"),
        "target_whole_tree_verified": target.get("whole_tree_verified"),
        "verdict": UNMEASURED,
        "similarity_score": None,
        "measurement": None,
        "measurement_state": UNMEASURED,
        "experiment_that_would_settle": L1_METHOD_EXPERIMENT,
        "why_not_copied": (
            "ARM A 497.4 / production 329.6 are sealed-3.14 MLP numbers; "
            "writing them onto Falcon-H1 would fabricate a hardware number"
        ),
        "scope_before": law.scope,
        "scope_after": law.scope,
        "what_transfer_bought": {
            "experiments_saved": 0,
            "precise_reason_the_law_did_not_carry": (
                "METHOD unmeasured on Falcon-H1; VALUE transfer already failed. "
                "§65: new search is the ARM A experiment named above, not a "
                "re-run of the source ladder expecting the same GB/s."
            ),
        },
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def transfer_l2_to_falcon() -> dict[str, Any]:
    """L2 (broadcast-aux not on the critical path) onto Falcon-H1.

    The organ is the affine-q2 broadcast-aux stream of this parent. Falcon-H1
    has no such stream in the layout survey (no expert tensors; different
    family). Organ absent is a transfer failure with static evidence, not a
    fabricated GB/s.
    """
    law = campaign_law("LAW-L2-BROADCAST-AUX-NOT-CRITICAL-PATH")
    target = require_sealed("Falcon-H1-7B-Instruct")
    citations = campaign_citations(law.law_id)
    layout = falcon_layout()
    reason = (
        "organ absent on B: LAW-L2 is about the affine-q2 broadcast-aux stream "
        f"(stream_on_critical_path copied as {citations['aux_stream_on_critical_path']['value']!r} "
        f"from {AUX_NATIVE_REL}). Falcon-H1 layout storage={layout.get('storage')!r}, "
        f"family={target.get('architecture_family')!r}. There is no broadcast-aux "
        "organ to transfer onto. The law stays MODEL_LOCAL to sealed-3.14 mlp.broadcast_aux."
    )
    narrowing = record_narrowing(law, target=str(target.get("specimen_id")), reason=reason)
    return {
        "trial_id": "L2-ORGAN-ABSENT-ON-FALCON-H1",
        "law_id": law.law_id,
        "named_law": law.law_id,
        "statement": law.statement,
        "kind": "organ_presence",
        "source_specimen": SOURCE_SPECIMEN,
        "source_display": SOURCE_DISPLAY,
        "target_specimen": target.get("specimen_id"),
        "target_alias": target.get("alias"),
        "target_repo": target.get("repo"),
        "target_architecture_family": target.get("architecture_family"),
        "target_whole_tree_verified": target.get("whole_tree_verified"),
        "verdict": TRANSFER_FAILED,
        "similarity_score": None,
        "citations": citations,
        "falcon_layout": layout,
        "scope_before": law.scope,
        "scope_after": narrowing["scope_after"],
        "narrowing": narrowing,
        "what_transfer_bought": {
            "experiments_saved": 1,
            "saved": (
                "did not run AUX_U8_NATIVE / ECONOMICS_CALIBRATION on Falcon "
                "expecting a broadcast-aux critical-path price of ~0"
            ),
            "precise_reason_the_law_did_not_carry": reason,
        },
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def identity_transfer_refused() -> dict[str, Any]:
    law = campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    source = sealed_specimens().get("qwen27") or {
        "specimen_id": SOURCE_SPECIMEN,
        "repo": SOURCE_MODEL,
        "whole_tree_verified": True,
        "source_of_campaign_laws": True,
    }
    try:
        if _same_origin(law, source):
            raise IdentityTransferError(
                f"{law.law_id}: target {source.get('specimen_id')} is the origin; "
                "a transfer onto the law's origin is not a transfer"
            )
        raise TransferError("identity transfer was not caught")
    except (IdentityTransferError, pl.NotATransferError) as e:
        return {
            "refused": True,
            "reason_code": "not_a_transfer",
            "reason": str(e),
            "law_id": law.law_id,
            "target_specimen": source.get("specimen_id"),
        }


def run_transfers() -> dict[str, Any]:
    values = transfer_values_l1_to_falcon()
    method = transfer_method_l1_to_falcon()
    l2 = transfer_l2_to_falcon()
    identity = identity_transfer_refused()
    headline = values
    return {
        "headline": headline,
        "trials": [values, method, l2],
        "identity_transfer_refused": identity,
        "n_held": sum(1 for t in (values, method, l2) if t["verdict"] == TRANSFER_HELD),
        "n_failed": sum(1 for t in (values, method, l2) if t["verdict"] == TRANSFER_FAILED),
        "n_unmeasured": sum(1 for t in (values, method, l2) if t["verdict"] == UNMEASURED),
        "named_law": headline["named_law"],
        "specimen_used": headline["target_specimen"],
        "verdict": headline["verdict"],
    }


def selftest() -> dict[str, Any]:
    run = run_transfers()
    law = campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    if run["verdict"] != TRANSFER_FAILED:
        raise TransferError(f"headline L1 values->Falcon must FAIL, got {run['verdict']}")
    if run["named_law"] != law.law_id:
        raise TransferError("headline did not consume the named campaign law")
    if not run["specimen_used"] or "Falcon" not in str(run["specimen_used"]):
        raise TransferError(f"headline did not name Falcon: {run['specimen_used']}")
    if odyssey_i_barrier() is not None:
        raise TransferError("Odyssey I barrier is not None")
    # Widen without a replicating specimen must RAISE, not clamp.
    try:
        widen(
            law,
            "ARCHITECTURE_FAMILY",
            {
                "models": [SOURCE_MODEL, "tiiuae/Falcon-H1-7B-Instruct"],
                "architecture_families": [SOURCE_FAMILY, "falcon_h1"],
                "evidence_strength": "DIAGNOSTIC_RELATIVE",
                "replications": [],
            },
        )
        raise TransferError("widen without replicating specimen did not raise")
    except ReplicatingSpecimenRequired as e:
        widen_refused = {
            "raised": True,
            "reason": e.reason,
            "law_id": e.law_id,
        }
    # Similarity score is not a transfer.
    try:
        _refuse_similarity({"similarity": 0.99})
        raise TransferError("similarity score was not refused")
    except NotAMeasurementError:
        similarity_refused = True
    # Law not deleted on failure.
    still = campaign_law(law.law_id)
    if still.law_id != law.law_id:
        raise TransferError("failed transfer deleted the law")
    return {
        "ok": True,
        "named_law": run["named_law"],
        "specimen_used": run["specimen_used"],
        "verdict": run["verdict"],
        "widen_without_replicating_specimen_raised": widen_refused,
        "similarity_score_refused": similarity_refused,
        "law_deleted": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
        "listen_rule": pl.LISTEN_RULE,
        "n_failed": run["n_failed"],
        "n_unmeasured": run["n_unmeasured"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def build() -> Any:
    laws = campaign_laws()
    for law in laws:
        ols.validate_law(law)
    run = run_transfers()
    loop = selftest()
    specs = sealed_specimens()
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Odyssey II transfer: consume a named campaign law and test it on a "
            "different sealed specimen. Report transfer or failure with the "
            "measurement (or UNMEASURED). Concurrent with Odyssey I and III; "
            "no Phase-I barrier."
        ),
        "odyssey": "II WHAT DID HAWKING ALREADY LEARN?",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "odyssey_i_barrier": None,
        "phase_ii_waits_for_odyssey_i_complete": False,
        "listen_rule": pl.LISTEN_RULE,
        "law_authority": {
            "module": "tools/future/odyssey2_law_store.py",
            "lattice": list(ols.SCOPES),
            "promote": "ols.promote — sequential, raises ScopeViolation",
            "transfer_candidates": "ols.transfer_candidates — atlas-dead levers raise",
            "not_a_fork": True,
        },
        "named_law": run["named_law"],
        "specimen_used": run["specimen_used"],
        "verdict": run["verdict"],
        "headline": run["headline"],
        "trials": run["trials"],
        "identity_transfer_refused": run["identity_transfer_refused"],
        "campaign_laws": [law.to_dict() for law in laws],
        "sealed_specimens": specs,
        "consulted_store_laws": consulted_store_laws(),
        "selftest": loop,
        "recovered_implementation": {
            "odyssey2_law_store": "field set, lattice, promote, atlas refusal — consumed, not forked",
            "phase_listeners": "LISTEN_RULE, NotATransferError, origin identity — consumed",
            "specimen_curriculum": "five first-wave sealed roles — consumed",
            "specimen_verify": "WHOLE_TREE_VERIFIED rows — consumed",
            "negative_transfer_atlas": "via ols.match_failed_atlas / transfer_candidates",
        },
        "gaps_closed": [
            "Named campaign laws L1–L5 as ols.Law records (MODEL_LOCAL, cited receipts).",
            "L1 VALUES transferred onto Falcon-H1 and FAILED with the cited organ-instance measurement, not a similarity score.",
            "L1 METHOD on Falcon-H1 recorded UNMEASURED with the ARM A experiment that would settle it.",
            "Failed transfer narrows scope (recorded) and does not delete the law.",
            "widen() raises ReplicatingSpecimenRequired before ols.promote when no second specimen HOLDS the statement.",
            "No Odyssey I completion barrier (LISTEN_RULE).",
        ],
        "negative_findings": [
            "No GPU lease: METHOD transfer of L1 onto Falcon is UNMEASURED, not a fabricated 497.4.",
            "Odyssey II lattice has no ORGAN_LOCAL rung; organ locality is recorded on the transfer, not by forking the lattice.",
            "ols.SCHOOLS names Flash/Qwen27 only; Falcon is a sealed curriculum specimen, not a second school table.",
            "time_to_first_useful_executable_ns is null on every campaign law.",
        ],
        "claim_boundary_reminder": (
            "DIAGNOSTIC_RELATIVE guides, never promotes. Copied GB/s figures "
            "are citations. UNMEASURED stays UNMEASURED."
        ),
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps({k: v for k, v in selftest().items() if k != "trials"}, indent=1, sort_keys=True))
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
