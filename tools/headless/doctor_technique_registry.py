#!/usr/bin/env python3
"""N043 — Doctor Technique Registry (S026 §5, §76; PHASE A; CPU).

External papers are HYPOTHESES, not authority. A CURRENT VERDICT other than
UNTESTED is illegal without a cited Hawking receipt. This lane is the
registry, not the experiments: it does not load a model, does not touch the
GPU, does not run cargo/Metal, and does not mutate NOETIC_PARENT_A.

    python3 tools/headless/doctor_technique_registry.py
    python3 -m pytest tools/headless/test_doctor_technique_registry.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "DOCTOR_TECHNIQUE_REGISTRY.json"
DOCS = REPO / "docs" / "ultragoals" / "DOCTOR_TECHNIQUE_REGISTRY.md"
GENERATOR = "tools/headless/doctor_technique_registry.py"
SCHEMA = "hawking.headless.doctor_technique_registry.v1"
OBLIGATION = (
    "N043 — DOCTOR_TECHNIQUE_REGISTRY (S026 §5, §76; DOC-DIAGNOSIS family, "
    "PHASE A; CPU). Register the paper mechanisms as HYPOTHESES with "
    "provenance + Hawking experiment mapping + CURRENT VERDICT, seeded from "
    "this campaign's negative science. Literature is not authority."
)

LITERATURE_STATUS = "HYPOTHESIS"
UNTESTED = "UNTESTED"
TESTED_NEGATIVE = "TESTED_NEGATIVE"
TESTED_PARTIAL = "TESTED_PARTIAL"
RELATED_NEGATIVE = "RELATED_NEGATIVE"
TESTED_POSITIVE = "TESTED_POSITIVE"

ALLOWED_VERDICT_STATUSES = frozenset(
    {UNTESTED, TESTED_NEGATIVE, TESTED_PARTIAL, RELATED_NEGATIVE, TESTED_POSITIVE}
)
VERDICT_REQUIRES_RECEIPT = frozenset(
    {TESTED_NEGATIVE, TESTED_PARTIAL, RELATED_NEGATIVE, TESTED_POSITIVE}
)

REQUIRED_TECHNIQUE_IDS = (
    "spinquant",
    "twla",
    "cat_q",
    "ptqtp",
    "onebit",
    "aqlm",
    "vptq",
    "caldera",
    "squeezellm",
    "kivi",
    "minicache",
    "h2o",
    "mixture_of_depths",
    "prosparse",
    "medusa_mtp",
)

REQUIRED_ENTRY_FIELDS = (
    "technique_identity",
    "source_paper",
    "claimed_mechanism",
    "architecture_assumptions",
    "training_calibration_runtime",
    "storage_vs_execution",
    "expected_useful_organs",
    "expected_physical_win",
    "risks",
    "licensing_provenance",
    "hawking_experiment_mapping",
    "current_verdict",
)

# Contract phrases (S026 N043). Tests assert these strings.
SCAR_SHARED_BASIS = "competent kernel but dead <2.25"
SCAR_BINARY = "fast, uniformly injured"
SCAR_LOWRANK = "never heals"
SCAR_TERNARY = "slower + argmax flip"
SCAR_SPARSE = "indices cost more"

R_BYTES = "receipts/headless/BYTES_FRONTIER.json"
R_BINARY_HEALING = "receipts/headless/BINARY_HEALING.json"
R_SHARED_K = "receipts/headless/SHARED_BASIS_KERNEL.json"
R_SHARED_C = "receipts/headless/SHARED_BASIS_COHERENT.json"
R_HYBRID = "receipts/headless/HYBRID_OPERATOR.json"
R_TERNARY_COMP = "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"
R_FRACTIONAL = "receipts/headless/FRACTIONAL_BIT_CANON.json"
R_FIRST_EXEC = "receipts/headless/FIRST_NOETIC_EXECUTABLE.json"
R_ONEBIT = "receipts/headless/ONEBIT_FAMILIES.json"
R_C1 = "receipts/headless/C1SHAREDBASIS_DESIGN.json"
R_C3 = "receipts/headless/C3LOWRANKSPARSE_DESIGN.json"
R_C4 = "receipts/headless/C4CODEBOOK_DESIGN.json"
R_C5 = "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json"
R_PREFILL = "receipts/headless/PREFILL_KV.json"
R_NNS = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
R_REPR = "receipts/headless/REPRESENTATION_LIBRARY.json"
R_GQA = "receipts/headless/NOETIC_GQA_DESIGN.json"
R_FRONTIERS = "receipts/headless/ORGAN_FRONTIERS.json"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def git_exists(rel_path: str) -> bool:
    rel_path = rel_path.lstrip("./")
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        timeout=20,
    )
    return r.returncode == 0


def citation_exists(rel_path: str) -> bool:
    """Sparse-missing is not absence — check git as well as the working tree."""
    rel_path = rel_path.lstrip("./")
    if (REPO / rel_path).is_file():
        return True
    return git_exists(rel_path)


def load_json(rel_path: str) -> dict[str, Any]:
    rel_path = rel_path.lstrip("./")
    p = REPO / rel_path
    if p.is_file():
        return json.loads(p.read_text())
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise FileNotFoundError(rel_path)
    return json.loads(r.stdout)


def is_hawking_receipt_path(path: str) -> bool:
    """A Hawking receipt is a receipts/**/*.json path, not a paper URL."""
    if not isinstance(path, str):
        return False
    p = path.lstrip("./")
    if p.startswith("http://") or p.startswith("https://"):
        return False
    if not p.startswith("receipts/"):
        return False
    if not p.endswith(".json"):
        return False
    if ".." in p:
        return False
    return True


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# campaign seed (cited, not re-derived)
# ---------------------------------------------------------------------------


def _rep(bytes_frontier: dict[str, Any], rid: str) -> dict[str, Any]:
    for r in bytes_frontier.get("representations") or []:
        if r.get("id") == rid:
            return r
    raise KeyError(rid)


def _composed_ns(rep: dict[str, Any]) -> int | None:
    ns = rep.get("COMPLETE_TOKEN_NS") or {}
    composed = ns.get("composed") or {}
    v = composed.get("complete_token_ns")
    return int(v) if v is not None else None


def _ternary_argmax(comp: dict[str, Any]) -> dict[str, Any]:
    for r in comp.get("rungs") or []:
        if r.get("rung") in ("complete_token_loop", "complete_token"):
            return {
                "teacher_argmax": r.get("teacher_argmax"),
                "student_argmax": r.get("student_argmax"),
                "argmax_agree": r.get("argmax_agree"),
            }
    return {"teacher_argmax": None, "student_argmax": None, "argmax_agree": None}


def extract_campaign_seed() -> dict[str, Any]:
    """Pull the numbers this registry is allowed to quote. Missing receipts fail."""
    bf = load_json(R_BYTES)
    binary = _rep(bf, "binary_g64")
    ternary = _rep(bf, "ternary_5in8_g64")
    sparse = _rep(bf, "binary_residual_sparse_2pct")
    shared_k2 = _rep(bf, "shared_binary_k2")
    q2f = _rep(bf, "q2_4level_fitted_g64")
    sbk = load_json(R_SHARED_K)
    sbc = load_json(R_SHARED_C)
    hyb = load_json(R_HYBRID)
    bh = load_json(R_BINARY_HEALING)
    tern_comp = load_json(R_TERNARY_COMP)
    c1 = load_json(R_C1)
    c3 = load_json(R_C3)
    c4 = load_json(R_C4)
    c5 = load_json(R_C5)
    onebit = load_json(R_ONEBIT)
    prefill = load_json(R_PREFILL)
    argmax = _ternary_argmax(tern_comp)
    cfm = bh.get("COHERENCE_FAILURE_MAP") or {}
    bh_find = bh.get("finding") or {}
    sparse_notes = sparse.get("notes") or {}
    headline = prefill.get("headline_footprint") or {}
    q4_long_c4 = headline.get("q4_long_c4") or {}
    kv_prec = (prefill.get("kv_precision") or {}).get("4k") or {}
    onebit_v = onebit.get("verdict") or {}
    return {
        "q2f_bpw": q2f.get("active_bpw"),
        "q2f_ns": _composed_ns(q2f),
        "binary_bpw": binary.get("active_bpw"),
        "binary_ns": _composed_ns(binary),
        "binary_delta_ns": (binary.get("toward_roof_729_7") or {}).get("delta_ns"),
        "binary_moved_ns": (binary.get("toward_roof_729_7") or {}).get("moved"),
        "binary_died_at": (binary.get("coherence") or {}).get("died_at"),
        "binary_why": (binary.get("coherence") or {}).get("why"),
        "ternary_bpw": ternary.get("active_bpw"),
        "ternary_ns": _composed_ns(ternary),
        "ternary_delta_ns": (ternary.get("toward_roof_729_7") or {}).get("delta_ns"),
        "ternary_moved_ns": (ternary.get("toward_roof_729_7") or {}).get("moved"),
        "ternary_died_at": (ternary.get("coherence") or {}).get("died_at"),
        "ternary_why": (ternary.get("coherence") or {}).get("why"),
        "ternary_teacher_argmax": argmax["teacher_argmax"],
        "ternary_student_argmax": argmax["student_argmax"],
        "ternary_argmax_agree": argmax["argmax_agree"],
        "sparse_bpw": sparse.get("active_bpw"),
        "sparse_ns": _composed_ns(sparse),
        "sparse_delta_ns": (sparse.get("toward_roof_729_7") or {}).get("delta_ns"),
        "sparse_moved_ns": (sparse.get("toward_roof_729_7") or {}).get("moved"),
        "sparse_csr_bytes": sparse_notes.get("csr_bytes"),
        "sparse_binary_bytes": sparse_notes.get("binary_bytes"),
        "sparse_nnz_frac": sparse_notes.get("nnz_frac"),
        "shared_k2_bpw": shared_k2.get("active_bpw"),
        "shared_k2_ns": _composed_ns(shared_k2),
        "shared_kernel_competent": sbk.get("competent"),
        "shared_byte_win_translates": sbk.get("byte_win_translates_to_token_ns"),
        "shared_active_bpw": sbk.get("active_bpw"),
        "shared_beats_q2f": sbc.get("coherent_shared_basis_beats_q2f"),
        "shared_op": sbc.get("operating_point") or {},
        "hybrid_beats_q2f": hyb.get("coherent_hybrid_beats_q2f"),
        "hybrid_finding": (hyb.get("finding") or {}).get("reason"),
        "hybrid_died_at": (hyb.get("finding") or {}).get("died_at"),
        "uniformly_injured": cfm.get("uniformly_injured"),
        "healing_n_coherent": bh_find.get("n_that_reached_coherent_generation"),
        "healing_n_candidates": bh_find.get("n_healing_candidates"),
        "binary_injured_ns": ((bh_find.get("injured_body") or {}).get("COMPLETE_TOKEN_NS")),
        "c1_verdict": c1.get("verdict"),
        "c1_failure": c1.get("failure_that_killed_it") or c1.get("answer"),
        "c3_answer": c3.get("answer"),
        "c4_answer": c4.get("answer"),
        "c5_verdict": c5.get("verdict"),
        "onebit_best": onebit_v.get("best_survivor"),
        "onebit_n_survive": onebit_v.get("n_survive_at_matched_bytes"),
        "prefill_answer": prefill.get("answer"),
        "q4_c4_32k_state_exceeds": q4_long_c4.get("state_exceeds_weights"),
        "q4_c4_32k_session_gib": q4_long_c4.get("PRODUCTION_FOOTPRINT_GiB"),
        "kv_production_dtype": kv_prec.get("production_dtype"),
        "kv_int4_wired": (
            ((kv_prec.get("candidates") or {}).get("int4") or {}).get(
                "wired_into_production_session"
            )
        ),
        "kv_int4_kernel_exists": (
            ((kv_prec.get("candidates") or {}).get("int4") or {}).get("kernel_exists")
        ),
    }


# ---------------------------------------------------------------------------
# catalog helpers
# ---------------------------------------------------------------------------


def identity(
    tid: str,
    short: str,
    s026_name: str,
    family: str,
) -> dict[str, Any]:
    return {
        "id": tid,
        "short_name": short,
        "s026_name": s026_name,
        "s026_family": family,
        "literature_status": LITERATURE_STATUS,
        "not_authority": True,
        "s026": ["§5", "§76"],
    }


def paper(
    title: str,
    authors: str,
    approx_date: str,
    arxiv: str | None,
    venue: str | None = None,
    extra: str | None = None,
) -> dict[str, Any]:
    url = f"https://arxiv.org/abs/{arxiv}" if arxiv else None
    return {
        "title": title,
        "authors": authors,
        "approx_date": approx_date,
        "venue": venue,
        "arxiv": arxiv,
        "url": url,
        "note": extra,
        "status_in_this_registry": LITERATURE_STATUS,
    }


def cheapest(
    xid: str,
    name: str,
    why: str,
    success: str,
    *,
    loads_model: bool = False,
    touches_gpu: bool = False,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": xid,
        "name": name,
        "why_cheapest": why,
        "cpu_only": not touches_gpu,
        "loads_model": loads_model,
        "touches_gpu": touches_gpu,
        "stream_parent_tensors_if_needed": True,
        "no_second_27b": True,
        "real_x_not_gaussian": True,
        "depends_on": depends_on or [],
        "success_criterion": success,
    }


def verdict(
    status: str,
    *,
    campaign_verdict: str | None,
    receipts: list[str],
    measured: dict[str, Any] | None = None,
    scope: str,
    remainder: str,
    experiment: dict[str, Any] | None,
) -> dict[str, Any]:
    if status not in ALLOWED_VERDICT_STATUSES:
        raise ValueError(status)
    return {
        "status": status,
        "literature_is": LITERATURE_STATUS,
        "campaign_verdict": campaign_verdict,
        "hawking_receipts": list(receipts),
        "measured_numbers": measured or {},
        "scope_of_measurement": scope,
        "untested_remainder": remainder,
        "cheapest_hawking_experiment": experiment,
        "citation_rule": "no CURRENT VERDICT other than UNTESTED without a cited Hawking receipt",
    }


def mapping(
    overlap: str,
    related: list[str],
    not_the_same: str,
    next_phase: str,
) -> dict[str, Any]:
    return {
        "campaign_mechanism_overlap": overlap,
        "related_receipts": related,
        "related_but_not_the_same": not_the_same,
        "next_phase_if_probe_promises": next_phase,
        "native_execution_required": (
            "Trit-plane / codebook / sparse MUST execute native "
            "(S026 §17, §27, §28). No reconstruct-to-dense. Sparse only "
            "with a competent fused path (N033/CSR lesson)."
        ),
        "competency_gate": (
            "S026 §90 / N003: condemn the KERNEL not the representation "
            "until the competence screen passes."
        ),
    }


def provenance(
    paper_license: str,
    code_note: str,
    campaign: str,
) -> dict[str, Any]:
    return {
        "s026_88": "provenance preserved",
        "s026_6": "no blind implementation",
        "paper_license_note": paper_license,
        "code_license_note": code_note,
        "do_not_copy_third_party_code": True,
        "rederive_from_paper_if_implementing": True,
        "campaign_provenance": campaign,
        "this_registry_imports_no_third_party_impl": True,
    }


# ---------------------------------------------------------------------------
# the 15 S026 §76 mechanisms
# ---------------------------------------------------------------------------


def build_techniques(seed: dict[str, Any]) -> list[dict[str, Any]]:
    s = seed
    q2f_ns = s["q2f_ns"]
    techniques = [
        {
            "technique_identity": identity(
                "spinquant",
                "SpinQuant",
                "SpinQuant (learned/function-preserving rotations)",
                "DOC-COORDINATES",
            ),
            "source_paper": paper(
                "SpinQuant: LLM quantization with learned rotations",
                "Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, "
                "Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, "
                "Yuandong Tian, Tijmen Blankevoort",
                "2024-05",
                "2405.16406",
                "ICLR 2025",
            ),
            "claimed_mechanism": (
                "Learned orthogonal rotations (Cayley-SGD, Hadamard-init) "
                "redistribute weight/activation outliers so a subsequent "
                "quantizer (RTN or GPTQ) sees a friendlier coordinate "
                "system. Rotations are function-preserving in full precision "
                "and preferably absorbed into adjacent weights (zero extra "
                "runtime matmul). S026 §7-10: CoordinateTransformGenome."
            ),
            "architecture_assumptions": (
                "Dense transformer with residual Linear maps that can absorb "
                "an orthogonal Q: RMSNorm-Linear sandwiches, GQA QKV, SwiGLU "
                "MLP. Online (not absorbed) rotations cost a GEMV per site. "
                "Qwen3.8 also has 48 DeltaNet layers whose recurrent state is "
                "not a Linear sandwich — absorption is not free there."
            ),
            "training_calibration_runtime": (
                "PTQ calibration: ~100 Cayley steps on ~128-800 C4 samples "
                "in the paper. No full QAT. Runtime: zero if absorbed into "
                "weights; otherwise an extra orthogonal matmul per rotated "
                "activation. Completeness (S026 §93) must count any unabsorbed "
                "rotation bytes."
            ),
            "storage_vs_execution": (
                "Storage: rotation is either fused into the quantized weights "
                "(zero extra bytes) or stored as orthogonal factors. "
                "Execution: identity in the rotated frame; do not reconstruct "
                "W_fp16 to apply Q at runtime."
            ),
            "expected_useful_organs": [
                "mlp_gate_up",
                "mlp_down",
                "gqa_attention",
            ],
            "expected_physical_win": (
                "Paper claim (hypothesis): W4A4 / W4A8 closer to FP16 by "
                "outlier spreading. Campaign hope (S026 §78, §117): a "
                "function-preserving rotation might MOVE the 2.25 MLP "
                "composition barrier for ternary (~1.58) or binary (1.25). "
                "That move must be physically demonstrated; the 2.25 floor "
                "stays CLOSED for the unrotated family."
            ),
            "risks": [
                "C5 already REFUTED Walsh-Hadamard / butterfly as the "
                "EXECUTABLE operator (NOT_WORTH_BUILDING). That is not "
                "SpinQuant: C5 is 'the transform IS the code'; SpinQuant is "
                "'rotate, then quantize in the new frame'. Conflating them "
                "would re-kill a different idea.",
                "Unabsorbed rotations hide bytes (S026 §93).",
                "Random non-orthogonal 'rotations' are a control, not a method.",
                "A floor that does not move under rotation stays closed (S026 §11).",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive distribution; paper is a citation, not a grant to ship.",
                "facebookresearch/SpinQuant code is CC-BY-NC 4.0 — do not copy. "
                "Re-derive from the paper if N044 warrants an implementation.",
                "S026 §5/§76; related-but-different C5STRUCTTRANSFORM_DESIGN.",
            ),
            "hawking_experiment_mapping": mapping(
                "None measured as SpinQuant. C5 tested structured transforms as "
                "the operator, not as an absorbed quantizer rotation.",
                [R_C5],
                "C5 Walsh-Hadamard / Monarch-Hadamard as implicit operator "
                f"(verdict {s['c5_verdict']}) is NOT a learned absorbed rotation.",
                "N044 COORDINATE_TRANSFORM_PROBE (S026 §78). If "
                "ROTATION_MOVES_BARRIER, bounded reopening of "
                "QWEN_MLP_ROTATED_TERNARY; else 2.25 stays closed.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                scope=(
                    "No Hawking receipt measures a learned or structured "
                    "rotation absorbed into Qwen3.8 MLP then re-quantized."
                ),
                remainder="The whole technique.",
                experiment=cheapest(
                    "HX-SPINQUANT-N044",
                    "Coordinate transform discriminator on one MLP organ",
                    "Already queued as N044. CPU. Hadamard + one structured "
                    "orthogonal vs identity vs random non-orthogonal; refit "
                    "ternary ~1.58 and binary 1.25 in the rotated frame on "
                    "real held-out X. Identity must reproduce the unrotated "
                    "baseline.",
                    "ROTATION_MOVES_BARRIER true/false with deltas. If false, "
                    "close SpinQuant's reopening claim for this body.",
                    depends_on=["N044"],
                ),
            ),
        },
        {
            "technique_identity": identity(
                "twla",
                "TWLA",
                "TWLA (ternary-friendly coords + low-bit activations)",
                "DOC-COORDINATES",
            ),
            "source_paper": paper(
                "TWLA: Achieving Ternary Weights and Low-Bit Activations "
                "for LLMs via Post-Training Quantization",
                "Z. Zhao et al.",
                "2026-06",
                "2606.13054",
                "ICML 2026 (submission)",
            ),
            "claimed_mechanism": (
                "W1.58A4 PTQ: (1) Euclidean-to-manifold asymmetric ternary "
                "quantizer, (2) Kronecker orthogonal tri-modal shaping so "
                "weights become ternary-friendly and the shared rotation "
                "suppresses activation outliers, (3) inter-layer-aware "
                "mixed-precision activations. S026 mapping: ternary-friendly "
                "coordinates PLUS low-bit activations — not ternary weights "
                "alone."
            ),
            "architecture_assumptions": (
                "Standard dense Linear stacks that accept a shared Kronecker "
                "orthogonal. Heavy-tailed activations are the paper's motive "
                "for keeping A high-precision in prior work; TWLA claims to "
                "remove that. Qwen3.8 SwiGLU intermediates (17408) and GQA "
                "head_dim=256 are in-scope; DeltaNet recurrent state is not "
                "an activation the paper quantized."
            ),
            "training_calibration_runtime": (
                "PTQ, not from-scratch 1.58-bit training. Needs calibration "
                "activations for the manifold ternary fit and the rotation. "
                "Runtime: ternary weight matvec + 4-bit activations. Native "
                "trit unpack + low-bit A kernel required; reconstruct-to-f16 "
                "is a fail (S026 §17)."
            ),
            "storage_vs_execution": (
                "Storage: ~1.58 bpw weights + rotation (absorb if possible) + "
                "4-bit activation codebook/scales. Execution must stream "
                "packed trits and packed activations; 5-in-8 still reads "
                "every byte (BYTES_FRONTIER ternary note)."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: W1.58A4 with high accuracy and end-to-end "
                "speedup from low-bit activations. Campaign: even if weight "
                "ternary is dead in the original frame, a ternary-friendly "
                "rotation plus cheap activations is a DIFFERENT condition "
                "(S026 §11, §117)."
            ),
            "risks": [
                f"Ternary 5-in-8 on this body is already {SCAR_TERNARY} "
                f"(active_bpw={s['ternary_bpw']}, complete_token_ns="
                f"{s['ternary_ns']} vs q2f {q2f_ns}; argmax "
                f"{s['ternary_student_argmax']} vs teacher "
                f"{s['ternary_teacher_argmax']}).",
                "Activation quant is untested here; GQA is the quality floor "
                "and cannot cheaply go below Q4 *weights* — KV/activation "
                "quant is a different axis (PREFILL_KV).",
                "Kronecker orthogonal may fail the same way C5/G034 energy "
                "tests failed if it is used as an implicit operator.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; literature hypothesis only.",
                "Kishon-zzx/TWLA — check license before any port; re-derive.",
                f"S026 §76; ternary scar on {R_BYTES} + {R_TERNARY_COMP}.",
            ),
            "hawking_experiment_mapping": mapping(
                "Ternary WEIGHT body (5-in-8 + g64 scale) was executed natively.",
                [R_BYTES, R_TERNARY_COMP, R_FRACTIONAL, R_REPR],
                "TWLA's Kronecker tri-modal shaping and 4-bit activations "
                "were not in the 5-in-8 experiment.",
                "N044 first (does a rotation move 1.58?). If no, TWLA's "
                "coordinate claim closes with SpinQuant. If yes, one-block "
                "W1.58A4 on real X.",
            ),
            "current_verdict": verdict(
                RELATED_NEGATIVE,
                campaign_verdict=SCAR_TERNARY,
                receipts=[R_BYTES, R_TERNARY_COMP, R_FRACTIONAL],
                measured={
                    "ternary_5in8_active_bpw": s["ternary_bpw"],
                    "ternary_complete_token_ns": s["ternary_ns"],
                    "q2f_complete_token_ns": q2f_ns,
                    "delta_ns": s["ternary_delta_ns"],
                    "moved_toward_roof": s["ternary_moved_ns"],
                    "teacher_argmax": s["ternary_teacher_argmax"],
                    "student_argmax": s["ternary_student_argmax"],
                    "argmax_agree": s["ternary_argmax_agree"],
                    "died_at": s["ternary_died_at"],
                },
                scope=(
                    "Native ternary 5-in-8 g64 MLP. Organ-local CANON "
                    "(ONEBIT B3 rel_fro 0.321) flipped whole-model argmax. "
                    "Fewer bytes, worse token_ns. Coordinate transform and "
                    "low-bit activations were not in that measurement."
                ),
                remainder=(
                    "Kronecker orthogonal tri-modal shaping + W1.58A4 "
                    "activations. Untested."
                ),
                experiment=cheapest(
                    "HX-TWLA-AFTER-N044",
                    "Ternary in rotated coords, then 4-bit activations on one block",
                    "N044 is the discriminator. If rotation does not move "
                    "1.58, stop. If it does, grouped-absmax INT4 activations "
                    "on the same held-out SwiGLU vs f16 A, CPU, streamed "
                    "parent tensors.",
                    "Held-out rel_fro/gain vs unrotated ternary and vs q2f; "
                    "argmax on a 16-token complete-token loop.",
                    depends_on=["N044"],
                ),
            ),
        },
        {
            "technique_identity": identity(
                "cat_q",
                "CAT-Q",
                "CAT-Q (nearby-weight ternary healing)",
                "DOC-HEALING",
            ),
            "source_paper": paper(
                "CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs",
                "Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Anbang Yao",
                "2026-06",
                "2606.26650",
                "ICML 2026 oral",
            ),
            "claimed_mechanism": (
                "PTQ ternary via learnable modulation (LM) of the weight "
                "distribution and ternary threshold, plus softened "
                "ternarization (ST) with a differentiable transition. "
                "S026 mapping: nearby-weight ternary healing — move weights "
                "toward a ternary-friendly neighbourhood rather than snap "
                "in place. Calibration on 512 C4 samples; group size 128 in "
                "the paper."
            ),
            "architecture_assumptions": (
                "Architecture-agnostic PTQ claimed on 1.7B–235B dense/MoE "
                "LLMs. Group-wise ternary W ≈ α T, no residual μ. Assumes "
                "a calibration set and that LM+ST converge without QAT."
            ),
            "training_calibration_runtime": (
                "PTQ with a short LM+ST optimization (hours on A100s in the "
                "paper, 512 samples). Not BitNet-from-scratch. Runtime is a "
                "ternary matvec; healing is a CALIBRATION cost, not a decode "
                "cost, if absorbed."
            ),
            "storage_vs_execution": (
                "Storage: ternary codes + per-group scale (and any unfused "
                "modulation). Execution: packed trit matvec. Healing "
                "coefficients must not hide in a second copy of W."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: ternary LLMs matching or beating BitNet 1.58 "
                "without 100B-token QAT. Campaign hope: healing could reopen "
                "the ternary argmax death if the injury is a snap-threshold "
                "artifact rather than missing information."
            ),
            "risks": [
                "BINARY_HEALING: 1.25-bit binary is uniformly injured across "
                f"64/64 layers; {s['healing_n_coherent']}/"
                f"{s['healing_n_candidates']} island/sparse heals reached "
                "coherent generation. Nearby-weight islands did not restore "
                "the binary body.",
                "Ternary already flipped argmax. Healing a dead body can "
                "spend bytes without climbing the composition ladder.",
                "LM+ST that stores a dense modulation at runtime is a second "
                "copy of W (S026 §93).",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; ICML 2026 oral. Hypothesis only.",
                "IntelChina-AI/BitTern — check license; do not copy.",
                f"S026 §76 nearby-weight mapping; binary analog {R_BINARY_HEALING}.",
            ),
            "hawking_experiment_mapping": mapping(
                "Ternary snap is dead on composition. Binary nearby-island "
                "healing was tried and did not restore generation.",
                [R_BYTES, R_TERNARY_COMP, R_BINARY_HEALING],
                "CAT-Q's LM+ST is not the binary q2f-island / 0.5% CSR heal.",
                "CPU LM-style scale+threshold fit on one injured MLP tensor "
                "vs unmodulated 5-in-8, real X, then a 16-token argmax.",
            ),
            "current_verdict": verdict(
                RELATED_NEGATIVE,
                campaign_verdict=SCAR_TERNARY,
                receipts=[R_BYTES, R_TERNARY_COMP, R_BINARY_HEALING],
                measured={
                    "ternary_argmax_agree": s["ternary_argmax_agree"],
                    "teacher_argmax": s["ternary_teacher_argmax"],
                    "student_argmax": s["ternary_student_argmax"],
                    "binary_uniformly_injured": s["uniformly_injured"],
                    "binary_heals_that_reached_coherent_generation": s[
                        "healing_n_coherent"
                    ],
                },
                scope=(
                    "Ternary 5-in-8 composition death + binary island healing "
                    "as the nearby-weight analog. CAT-Q's specific LM+ST was "
                    "not run."
                ),
                remainder="Learnable modulation + softened ternarization on this parent.",
                experiment=cheapest(
                    "HX-CATQ-ONE-TENSOR-LM",
                    "Learnable modulation on one injured MLP tensor",
                    "Stream L0.up_proj (BINARY_HEALING earliest organ) or "
                    "L0.down_proj. Fit a per-group scale+threshold (soft "
                    "ternary) on real hold-set X vs snap 5-in-8. No GPU. "
                    "No second 27B.",
                    "Held-out rel_fro/gain vs snap-ternary and vs q2f. If it "
                    "does not beat snap, close CAT-Q for this body.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "ptqtp",
                "PTQTP",
                "PTQTP (structured trit planes)",
                "DOC-REPRESENTATION",
            ),
            "source_paper": paper(
                "PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models",
                "He Xiao, Runming Yang, Qingyao Yang, Wendong Xu, Zhen Li, "
                "Yupeng Su, Zhengwu Liu, Hongxia Yang, Ngai Wong",
                "2025-09",
                "2509.16989",
            ),
            "claimed_mechanism": (
                "Decompose each FP16 weight matrix into two structured "
                "ternary trit-planes plus scales: W ≈ α1 T1 + α2 T2. "
                "Multiplication-free additive inference (same MAC shape as "
                "1-bit, more expressiveness than one trit). Progressive "
                "approximation for global consistency. S026: structured trit "
                "planes, executed native (not reconstruct-to-dense)."
            ),
            "architecture_assumptions": (
                "Model-agnostic PTQ claimed on LLaMA3.x and Qwen3 families "
                "0.6B–70B. Two-plane additive reconstruction assumes the "
                "second plane carries residual magnitude the first cannot."
            ),
            "training_calibration_runtime": (
                "PTQ, ~1 hour in the paper vs 10–14 GPU-days QAT. Runtime: "
                "two ternary accumulations per output (or a fused dual-plane "
                "kernel). Two-pass reconstruct-then-GEMV is illegal here."
            ),
            "storage_vs_execution": (
                "Storage: 2 × ~1.58-bit planes + scales (~3.16 bpw before "
                "amortizing scales). Execution: additive trit MACs. "
                "BYTES_FRONTIER 5-in-8 is ONE plane packed; PTQTP is TWO "
                "structured planes."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: 1.58-class quality with binary-class arithmetic. "
                "Campaign: a second plane is a residual. HYBRID_OPERATOR "
                "already found distributed residuals do not heal binary "
                "under 2.25 bpw; PTQTP's residual is ternary-structured, "
                "which is a different code."
            ),
            "risks": [
                f"Single-plane ternary is {SCAR_TERNARY}.",
                "Two planes at ~3.16 bpw may exceed the 2.25 MLP floor they "
                "are trying to beat; bill both planes + scales.",
                "Without a fused dual-plane kernel the second plane is a "
                "second dispatch (N033).",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; hypothesis only.",
                "HeXiao-55/PTQTP — code 'to be released'; do not wait on it. Re-derive.",
                f"S026 §76; single-plane ternary scar {R_BYTES}.",
            ),
            "hawking_experiment_mapping": mapping(
                "One trit-plane (5-in-8) was natively executed.",
                [R_BYTES, R_TERNARY_COMP, R_FRACTIONAL, R_ONEBIT],
                "PTQTP's dual structured planes + progressive fit are not 5-in-8.",
                "CPU two-plane decomposition of one MLP tensor vs 5-in-8 and vs q2f.",
            ),
            "current_verdict": verdict(
                RELATED_NEGATIVE,
                campaign_verdict=SCAR_TERNARY,
                receipts=[R_BYTES, R_TERNARY_COMP, R_ONEBIT],
                measured={
                    "single_plane_active_bpw": s["ternary_bpw"],
                    "single_plane_complete_token_ns": s["ternary_ns"],
                    "onebit_best_survivor": s["onebit_best"],
                    "onebit_n_survive_at_matched_bytes": s["onebit_n_survive"],
                },
                scope="Single ternary plane. Dual trit-planes untested.",
                remainder="Structured two-plane decomposition + native additive kernel.",
                experiment=cheapest(
                    "HX-PTQTP-TWO-PLANE-ONE-TENSOR",
                    "Two trit-planes vs 5-in-8 vs q2f on one streamed tensor",
                    "Fit T1 then T2 on L31.gate_proj against real X (function "
                    "space, not Gaussian). Bill 2×trit + scales. Compare "
                    "held-out rel_fro to 5-in-8 (1.85) and q2f (2.25).",
                    "If two planes do not beat 5-in-8 at matched extra bytes, "
                    "close PTQTP. Do not write a Metal kernel first.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "onebit",
                "OneBit",
                "OneBit (decomposition binary)",
                "DOC-REPRESENTATION",
            ),
            "source_paper": paper(
                "OneBit: Towards Extremely Low-bit Large Language Models",
                "Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, "
                "Zhiyuan Liu, Weidong Liu, Wanxiang Che",
                "2024-02",
                "2402.11295",
                "NeurIPS 2024",
            ),
            "claimed_mechanism": (
                "Sign-Value-Independent Decomposition (SVID): a 1-bit sign "
                "matrix plus a compact value/magnitude factor, with a "
                "decomposition-based initialization and QAT to ~1 bit. "
                "S026 mapping: decomposition binary. This campaign's "
                "binary_g64 (sign + g64 f16 scale, no QAT) is the PTQ "
                "cousin of that body."
            ),
            "architecture_assumptions": (
                "Linear layers replaced by a OneBit Linear (sign ⊗ value). "
                "Paper used QAT on LLaMA-class dense models. Qwen3.8 GQA + "
                "DeltaNet hybrid is a different vehicle; MLP Linear maps "
                "are the overlapping organ."
            ),
            "training_calibration_runtime": (
                "Paper: QAT from a decomposed init. This campaign did NOT "
                "QAT: native binary g64 is a post-hoc sign+scale. Runtime: "
                "1-bit matvec + per-group scale. Binary geo kernel exists "
                "and is competent enough to MOVE token_ns."
            ),
            "storage_vs_execution": (
                "Storage: 1 sign bit + f16 mean-abs / 64 = 1.25 bpw "
                "(scales counted). Execution: packed sign matvec, dense_w=0. "
                "SVID's extra value factors, if not the group scale, must be billed."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: usable 1-bit LLMs via SVID+QAT. Campaign "
                "measurement: the PTQ binary body is the only family that "
                "moved COMPLETE_TOKEN_NS toward the roof, and it is dead "
                "for promotion (generation degenerates)."
            ),
            "risks": [
                "Speed without coherent generation is not a promotion (S022 §38).",
                "QAT SVID is untested; it is a different training regime "
                "(S026 §11 reopen condition) and still not a reason to "
                "ignore the PTQ injury map.",
                "Signed-symmetric absmax at 1 bit is the ZERO TENSOR "
                "(ONEBIT_FAMILIES / Doctor v2). A miss of that instrument "
                "is not '1-bit is impossible'.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; NeurIPS 2024.",
                "xuyuzhuang11/OneBit — check license; do not copy QAT code.",
                f"S026 §76; measured PTQ cousin {R_BYTES} + {R_BINARY_HEALING}.",
            ),
            "hawking_experiment_mapping": mapping(
                "Binary sign+scale MLP, natively executed, then healing islands.",
                [R_BYTES, R_BINARY_HEALING, R_FIRST_EXEC, R_ONEBIT],
                "OneBit SVID+QAT is not binary_g64 PTQ. The 1-bit BODY is the overlap.",
                "Do not re-run the dead PTQ body. S026 §63: the failed 1.25 "
                "binary may still be a DRAFT for Medusa/MTP (separate technique).",
            ),
            "current_verdict": verdict(
                TESTED_NEGATIVE,
                campaign_verdict=SCAR_BINARY,
                receipts=[R_BYTES, R_BINARY_HEALING, R_FIRST_EXEC],
                measured={
                    "binary_g64_active_bpw": s["binary_bpw"],
                    "binary_complete_token_ns": s["binary_ns"],
                    "q2f_complete_token_ns": q2f_ns,
                    "delta_ns": s["binary_delta_ns"],
                    "moved_toward_roof": s["binary_moved_ns"],
                    "died_at": s["binary_died_at"],
                    "uniformly_injured": s["uniformly_injured"],
                    "heals_that_reached_coherent_generation": s["healing_n_coherent"],
                    "injured_complete_token_ns": s["binary_injured_ns"],
                },
                scope=(
                    "PTQ binary g64 on Qwen3.8 MLP, native geo kernel, "
                    "dense_w=0. Faster than q2f (delta_ns "
                    f"{s['binary_delta_ns']}). mix_c emitted 16 copies of "
                    "token 271. Injury is uniform across 64 layers. No "
                    "tested island restored coherent generation while "
                    "staying faster than q2f."
                ),
                remainder=(
                    "SVID+QAT (paper training). S026 §11: a new training "
                    "regime may reopen, but the PTQ body is closed for promotion."
                ),
                experiment=cheapest(
                    "HX-ONEBIT-DO-NOT-RERUN-PTQ",
                    "Do not re-run PTQ binary g64 as a promotion candidate",
                    "The PTQ body is measured. Reopen only if QAT/SVID or a "
                    "coordinate transform changes the family (S026 §11). "
                    "The cheap next use is as a DRAFT head (see medusa_mtp).",
                    "No new promotion experiment. Draft-acceptance is Medusa/MTP.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "aqlm",
                "AQLM",
                "AQLM (additive codebooks)",
                "DOC-REPRESENTATION",
            ),
            "source_paper": paper(
                "Extreme Compression of Large Language Models via Additive Quantization",
                "Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, "
                "Elias Frantar, Artem Babenko, Dan Alistarh",
                "2024-01",
                "2401.06118",
                "ICML 2024",
            ),
            "claimed_mechanism": (
                "Additive quantization: each weight group is a SUM of "
                "learned codebook lookups (input-adaptive AQ), with joint "
                "codebook optimization across a transformer block. "
                "Homogeneous format, no hybrid sparse-outlier split."
            ),
            "architecture_assumptions": (
                "Dense Linear weights grouped into subvectors. Codebooks "
                "fit in registers/TG memory. Paper is LLaMA/Mistral PTQ. "
                "A kernel that dequantizes to dense then GEMVs has already "
                "lost (C4)."
            ),
            "training_calibration_runtime": (
                "PTQ codebook learning (block-wise). Runtime MUST be fused "
                "ADC (lookup-plus-accumulate), not reconstruct-to-GEMV. "
                "gravity_pq_matvec already exists in this tree and is not "
                "on the Qwen3.8 path."
            ),
            "storage_vs_execution": (
                "Storage: codebooks + indices (bill both; S026 §93). "
                "Execution: additive codebook accumulate. ONEBIT B6 is a "
                "routed PQ cousin at matched 2.00 bpw."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: SOTA extreme compression via additive "
                "codebooks. Campaign C4: do not port gravity_pq_matvec onto "
                "Qwen3.8 (NOT_WORTH_BUILDING_THE_QWEN38_PORT) — the existing "
                "kernel still pays GEMV. ONEBIT B6 beat the null on "
                "function-space error but was UNHEALTHY at matched bytes."
            ),
            "risks": [
                "C4: reconstruct-then-GEMV PQ does not beat q4 on this box.",
                "Codebook bytes hide behind a 2.00 headline if not billed.",
                "Joint block codebooks can couple layers the composition "
                "ladder must still climb independently.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; ICML 2024.",
                "vahe1994/AQLM — typically research-permissive; still re-derive, do not vendor.",
                f"S026 §76; C4 {s['c4_answer']}; ONEBIT B6 related.",
            ),
            "hawking_experiment_mapping": mapping(
                "PQ / routed codebook (ONEBIT B6) and gravity_pq (C4).",
                [R_C4, R_ONEBIT, R_REPR],
                "AQLM additive residual codebooks with block-joint fit are "
                "not GLM residual-PQ and not B6 k-means fragments.",
                "CPU additive 2-codebook reconstruction vs q2f on one "
                "streamed MLP tensor, real X. Only if that beats q2f at "
                "matched bytes is an ADC microbench in scope (C4 already "
                "named that as the cheapest remaining).",
            ),
            "current_verdict": verdict(
                RELATED_NEGATIVE,
                campaign_verdict=(
                    "C4 NOT_WORTH_BUILDING_THE_QWEN38_PORT; ONEBIT B6 "
                    "beats-null-but-unhealthy at matched 2.00 bpw"
                ),
                receipts=[R_C4, R_ONEBIT],
                measured={
                    "c4_answer": s["c4_answer"],
                    "onebit_best_survivor": s["onebit_best"],
                    "onebit_n_survive_at_matched_bytes": s["onebit_n_survive"],
                },
                scope=(
                    "Existing PQ kernel + B6 routed codebook on Qwen3.8 MLP. "
                    "AQLM's additive multi-codebook joint block fit was not run."
                ),
                remainder="Additive residual codebooks with fused ADC on this parent.",
                experiment=cheapest(
                    "HX-AQLM-TWO-CODEBOOK-ONE-TENSOR",
                    "Two additive codebooks vs q2f on one tensor, CPU",
                    "C4 already forbade the Qwen38 PQ port. The cheaper "
                    "discriminator is function-space error of 2 learned "
                    "additive codebooks on L31.gate_proj real X vs q2f, "
                    "billing codebook+index bytes.",
                    "If worse than q2f at matched bytes, close AQLM for this "
                    "body. Do not bind gravity_pq_matvec.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "vptq",
                "VPTQ",
                "VPTQ (vector quant + outlier residual)",
                "DOC-REPRESENTATION",
            ),
            "source_paper": paper(
                "VPTQ: Extreme Low-bit Vector Post-Training Quantization "
                "for Large Language Models",
                "Yifei Liu, Jicheng Wen, Yang Wang, Shengyu Ye, Li Lyna Zhang, "
                "Ting Cao, Cheng Li, Mao Yang",
                "2024-09",
                "2409.17066",
                "EMNLP 2024",
            ),
            "claimed_mechanism": (
                "Second-order (Hessian-aware) vector quantization of weight "
                "subvectors plus an outlier/residual channel so extreme "
                "<2-bit still has a high-precision escape hatch."
            ),
            "architecture_assumptions": (
                "Dense Linear weights, Hessian from calibration. Outlier "
                "residual is a sparse or high-precision sidecar. CSR-style "
                "sidecars have already been billed on this body."
            ),
            "training_calibration_runtime": (
                "PTQ with Hessian / second-order proxy. Runtime: VQ lookup "
                "+ residual add. Residual MUST be fused (HYBRID_OPERATOR / "
                "C3 lesson) or it is two passes."
            ),
            "storage_vs_execution": (
                "Storage: codebook + indices + outlier residual (indices "
                "are not free: BYTES_FRONTIER 2% CSR). Execution: fused "
                "VQ+residual. Dequant-to-dense is a fail."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: <2-bit with high accuracy via VQ+outliers. "
                "Campaign: VQ cousin (B6) unhealthy; residual cousins "
                "(HYBRID low-rank, 2% CSR) did not heal under 2.25."
            ),
            "risks": [
                f"Sparse residual {SCAR_SPARSE}.",
                f"Low-rank residual {SCAR_LOWRANK}.",
                "Hessian from too few rows is NNS-007 (undersampled fits).",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; EMNLP 2024.",
                "microsoft/VPTQ — typically MIT; still re-derive, do not vendor.",
                f"S026 §76; residual scars {R_HYBRID} + {R_BYTES}.",
            ),
            "hawking_experiment_mapping": mapping(
                "Vector codebook (B6/C4) + outlier residual (HYBRID / CSR).",
                [R_C4, R_ONEBIT, R_HYBRID, R_BYTES],
                "VPTQ's Hessian-aware VQ + named outlier channel is not B6 k-means "
                "and not binary+lowrank r=8.",
                "CPU VQ (small M) + residual on one tensor vs q2f and vs HYBRID r=8.",
            ),
            "current_verdict": verdict(
                RELATED_NEGATIVE,
                campaign_verdict=f"{SCAR_LOWRANK}; {SCAR_SPARSE}",
                receipts=[R_HYBRID, R_BYTES, R_C4, R_ONEBIT],
                measured={
                    "hybrid_beats_q2f": s["hybrid_beats_q2f"],
                    "hybrid_died_at": s["hybrid_died_at"],
                    "sparse_csr_bytes": s["sparse_csr_bytes"],
                    "sparse_binary_bytes": s["sparse_binary_bytes"],
                    "sparse_active_bpw": s["sparse_bpw"],
                    "sparse_complete_token_ns": s["sparse_ns"],
                    "c4_answer": s["c4_answer"],
                },
                scope=(
                    "VQ and residual families as implemented here. VPTQ's "
                    "Hessian-aware codebook was not run."
                ),
                remainder="Hessian-aware VQ + fused outlier residual on this parent.",
                experiment=cheapest(
                    "HX-VPTQ-VQ-PLUS-RESIDUAL-ONE-TENSOR",
                    "Small VQ + residual vs q2f and vs HYBRID r=8",
                    "One streamed tensor, real X. Bill codebook+indices+"
                    "residual. Compare to HYBRID r=8 (already dead) and q2f.",
                    "If it does not beat HYBRID r=8 held-out, close VPTQ. "
                    "Do not write a kernel.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "caldera",
                "CALDERA",
                "CALDERA (low-rank + low-precision)",
                "DOC-HEALING",
            ),
            "source_paper": paper(
                "Compressing Large Language Models using Low Rank and "
                "Low Precision Decomposition",
                "Rajarshi Saha, Naomi Sagan, Varun Srivastava, "
                "Andrea J. Goldsmith, Mert Pilanci",
                "2024-05",
                "2405.18886",
                "NeurIPS 2024",
            ),
            "claimed_mechanism": (
                "W ≈ Q + L R with Q, L, R all quantized (calibration-aware "
                "low-precision decomposition). Low-rank factors capture top "
                "energy; Q is a low-precision backbone for the residual."
            ),
            "architecture_assumptions": (
                "Weight matrices with usable low-rank energy in function "
                "space (not just SVD of W). Qwen3.8 MLP down_proj inverted "
                "some Q80 rankings; G034 matched-bit low-rank was 2.93× q3 "
                "error. Activation-aware rank on real X is the only legal "
                "screen (NNS-001, NNS-014)."
            ),
            "training_calibration_runtime": (
                "PTQ iterative LPLR factorization on calibration X. Runtime: "
                "low-rank GEMVs + quantized residual. Two unfused passes "
                "were slower (NS-030 5–13×). HYBRID_OPERATOR fused the "
                "binary+lowrank cousin natively."
            ),
            "storage_vs_execution": (
                "Storage: quantized Q + L + R (all billed). Execution: "
                "fused (Q x + L (R x)), dense_w=0. Reconstructing W is a fail."
            ),
            "expected_useful_organs": ["mlp_down", "mlp_gate_up"],
            "expected_physical_win": (
                "Paper claim: sub-2.5 bpw competitive with QuIP# via Q+LR. "
                "Campaign measurement: binary + distributed low-rank "
                "correction never restored held-out activations under the "
                "2.25 bpw / 27.55 ms joint constraint."
            ),
            "risks": [
                "HYBRID_OPERATOR: even r=256 (body 2.285 > 2.25) rel_fro=0.4798.",
                "C3: fusion of low-rank+sparse NOT_WORTH_BUILDING on accounting.",
                "G034 / NNS-014: activation-aware low-rank has its own closed doors.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; NeurIPS 2024.",
                "pilancilab/caldera — check license; re-derive.",
                f"S026 §76; measured cousin {R_HYBRID}; design {R_C3}.",
            ),
            "hawking_experiment_mapping": mapping(
                "Low-rank residual on a cheap binary backbone, fused native.",
                [R_HYBRID, R_C3],
                "CALDERA quantizes Q, L and R; HYBRID used binary Q plus f16 "
                "low-rank. Same family, not the same codec.",
                "Do not re-run the dead fused hybrid as a promotion. Reopen "
                "only if Q/L/R are all low-precision AND a coordinate "
                "transform changed the energy (S026 §11) — that is N044 then "
                "a one-tensor CALDERA fit, not a new kernel.",
            ),
            "current_verdict": verdict(
                TESTED_NEGATIVE,
                campaign_verdict=SCAR_LOWRANK,
                receipts=[R_HYBRID, R_C3],
                measured={
                    "coherent_hybrid_beats_q2f": s["hybrid_beats_q2f"],
                    "died_at": s["hybrid_died_at"],
                    "finding": s["hybrid_finding"],
                    "c3_answer": s["c3_answer"],
                },
                scope=(
                    "Fused native binary + distributed low-rank on Qwen3.8 "
                    "MLP, real X, dense_w=0. No correction under the 1.0 "
                    "extra-bpw budget restored held-out activations. C3 "
                    "independently closed low-rank+sparse fusion on accounting."
                ),
                remainder=(
                    "CALDERA's jointly-quantized Q+LR (all low-precision) "
                    "is a codec variant. The residual-healing claim is what "
                    "HYBRID already killed."
                ),
                experiment=cheapest(
                    "HX-CALDERA-DO-NOT-RERUN-HYBRID",
                    "Do not re-run fused low-rank residual as a promotion",
                    "HYBRID_OPERATOR is the measurement. A quantized-Q "
                    "variant is only in scope after N044 moves the barrier.",
                    "No promotion experiment until a reopen condition holds.",
                    depends_on=["N044"],
                ),
            ),
        },
        {
            "technique_identity": identity(
                "squeezellm",
                "SqueezeLLM",
                "SqueezeLLM (sensitivity dense+sparse)",
                "DOC-REPRESENTATION",
            ),
            "source_paper": paper(
                "SqueezeLLM: Dense-and-Sparse Quantization",
                "Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, "
                "Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer",
                "2023-06",
                "2306.07629",
                "ICML 2024",
            ),
            "claimed_mechanism": (
                "Sensitivity-weighted (Fisher/Hessian) bit assignment: a "
                "dense low-bit body plus a sparse high-precision hold-out "
                "of sensitive entries. Dense-and-sparse, not uniform quant."
            ),
            "architecture_assumptions": (
                "A meaningful per-weight sensitivity from calibration. "
                "Sparse sidecar is CSR-class. This campaign already billed "
                "CSR: 2% nnz on binary cost ~2.07e9 index bytes and SLOWED "
                "the graph."
            ),
            "training_calibration_runtime": (
                "PTQ sensitivity census + threshold. Runtime: dense low-bit "
                "matvec + sparse gather. Unfused gather lost (G070 / N033). "
                "BINARY_HEALING sparse_05 was one of the island candidates."
            ),
            "storage_vs_execution": (
                "Storage: dense codes + sparse values + INDICES (bill "
                "indices). Execution: fused dense+sparse or it is two "
                "passes. BYTES_FRONTIER fused 2% CSR still lost on ns."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: better perplexity at the same bpw by "
                "protecting sensitive weights. Campaign: index-carrying "
                "sparsity cost more than the byte win on this MLP graph."
            ),
            "risks": [
                f"{SCAR_SPARSE}: csr_bytes={s['sparse_csr_bytes']} vs "
                f"binary_bytes={s['sparse_binary_bytes']} at nnz="
                f"{s['sparse_nnz_frac']}; complete_token_ns={s['sparse_ns']} "
                f"vs q2f {q2f_ns}.",
                "Sensitivity from too few tokens is NNS-007.",
                "GQA quality floor: sensitivity-sparse attention is a "
                "different, more expensive organ.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; ICML 2024.",
                "SqueezeLLM reference code — check license; re-derive the census.",
                f"S026 §76; sparse scar {R_BYTES}.",
            ),
            "hawking_experiment_mapping": mapping(
                "Fused binary + 2% CSR residual; BINARY_HEALING sparse_05 island.",
                [R_BYTES, R_BINARY_HEALING, R_C3],
                "SqueezeLLM's Fisher-weighted choice of WHICH entries are "
                "dense is not a uniform 2% CSR of the residual.",
                "CPU sensitivity census on one organ from existing capture, "
                "then dense-body + sparse-outliers vs uniform 2% CSR.",
            ),
            "current_verdict": verdict(
                RELATED_NEGATIVE,
                campaign_verdict=SCAR_SPARSE,
                receipts=[R_BYTES, R_BINARY_HEALING],
                measured={
                    "sparse_active_bpw": s["sparse_bpw"],
                    "sparse_complete_token_ns": s["sparse_ns"],
                    "sparse_delta_ns": s["sparse_delta_ns"],
                    "sparse_moved_toward_roof": s["sparse_moved_ns"],
                    "csr_bytes": s["sparse_csr_bytes"],
                    "binary_bytes": s["sparse_binary_bytes"],
                    "nnz_frac": s["sparse_nnz_frac"],
                    "binary_heals_coherent": s["healing_n_coherent"],
                },
                scope=(
                    "Uniform 2% CSR on a binary body, fused native, plus "
                    "sparse_05 as a healing island. Fisher-weighted selection "
                    "of the dense/sparse split was not run."
                ),
                remainder="Sensitivity-chosen dense+sparse split on this parent.",
                experiment=cheapest(
                    "HX-SQUEEZELLM-SENSITIVITY-CENSUS",
                    "Per-weight sensitivity census on one MLP organ, CPU",
                    "From existing capture_diverse2 rows (no new 27B). "
                    "Rank |W * X_rms| or a diagonal Fisher proxy. Protect "
                    "top 0.5% in f16, quantize the rest binary, bill indices. "
                    "Compare held-out to uniform 2% CSR.",
                    "If the sensitivity split does not beat uniform CSR "
                    "held-out, close SqueezeLLM. Do not write a gather kernel.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "kivi",
                "KIVI",
                "KIVI (asymmetric KV)",
                "DOC-STATE",
            ),
            "source_paper": paper(
                "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
                "Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, "
                "Zhaozhuo Xu, Braverman, Beidi Chen, Cong Liu",
                "2024-02",
                "2402.02750",
                "ICML 2024",
            ),
            "claimed_mechanism": (
                "Asymmetric KV-cache quant: K and V get different bit-widths "
                "(paper: K 2-bit grouping-along-channel, V 2-bit "
                "grouping-along-token) because their outlier structure "
                "differs. Tuning-free PTQ on the cache, not the weights."
            ),
            "architecture_assumptions": (
                "Prefill/decode KV cache for MHA/GQA. Qwen3.8: 16 GQA "
                "layers (KV grows with seq) + 48 DeltaNet layers "
                "(recurrent state, constant in seq). KIVI does not apply "
                "to DeltaNet state. Production GQA KV is f32 today."
            ),
            "training_calibration_runtime": (
                "Tuning-free; per-cache grouping at runtime. A kernel exists "
                "in-tree for int4 KV (mha_decode_flash_int4kv_parity) and is "
                "NOT wired into production. Capability cost is ABSENT."
            ),
            "storage_vs_execution": (
                "Storage: quantized K and V plus per-group scales (session "
                "state, not model EBPW — but S026 §93 still counts it in "
                "the production footprint). Execution: dequant-on-read inside "
                "MHA, not a second copy in f32."
            ),
            "expected_useful_organs": ["gqa_attention"],
            "expected_physical_win": (
                "Paper claim: 2-bit KV with negligible quality loss, large "
                "memory cut. Campaign: at q4 c=4 32K, SESSION_STATE_x_c "
                "already exceeds MODEL_BYTES (PREFILL_KV). Asymmetric KV is "
                "the first state lever that is not a weight-floor fight. "
                "Unmeasured on this body."
            ),
            "risks": [
                "GQA is the quality floor (NOETIC_GQA_DESIGN / ORGAN_FRONTIERS). "
                "KV quant is a different axis but not a free one.",
                "int4 KV kernel exists; wiring it is not a measurement of KIVI.",
                "DeltaNet state (156.9 MiB, const in seq) is a separate organ.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; ICML 2024.",
                "KIVI reference impl — check license; do not copy into Metal.",
                f"S026 §50-58; related census {R_PREFILL}.",
            ),
            "hawking_experiment_mapping": mapping(
                "KV precision candidates (f16, int4) were CENSUSED, not measured "
                "for capability. Symmetric int4 ≠ KIVI asymmetric 2-bit.",
                [R_PREFILL, R_GQA, R_FRONTIERS],
                "PREFILL_KV int4 candidate is a flash-MHA test kernel, not KIVI.",
                "CPU K-vs-V dynamic-range census on any existing GQA dump; "
                "if absent, stream 128 tokens of parent K/V (no second 27B) "
                "and report per-channel vs per-token outlier stats. That "
                "discriminator decides whether K2V4 or K4V2 is even motivated.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                measured={
                    "related_prefill_answer": s["prefill_answer"],
                    "q4_c4_32k_state_exceeds_weights": s["q4_c4_32k_state_exceeds"],
                    "kv_production_dtype": s["kv_production_dtype"],
                    "int4_kv_kernel_exists": s["kv_int4_kernel_exists"],
                    "int4_kv_wired_into_production": s["kv_int4_wired"],
                },
                scope="No Hawking receipt measures asymmetric KV quant on Qwen3.8 GQA.",
                remainder="The whole technique. PREFILL_KV is a related census, not a verdict.",
                experiment=cheapest(
                    "HX-KIVI-K-VS-V-RANGE",
                    "K vs V outlier-axis census (CPU)",
                    "PREFILL_KV already showed session state can exceed "
                    "weights. The cheapest KIVI discriminator is whether K "
                    "and V even HAVE different outlier axes on this GQA. "
                    "No Metal re-bench. No model-wide decode.",
                    "If K and V outlier axes match, KIVI's asymmetry claim "
                    "is unmotivated here; fall back to a symmetric KV-bit "
                    "probe later. If they differ, name K2V4 vs K4V2 as the "
                    "next (still CPU) reconstruction probe.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "minicache",
                "MiniCache",
                "MiniCache (depth state merge)",
                "DOC-STATE",
            ),
            "source_paper": paper(
                "MiniCache: KV Cache Compression via Depth-Wise Attention",
                "Akide Liu, Jing Liu, Zizheng Pan, Yefei He, "
                "Gholamreza Haffari, Bohan Zhuang",
                "2024-05",
                "2405.14366",
                "ICML 2024",
            ),
            "claimed_mechanism": (
                "Cross-layer KV merging: adjacent-layer caches are similar "
                "enough to interpolate/merge, cutting KV depth. S026: depth "
                "state merge. ZERO_EXECUTION of redundant state, not of compute."
            ),
            "architecture_assumptions": (
                "Stacked GQA/MHA layers with comparable KV geometry. "
                "Qwen3.8 has only 16 GQA layers (the rest are DeltaNet). "
                "Merging GQA with DeltaNet state is a type error. MiniCache "
                "is a GQA-only hypothesis here."
            ),
            "training_calibration_runtime": (
                "Inference-time merge; some variants need a small "
                "interpolation fit. Runtime: fewer KV bytes, possible extra "
                "blend. Must not keep both the merged cache AND the originals."
            ),
            "storage_vs_execution": (
                "Storage: merged KV (depth reduced). Execution: attention "
                "against the merged cache. Token-inflation / long-context "
                "accounting still applies (S026 §50-58)."
            ),
            "expected_useful_organs": ["gqa_attention"],
            "expected_physical_win": (
                "Paper claim: KV memory cut by merging similar depths. "
                "Campaign: GQA KV at 16K is 2.15 GiB f32 (PREFILL_KV); "
                "depth-merge of 16 layers is a smaller lever than KIVI's "
                "bit cut, but orthogonal. Unmeasured."
            ),
            "risks": [
                "16 GQA layers is a thin stack; merge pairs may not exist.",
                "DeltaNet recurrent state is not KV and does not merge this way.",
                "Merging away a quality-floor organ is how you get silent capability loss.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; ICML 2024.",
                "MiniCache code — check license; re-derive.",
                f"S026 §50-58; related {R_PREFILL}.",
            ),
            "hawking_experiment_mapping": mapping(
                "None. PREFILL_KV bills GQA KV bytes; it does not merge depths.",
                [R_PREFILL],
                "Byte census ≠ depth-merge quality.",
                "Cosine between adjacent GQA-layer K (and V) on a short "
                "captured prefix. If cosine is not high, close MiniCache.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                scope="No Hawking receipt measures cross-layer KV merge on Qwen3.8 GQA.",
                remainder="The whole technique.",
                experiment=cheapest(
                    "HX-MINICACHE-ADJACENT-KV-COSINE",
                    "Adjacent GQA-layer K/V cosine on a short prefix, CPU",
                    "If a 16-token K/V dump exists, use it. Else stream a "
                    "short parent prefix (no second resident 27B) for the 16 "
                    "GQA layers only. Cosine(K_i, K_{i+1}) and same for V.",
                    "If adjacent cosine is not high enough to beat a "
                    "constant-mean null by a stated margin, close MiniCache. "
                    "Do not implement a merge kernel first.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "h2o",
                "H2O",
                "H2O (heavy-hitter state)",
                "DOC-STATE",
            ),
            "source_paper": paper(
                "H2O: Heavy-Hitter Oracle for Efficient Generative Inference "
                "of Large Language Models",
                "Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, "
                "Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, "
                "Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen",
                "2023-06",
                "2306.14048",
                "NeurIPS 2023",
            ),
            "claimed_mechanism": (
                "KV eviction: keep a small set of heavy-hitter tokens "
                "(high cumulative attention) plus recent tokens; drop the "
                "rest. S026: heavy-hitter state. This is ZERO_EXECUTION of "
                "cold prefix state, not a weight codec."
            ),
            "architecture_assumptions": (
                "Attention that produces a meaningful per-token mass. GQA "
                "yes; DeltaNet is a recurrent summary and does not have a "
                "prefix-shareable KV to evict. H2O is GQA-only here."
            ),
            "training_calibration_runtime": (
                "Runtime policy, no training. Cost: attention-score "
                "bookkeeping + gather of surviving KV. A data-dependent "
                "inner branch is a kernel-competence smell (N003)."
            ),
            "storage_vs_execution": (
                "Storage: reduced KV (budgeted slots). Execution: sparse "
                "along sequence, dense along head. Score metadata must be billed."
            ),
            "expected_useful_organs": ["gqa_attention"],
            "expected_physical_win": (
                "Paper claim: large KV cut with small quality loss. "
                "Campaign: long-context SESSION_STATE is the production "
                "problem (PREFILL_KV). Heavy-hitters are untested; they "
                "interact with AgentOS long-context, not with the 2.25 MLP floor."
            ),
            "risks": [
                "Evicting tool/JSON/schema tokens is S026 §82 / §109 "
                "(free capability ≠ free information).",
                "Attention-mass on a math prompt may not match a code/JSON prompt.",
                "DeltaNet layers do not participate; overselling whole-model KV cut is a lie.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; NeurIPS 2023.",
                "H2O reference code — check license; re-derive the policy.",
                f"S026 §50-58; related {R_PREFILL}.",
            ),
            "hawking_experiment_mapping": mapping(
                "None on heavy-hitter eviction. PREFILL_KV is the byte problem statement.",
                [R_PREFILL],
                "A footprint census is not an eviction policy.",
                "On one GQA layer, 1K-token CPU attention-mass histogram "
                "from existing scores if any; else a short parent prefix. "
                "Report heavy-hitter rank-frequency. If mass is flat, close H2O.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                scope="No Hawking receipt measures heavy-hitter KV eviction on Qwen3.8 GQA.",
                remainder="The whole technique.",
                experiment=cheapest(
                    "HX-H2O-ATTENTION-MASS-HISTOGRAM",
                    "Per-token attention-mass histogram on one GQA layer, CPU",
                    "Need a score tensor. If none on disk, a short CPU "
                    "attention on streamed Q/K for one layer is the probe. "
                    "Do not implement an eviction kernel.",
                    "If cumulative mass is not heavy-tailed, H2O's oracle "
                    "has nothing to grab on this mixer. Close it.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "mixture_of_depths",
                "Mixture-of-Depths",
                "Mixture-of-Depths (conditional depth)",
                "DOC-CONDITIONAL",
            ),
            "source_paper": paper(
                "Mixture-of-Depths: Dynamically allocating compute in "
                "transformer-based language models",
                "David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, "
                "Peter Conway Humphreys, Adam Santoro (Google DeepMind)",
                "2024-04",
                "2404.02258",
            ),
            "claimed_mechanism": (
                "Per-token routing over depth: some tokens skip blocks "
                "(identity residual), a learned router spends compute on "
                "hard tokens. S026: conditional depth / ZERO_EXECUTION of "
                "this token at this layer. Capacity is a static budget."
            ),
            "architecture_assumptions": (
                "Uniform stacked blocks and a router trained (or distilled) "
                "to skip. Qwen3.8 already MIXES architectures (48 DeltaNet "
                "+ 16 GQA) — that is not MoD. MoD would skip inside a "
                "family. Skipping a quality-floor GQA layer is high-risk."
            ),
            "training_calibration_runtime": (
                "Paper trains the router. A training-free probe can still "
                "ask: which layers are already near-identity on real X? "
                "That is the cheapest Hawking experiment and does not "
                "implement MoD."
            ),
            "storage_vs_execution": (
                "Storage: full weights still exist (unless elimination, "
                "S026 §29). Execution: skipped layers do not run THIS "
                "token (ZERO_EXECUTION). Router bytes and capacity tables "
                "are billed. Weights not executed still count in complete EBPW."
            ),
            "expected_useful_organs": [
                "mlp_gate_up",
                "mlp_down",
                "gqa_attention",
                "deltanet",
            ],
            "expected_physical_win": (
                "Paper claim: ~50% fewer FLOPs at matched quality. Campaign: "
                "decode is bandwidth-bound on the q4 incumbent (GPU_LEDGER). "
                "Skipping a layer wins only if it also skips the weight "
                "stream, or if the organ is compute-bound (ternary was)."
            ),
            "risks": [
                "Complete EBPW still counts skipped weights unless they are ELIMINATED.",
                "A router that fires densely is a second attention.",
                "NNS-029: activation sparsity / uniform bit-descent is not a "
                "clean path under the Qwen3.8 coherent floor — cousin warning.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive.",
                "DeepMind paper; no in-tree impl. Do not vendor an MoD trainer.",
                "S026 §39-46, §76; NNS-029 related.",
            ),
            "hawking_experiment_mapping": mapping(
                "None as MoD. Closest: identity/skip as a diagnostic (scale trap, "
                "ZERO_DENSE). Not a router.",
                [R_NNS],
                "A skip-layer ablation is not a trained Mixture-of-Depths router.",
                "Identity-residual ablation of one layer at a time on real "
                "held-out X (CPU). Rank layers by how little they move the "
                "residual. That names ZERO_EXECUTION candidates.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                scope="No Hawking receipt trains or measures a depth router on Qwen3.8.",
                remainder="The whole technique.",
                experiment=cheapest(
                    "HX-MOD-SKIP-LAYER-ABLATION",
                    "Identity-skip ablation across layers on real X, CPU",
                    "Teacher-forced: replace layer i mixer+MLP with identity "
                    "residual, score held-out rel_fro/gain vs the unskipped "
                    "teacher. Stream weights. No router training.",
                    "A layer whose skip stays healthy is a ZERO_EXECUTION "
                    "candidate. If none do, close MoD as a Qwen3.8 decode lever.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "prosparse",
                "ProSparse",
                "ProSparse (activation sparsity)",
                "DOC-CONDITIONAL",
            ),
            "source_paper": paper(
                "ProSparse: Introducing and Enhancing Intrinsic Activation "
                "Sparsity within Large Language Models",
                "Chenyang Song, Xu Han, Zhengyan Zhang, Shengding Hu, "
                "Xiyuan Li, Zhiyuan Liu, Maosong Sun et al.",
                "2024-02",
                "2402.13516",
                "COLM 2024",
            ),
            "claimed_mechanism": (
                "Replace ReLU/SwiGLU-like activations so the MLP hidden is "
                "intrinsically sparse, then exploit that sparsity at "
                "runtime. S026: activation sparsity; ZERO_EXECUTION of "
                "zero channels, only if STRUCTURED and it beats dense."
            ),
            "architecture_assumptions": (
                "MLP activations that can go exactly zero. Qwen3.8 uses "
                "SwiGLU (SiLU gate): near-sparse, not ReLU-sparse. "
                "ProSparse typically SWAPS the activation — that is a "
                "behavior/representation change, not a PTQ of the current "
                "SwiGLU. Tabula vs physical (S026 §71-74) applies if the "
                "swap changes behavior."
            ),
            "training_calibration_runtime": (
                "Paper includes continued training to induce sparsity. A "
                "training-free probe is: how sparse is SwiGLU already on "
                "real X? If not sparse, ProSparse is a training program, "
                "not a PTQ."
            ),
            "storage_vs_execution": (
                "Storage: weights unchanged unless the activation swap "
                "forces a refit. Execution: skip zero MAC/channels. "
                "BYTES_FRONTIER ternary already noted: skipping a zero "
                "trit saves an FMA, not a DRAM load, unless zeros are "
                "structured enough to skip bytes."
            ),
            "expected_useful_organs": ["mlp_gate_up", "mlp_down"],
            "expected_physical_win": (
                "Paper claim: ~2× sparsity with small quality loss, faster "
                "MLP. Campaign NNS-029: activation sparsity (~2× MLP max) "
                "is not a clean path under the Qwen3.8 coherent floor. "
                "Unstructured skip does not cut DRAM on a bandwidth-bound graph."
            ),
            "risks": [
                "NNS-029 related negative on uniform activation-sparsity as a path.",
                "Unstructured zeros do not skip DRAM (ternary 5-in-8 note).",
                "Activation swap is a Tabula/behavior change unless proven otherwise.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; COLM 2024.",
                "ProSparse code — check license; a swap of SwiGLU is not a drop-in.",
                "S026 §41-46, §76; NNS-029.",
            ),
            "hawking_experiment_mapping": mapping(
                "NNS-029 (activation sparsity not a clean path). Ternary zero-MAC elision.",
                [R_NNS, R_BYTES],
                "A histogram of SwiGLU zeros is not ProSparse training.",
                "Activation sparsity histogram on existing capture (post-SwiGLU "
                "if present, else SiLU(gate)*up on streamed tensors). "
                "Structured vs unstructured. If unstructured, close as a DRAM lever.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                measured={
                    "nns_029": "activation sparsity / uniform bit-descent is not a clean path under the Qwen3.8 coherent floor (related, not this measurement)",
                    "ternary_zero_macs_elided_did_not_skip_dram": True,
                },
                scope=(
                    "No Hawking receipt measures ProSparse (activation swap + "
                    "induced sparsity) on Qwen3.8. NNS-029 is a related "
                    "negative, not a substitute verdict on ProSparse itself."
                ),
                remainder="The whole technique. Related NNS-029 is cited as context, not as a verdict receipt.",
                experiment=cheapest(
                    "HX-PROSPARSE-SWIGLU-SPARSITY-HISTOGRAM",
                    "SwiGLU activation sparsity histogram on real X, CPU",
                    "Existing capture preferred. Else stream L0 gate/up, "
                    "compute silu(g)*u sparsity. Report fraction exactly-0 "
                    "and near-0, structured (channel) vs scattered.",
                    "If sparsity is unstructured and DRAM-bound, close "
                    "ProSparse as a decode lever. Structured channel-sparsity "
                    "would reopen a fused skip-channel kernel, not this registry.",
                ),
            ),
        },
        {
            "technique_identity": identity(
                "medusa_mtp",
                "Medusa/MTP",
                "Medusa/MTP (multi-token decode)",
                "DOC-DECODE",
            ),
            "source_paper": paper(
                "Medusa: Simple LLM Inference Acceleration Framework with "
                "Multiple Decoding Heads; and Better & Faster Large Language "
                "Models via Multi-token Prediction",
                "Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, "
                "Jason D. Lee, Deming Chen, Tri Dao (Medusa); "
                "Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, "
                "David Lopez-Paz, Gabriel Synnaeve (MTP)",
                "2024-01 / 2024-04",
                "2401.10774",
                "Medusa: ICML 2024; MTP: arXiv 2404.19737",
                extra="Two papers, one S026 mechanism: extra heads predict "
                "future tokens so one forward accepts multiple tokens. "
                "Qwen3.8 checkpoints may ship native MTP — census first.",
            ),
            "claimed_mechanism": (
                "Reduce PASSES per generated token, not bits per weight. "
                "Medusa: attach extra decoding heads to a frozen/finetuned "
                "backbone and verify a tree of draft tokens. MTP: train "
                "the model to predict several future tokens; at decode, "
                "use those heads as a same-model draft. S026 §59-64, §121. "
                "S026 §63: the failed 1.25 binary may be a useful DRAFT."
            ),
            "architecture_assumptions": (
                "A hidden state that extra heads can read (final RMSNorm / "
                "layer-out). Speculative verification needs the target "
                "model's logits. Native Qwen MTP, if present in the parent "
                "weight_map, is the first thing to census — attaching Medusa "
                "heads is the fallback."
            ),
            "training_calibration_runtime": (
                "Medusa: head training (hours). MTP: pretraining or a "
                "shipped head. Runtime: extra head GEMVs + verification. "
                "Accepted-tokens/forward and accepted-tokens/byte are the "
                "metrics (S026 §94, §95), not tok/s of a rejected draft."
            ),
            "storage_vs_execution": (
                "Storage: auxiliary heads (billed in complete EBPW, S026 §93). "
                "Execution: draft then verify. A draft that is 1.25-bit "
                "binary (already faster, already dead as a target) is a "
                "different ROLE (S026 §64: negatives are purpose-scoped)."
            ),
            "expected_useful_organs": ["lm_head", "sampling", "gqa_attention"],
            "expected_physical_win": (
                "Paper claim: 2–3× accepted tokens per forward. Campaign: "
                "decode is bandwidth-bound; extra heads cost bytes. Win "
                "only if accepted-tokens/byte rises. Unmeasured."
            ),
            "risks": [
                "Aux heads that are never accepted are dead weight in EBPW.",
                "Tree attention / verification kernels are new Metal work "
                "(not this CPU registry).",
                "Using the injured binary as a draft can inject its token-271 "
                "collapse into the proposal distribution.",
            ],
            "licensing_provenance": provenance(
                "arXiv non-exclusive; Medusa ICML 2024.",
                "FasterDecoding/Medusa and MTP training stacks — check licenses; "
                "census native Qwen heads before attaching any.",
                "S026 §59-64, §76, §63 binary-as-draft.",
            ),
            "hawking_experiment_mapping": mapping(
                "Binary g64 is a measured FAST and DEAD target; S026 §63 "
                "reopens it as a possible DRAFT, which is a different purpose.",
                [R_BYTES, R_BINARY_HEALING, R_FIRST_EXEC],
                "Speed of binary as a TARGET is not an MTP acceptance rate.",
                "CPU census: does the qualified parent weight_map contain "
                "mtp/medusa/aux-head tensors? That is one streamed index "
                "JSON, no weights. If yes, name them. If no, binary-as-draft "
                "is the next (GPU) experiment and is NOT this lane.",
            ),
            "current_verdict": verdict(
                UNTESTED,
                campaign_verdict=None,
                receipts=[],
                measured={
                    "binary_as_draft_is_s026_63_hypothesis": True,
                    "binary_target_died_at": s["binary_died_at"],
                    "binary_moved_token_ns": s["binary_moved_ns"],
                },
                scope=(
                    "No Hawking receipt measures Medusa heads or MTP "
                    "acceptance on Qwen3.8. Binary speed is a TARGET "
                    "measurement, not a draft-acceptance measurement."
                ),
                remainder="Native MTP census, then (if absent) binary-as-draft — GPU, not this lane.",
                experiment=cheapest(
                    "HX-MTP-PARENT-WEIGHTMAP-CENSUS",
                    "Census parent index for MTP/Medusa tensors (CPU, no weights)",
                    "Read model.safetensors.index.json (and config.json) of "
                    "the qualified parent. Count keys matching mtp/medusa/"
                    "multi_token/aux_head. Stream the index only.",
                    "If native MTP tensors exist, the next experiment is "
                    "their byte bill + a decode-acceptance run (GPU, not "
                    "N043). If none, record ABSENT and leave binary-as-draft "
                    "to a DOC-DECODE GPU lane.",
                ),
            ),
        },
    ]
    return techniques


def build_campaign_cross_references(seed: dict[str, Any]) -> dict[str, Any]:
    """The five scars N043 names, whether or not they are a paper."""
    s = seed
    return {
        "shared_basis": {
            "maps_onto_techniques": [],
            "receipts": [R_SHARED_K, R_SHARED_C],
            "also": [R_C1, R_BYTES],
            "verdict": SCAR_SHARED_BASIS,
            "measured": {
                "kernel_competent": s["shared_kernel_competent"],
                "byte_win_translates_to_token_ns": s["shared_byte_win_translates"],
                "k2_active_bpw": s["shared_k2_bpw"],
                "k2_complete_token_ns": s["shared_k2_ns"],
                "coherent_shared_basis_beats_q2f": s["shared_beats_q2f"],
                "operating_point_active_bpw": (s["shared_op"] or {}).get("active_bpw"),
                "operating_point_coherent": (s["shared_op"] or {}).get("coherent"),
                "c1_verdict": s["c1_verdict"],
                "c1_failure": s["c1_failure"],
            },
            "reading": (
                "SHARED_BASIS_KERNEL: fused K=2 kernel is competent and the "
                "0.53-bpw byte win translates to token_ns. "
                "SHARED_BASIS_COHERENT: no coherent point beats q2f on both "
                "density and ns; K=2 dies at held_out_activation. Competent "
                "kernel, dead below 2.25."
            ),
        },
        "binary": {
            "maps_onto_techniques": ["onebit"],
            "receipts": [R_BYTES, R_BINARY_HEALING],
            "also": [R_FIRST_EXEC],
            "verdict": SCAR_BINARY,
            "measured": {
                "active_bpw": s["binary_bpw"],
                "complete_token_ns": s["binary_ns"],
                "delta_ns": s["binary_delta_ns"],
                "moved_toward_roof": s["binary_moved_ns"],
                "uniformly_injured": s["uniformly_injured"],
                "heals_coherent": s["healing_n_coherent"],
            },
            "reading": (
                "binary_g64 is the only BYTES_FRONTIER family that moved "
                "COMPLETE_TOKEN_NS toward the roof, and BINARY_HEALING "
                "found the injury uniform across 64 layers with no island "
                "restoring coherent generation."
            ),
        },
        "low_rank_residual": {
            "maps_onto_techniques": ["caldera", "vptq"],
            "receipts": [R_HYBRID],
            "also": [R_C3],
            "verdict": SCAR_LOWRANK,
            "measured": {
                "coherent_hybrid_beats_q2f": s["hybrid_beats_q2f"],
                "died_at": s["hybrid_died_at"],
            },
            "reading": (
                "HYBRID_OPERATOR: no distributed low-rank correction under "
                "the joint 2.25 bpw / 27.55 ms constraint restored held-out "
                "activations on real X. The residual never heals."
            ),
        },
        "ternary": {
            "maps_onto_techniques": ["twla", "cat_q", "ptqtp"],
            "receipts": [R_BYTES],
            "also": [R_TERNARY_COMP, R_FRACTIONAL, R_ONEBIT],
            "verdict": SCAR_TERNARY,
            "measured": {
                "active_bpw": s["ternary_bpw"],
                "complete_token_ns": s["ternary_ns"],
                "delta_ns": s["ternary_delta_ns"],
                "moved_toward_roof": s["ternary_moved_ns"],
                "teacher_argmax": s["ternary_teacher_argmax"],
                "student_argmax": s["ternary_student_argmax"],
                "argmax_agree": s["ternary_argmax_agree"],
            },
            "reading": (
                "ternary_5in8_g64 is slower than q2f despite fewer bytes, "
                "and the whole-model composition flipped the argmax "
                f"({s['ternary_student_argmax']} vs teacher "
                f"{s['ternary_teacher_argmax']})."
            ),
        },
        "sparse_residual": {
            "maps_onto_techniques": ["squeezellm", "vptq"],
            "receipts": [R_BYTES],
            "also": [R_BINARY_HEALING, R_C3],
            "verdict": SCAR_SPARSE,
            "measured": {
                "active_bpw": s["sparse_bpw"],
                "complete_token_ns": s["sparse_ns"],
                "csr_bytes": s["sparse_csr_bytes"],
                "binary_bytes": s["sparse_binary_bytes"],
                "nnz_frac": s["sparse_nnz_frac"],
            },
            "reading": (
                "binary + 2% CSR fused: indices cost more than the residual "
                "saves. active_bpw still below 2.25 and still slower than q2f."
            ),
        },
    }


# ---------------------------------------------------------------------------
# validation (the acceptance gate)
# ---------------------------------------------------------------------------


def technique_field_errors(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tid = (entry.get("technique_identity") or {}).get("id", "<unknown>")
    for key in REQUIRED_ENTRY_FIELDS:
        if key not in entry:
            errors.append(f"{tid}: missing field {key}")
    ident = entry.get("technique_identity") or {}
    if ident.get("literature_status") != LITERATURE_STATUS:
        errors.append(f"{tid}: literature_status must be {LITERATURE_STATUS}")
    if ident.get("not_authority") is not True:
        errors.append(f"{tid}: not_authority must be true")
    paper_doc = entry.get("source_paper") or {}
    for k in ("title", "approx_date"):
        if not paper_doc.get(k):
            errors.append(f"{tid}: source_paper.{k} required")
    v = entry.get("current_verdict") or {}
    if v.get("literature_is") != LITERATURE_STATUS:
        errors.append(f"{tid}: current_verdict.literature_is must be {LITERATURE_STATUS}")
    status = v.get("status")
    if status not in ALLOWED_VERDICT_STATUSES:
        errors.append(f"{tid}: illegal verdict status {status!r}")
    exp = v.get("cheapest_hawking_experiment")
    if status in {UNTESTED, RELATED_NEGATIVE} and not exp:
        errors.append(f"{tid}: {status} requires cheapest_hawking_experiment")
    mapping_doc = entry.get("hawking_experiment_mapping") or {}
    if not mapping_doc.get("campaign_mechanism_overlap"):
        errors.append(f"{tid}: hawking_experiment_mapping.campaign_mechanism_overlap required")
    return errors


def verdict_errors(entry: dict[str, Any]) -> list[str]:
    """Fail if a verdict is claimed without a cited Hawking receipt."""
    errors: list[str] = []
    tid = (entry.get("technique_identity") or {}).get("id", "<unknown>")
    v = entry.get("current_verdict") or {}
    status = v.get("status")
    receipts = v.get("hawking_receipts") or []
    if status in VERDICT_REQUIRES_RECEIPT:
        if not receipts:
            errors.append(
                f"{tid} claims verdict {status} without a cited Hawking receipt"
            )
        for path in receipts:
            if not is_hawking_receipt_path(path):
                errors.append(
                    f"{tid} citation {path!r} is not a Hawking receipt "
                    "(need receipts/**/*.json, not a paper URL)"
                )
            elif not citation_exists(path):
                errors.append(f"{tid} cited Hawking receipt does not exist: {path}")
    for path in receipts:
        if path and not is_hawking_receipt_path(path):
            errors.append(
                f"{tid} hawking_receipts contains a non-receipt path: {path!r}"
            )
    return errors


def registry_errors(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    techniques = doc.get("techniques") or []
    ids = [(t.get("technique_identity") or {}).get("id") for t in techniques]
    missing = [i for i in REQUIRED_TECHNIQUE_IDS if i not in ids]
    if missing:
        errors.append(f"missing required techniques: {missing}")
    dup = sorted({i for i in ids if ids.count(i) > 1 and i})
    if dup:
        errors.append(f"duplicate technique ids: {dup}")
    for t in techniques:
        errors.extend(technique_field_errors(t))
        errors.extend(verdict_errors(t))
    xref = doc.get("campaign_cross_references") or {}
    expected = {
        "shared_basis": SCAR_SHARED_BASIS,
        "binary": SCAR_BINARY,
        "low_rank_residual": SCAR_LOWRANK,
        "ternary": SCAR_TERNARY,
        "sparse_residual": SCAR_SPARSE,
    }
    for key, phrase in expected.items():
        block = xref.get(key) or {}
        if block.get("verdict") != phrase:
            errors.append(
                f"campaign_cross_references.{key}.verdict must be {phrase!r}"
            )
        recs = block.get("receipts") or []
        if not recs:
            errors.append(
                f"campaign_cross_references.{key} claims a verdict without a cited Hawking receipt"
            )
        for path in recs:
            if not is_hawking_receipt_path(path):
                errors.append(
                    f"campaign_cross_references.{key} non-receipt citation {path!r}"
                )
            elif not citation_exists(path):
                errors.append(
                    f"campaign_cross_references.{key} missing receipt {path}"
                )
    return errors


# ---------------------------------------------------------------------------
# build / write
# ---------------------------------------------------------------------------


def build() -> dict[str, Any]:
    t0 = time.time()
    seed = extract_campaign_seed()
    techniques = build_techniques(seed)
    xref = build_campaign_cross_references(seed)
    n_untested = sum(
        1 for t in techniques if t["current_verdict"]["status"] == UNTESTED
    )
    n_tested = sum(
        1
        for t in techniques
        if t["current_verdict"]["status"] in VERDICT_REQUIRES_RECEIPT
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "s026": ["§5", "§76", "§6", "§88", "§90", "§113"],
        "phase": "A",
        "family": "DOC-DIAGNOSIS",
        "hand_authored": False,
        "literature_is": LITERATURE_STATUS,
        "literature_is_not_authority": True,
        "did_not_load_a_model": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_modify_ascent_or_campaign": True,
        "this_is_the_registry_not_the_experiments": True,
        "one_line": (
            "Fifteen external techniques registered as HYPOTHESES; campaign "
            "scars (shared-basis / binary / low-rank / ternary / sparse) "
            "seed the verdicts that have Hawking receipts; the rest are UNTESTED "
            "with a cheapest CPU probe."
        ),
        "laws": [
            {
                "id": "S026_§5",
                "law": "Literature is hypothesis, not authority.",
            },
            {
                "id": "S026_§6",
                "law": "No blind implementation.",
            },
            {
                "id": "S026_§76",
                "law": "Register paper mechanisms before implementing them.",
            },
            {
                "id": "S026_§88",
                "law": "Provenance preserved.",
            },
            {
                "id": "S026_§90",
                "law": "Condemn the kernel, not the representation, until competence is screened.",
            },
            {
                "id": "verdict_rule",
                "law": (
                    "No CURRENT VERDICT other than UNTESTED without a cited "
                    "Hawking receipt (receipts/**/*.json that exists on disk or in git)."
                ),
            },
            {
                "id": "S026_§117",
                "law": (
                    "The information floor of a parameterization is not "
                    "necessarily the floor of the function; a moved floor "
                    "must be physically demonstrated."
                ),
            },
        ],
        "required_technique_ids": list(REQUIRED_TECHNIQUE_IDS),
        "techniques": techniques,
        "n_techniques": len(techniques),
        "n_untested": n_untested,
        "n_with_campaign_verdict": n_tested,
        "campaign_cross_references": xref,
        "docs": "docs/ultragoals/DOCTOR_TECHNIQUE_REGISTRY.md",
        "elapsed_s": round(time.time() - t0, 3),
    }
    errors = registry_errors(doc)
    if errors:
        raise SystemExit("registry invalid:\n  " + "\n  ".join(errors))
    return doc


def write_receipt(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    if doc is None:
        doc = build()
    write_json(RECEIPT, doc)
    return doc


def main() -> int:
    doc = write_receipt()
    print(f"schema {doc['schema']}")
    print(f"wrote  {RECEIPT.relative_to(REPO)}")
    print(f"n      {doc['n_techniques']} techniques, {doc['n_untested']} UNTESTED, "
          f"{doc['n_with_campaign_verdict']} with a campaign verdict")
    for t in doc["techniques"]:
        ident = t["technique_identity"]
        v = t["current_verdict"]
        print(f"  {ident['id']:<20} {v['status']:<20} {v.get('campaign_verdict') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
