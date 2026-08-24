#!/usr/bin/env python3
"""Noetic negative science: every representation idea this project already killed.

A new representation search is about to open (shared bases, tensor operators,
codebooks, low-rank, sparse correction, procedural generation, routing, stateful
compilation). This campaign has already REFUTED a great deal of that space.
A refutation nobody can find gets re-run — the most expensive failure available.

This tool is the archaeology, not a subsection of another receipt. It:

  * confirms or corrects the seed refutations against live files
  * sweeps receipts, ledgers, knowledge-plane jsonl, the foundry atlas,
    the ascent register, .haider/, worktrees and grok/* branches (read-only)
  * classifies each closure PROPERTY_OF_IDEA vs ARTIFACT_OF_METHOD
  * records a reopen condition and whether that condition holds TODAY
  * calls out reopen-already-true entries at the top (live opportunities
    being sat on)

Write: receipts/headless/NOETIC_NEGATIVE_SCIENCE.json
Run:   python3 tools/headless/noetic_negative_science.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

SCHEMA = "hawking.headless.noetic_negative_science.v1"
PROPERTY = "PROPERTY_OF_IDEA"
ARTIFACT = "ARTIFACT_OF_METHOD"

SWEEP_ROOTS = [
    "receipts",
    "tools/foundry",
    "reports",
    "workspace/campaign/records/ascension-sandbox/knowledge-plane",
    "workspace/ops/ascent-lanes",
    "workspace/campaign/evidence/systems/hawking",
    "workspace/campaign/evidence/models/glm52",
    "workspace/campaign/evidence/models/deepseek-v4",
    "hawking-experiments/superwave/g1",
    ".haider",
    "ramanujan/governance",
    "docs",
]


HAWKING_COPY = Path("/Users/scammermike/Downloads/hawking-copy")


def find_repo() -> Path:
    env = os.environ.get("HAWKING_REPO") or os.environ.get("HAWKING_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    if here.parent.name == "headless" and here.parent.parent.name == "tools":
        return here.parents[2]
    for p in [here.parent, *here.parents]:
        if (p / "Cargo.toml").exists() and (p / "tools" / "headless").is_dir() and (p / "receipts").is_dir():
            return p
    return Path.cwd().resolve()


REPO = find_repo()
_SIB_CACHE: Path | None | bool = False


def _git(args: list[str], cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd or REPO), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def sibling_science_root() -> Path | None:
    global _SIB_CACHE
    if _SIB_CACHE is not False:
        return _SIB_CACHE  # type: ignore[return-value]
    if not HAWKING_COPY.is_dir() or HAWKING_COPY.resolve() == REPO.resolve():
        _SIB_CACHE = None
        return None
    try:
        here = _git(["rev-parse", "HEAD"]).stdout.strip()
        there = _git(["rev-parse", "HEAD"], cwd=HAWKING_COPY).stdout.strip()
    except Exception:
        _SIB_CACHE = None
        return None
    _SIB_CACHE = HAWKING_COPY if here and here == there else None
    return _SIB_CACHE


def _e(path: str, number: Any = None, field: str | None = None, note: str | None = None) -> dict:
    return {"path": path, "number": number, "field": field, "note": note}


# field paths use '/' so keys that contain '.' (2.0856_BPW_..., first_router_below_0.5_layer)
# are not split. List-of-dicts with an 'id' key can be indexed by that id.

CATALOG: list[dict] = [
    {
        "id": "NNS-001",
        "seed": "sub_bit_synthetic_then_real",
        "seed_status": "CONFIRMED",
        "claim_refuted": (
            "That sub-bit is dead because six families scored negative on "
            "Gaussian / synthetic activations."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "The negatives were a property of the PROXY, not of sub-bit. "
            "Refitted on real teacher-capsule X the ranking inverted, the null "
            "moved 0.126 → 0.651, and activation-aware projection hit 0.755 "
            "cosine at 0.167 BPW on 12/12 GLM-5.2 experts. Raw-weight low-rank "
            "at 4× the rate still went 0/12. Standing law: never evaluate "
            "compression on synthetic activations. Sub-bit the IDEA is live; "
            "the Gaussian-proxy METHOD is dead."
        ),
        "scope": {
            "model": "glm-5.2 (routed experts); law transfers to q80/qwen38/dsv4f",
            "organ": "routed experts",
            "regime": "sub-bit, real captured activations vs Gaussian proxy",
            "codec": "activation-aware projection onto top-k activation covariance",
        },
        "evidence": [
            _e("workspace/campaign/evidence/systems/hawking/HAWKING_HEAVY_CONTINUATION_STATUS.json",
               0.755, "rebuild_glm/step_2_pilot/headline",
               "0.755 cosine at 0.167 BPW, 12/12 experts; ranking inverted off Gaussian"),
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               0.167, "entries/NS-009/what_was_measured/glm52"),
            _e("receipts/ascent-2026-08-16/FS_PER_WEIGHT_LAW.json",
               0.167, "precedent_for_sub_bit"),
            _e("receipts/ascent-2026-08-16/G016_PACKER_DESIGN.json",
               None, "hard_requirement",
               "synthetic/gaussian-proxy known-invalid; every prior sub-bit negative traced here"),
        ],
        "reopen_condition": (
            "Never as a promotion path on synthetic X. The IDEA reopens (and "
            "already has) on real teacher-forced / captured-BF16 X from the "
            "named source, with a stated null and a generation gate."
        ),
        "reopen_today": {
            "predicate": "real_x_vindication_on_disk",
            "if_true": "LIVE — sub-bit is vindicated on real X. Do not re-kill it with a Gaussian proxy.",
        },
    },
    {
        "id": "NNS-002",
        "seed": "complete_bpw_predicts_coherence",
        "seed_status": "CORRECTED",
        "seed_correction": (
            "The pair '3.5406 BPW INADEQUATE vs 3.6139 BPW COHERENT' is NOT in "
            "this tree (closest number: O005 mixed-q2q4 active_bpw=3.5412 "
            "DEGRADED; 3.6139 not found). The LAW the seed stated is confirmed: "
            "complete BPW does not predict coherence. Cheaper artifacts have "
            "both won and lost matched-density A/Bs. Cite the receipts below, "
            "not 3.5406/3.6139."
        ),
        "claim_refuted": (
            "That complete (or active) BPW predicts coherence: denser is worse, "
            "cheaper is worse, the number is the capability."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Multiply measured, not a metric bug. Qwen3.8 mixed-2p0 at 2.0856 "
            "BPW is native-INCOHERENT (0 fallbacks) while the Q4 oracle at "
            "4.2527 on the same binary is COHERENT. Independently, Q80 "
            "mixed-1p5 at 1.444 BPW GENERATED coherent text after the 0.8604 "
            "organ-cosine screen predicted failure. Odyssey O005 mixed-q2q4 at "
            "3.5412 active BPW is DEGRADED while q3-g128 at 3.2508 is "
            "CANDIDATE_PASS. GROUND_TRUTH F4: the 3.3448 artifact is ~10% "
            "SLOWER than uniform-q4 at 4.256 despite 21% fewer bytes. Density "
            "is not capability and is not velocity."
        ),
        "scope": {
            "model": "qwen3.8-27b, qwen3-80b, odyssey O005",
            "organ": "whole artifact (attention-dominated on Qwen3.8)",
            "regime": "matched-density / cross-density generate A/B",
            "codec": "mixed HGRAVB/R/S vs uniform-q4 vs q3-g128",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json",
               2.0856, "COHERENCE_FLOOR_BRACKETED",
               "key 2.0856_BPW_mixed-2p0-v1 = INCOHERENT native, 0 fallbacks; 4.2527 Q4 oracle COHERENT"),
            _e("receipts/ascent-2026-08-16/Q80_MIXED_GENERATE.json",
               1.4444456847927971, "artifact/complete_physical_bpw",
               "coherence_class=COHERENT at 1.444 BPW"),
            _e("receipts/odyssey-i/O005_GRAVITY_mixed-q2q4.json",
               3.5412, "active_bpw",
               "nearest number to the seed's 3.5406; verdict DEGRADED"),
            _e("receipts/odyssey-i/O005_GRAVITY_q3-g128.json",
               3.2508, "active_bpw",
               "CANDIDATE_PASS at lower BPW than the 3.5412 DEGRADED mix"),
            _e("receipts/ascent-2026-08-18/GROUND_TRUTH_TPS.json",
               3.3448, "runs/c-q3r1p22/complete_bpw",
               "denser 3.34 is ~10% slower than 4.256 uniform-q4"),
        ],
        "reopen_condition": (
            "Never as an unqualified 'BPW is the capability'. A new codec family "
            "may move the floor; it does not restore BPW as a predictor. Any "
            "claim that a cheaper pack is coherent (or a dearer pack is faster) "
            "must be a generate + complete-token measurement, not an arithmetic "
            "on BPW."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-003",
        "seed": "sub_1_5_whole_model_dead",
        "seed_status": "CORRECTED",
        "seed_correction": (
            "Sub-1.5 with CURRENT codecs is dead on Qwen3.8 and Q30, not "
            "'sub-1.5 as a density'. Attention sets the Qwen3.8 floor (74% of "
            "mixed-2p0 bytes at 4.250 BPW; MLP already 0.848). The first "
            "quality-intact cheap-to-reconstruct MLP rung is uniform-q3 at "
            "3.25 BPW (QWEN38_BPW_DESCENT coherence_floor), and the first "
            "gated-coherent whole artifact is flat q3 / 3.3448 complete, not "
            "'MLP at 3.25 is the first coherent low-BPW artifact' as a "
            "whole-model claim. Q80 mixed-1p5 at 1.444 DID generate coherent "
            "text — so sub-1.5 is not dead on Q80 experts. G006 (coherent "
            "under 1.5 on Qwen3.8) needs a new attention codec family."
        ),
        "claim_refuted": (
            "That a whole-model artifact at complete BPW ≤ 1.5, built from the "
            "existing Gravity families (HGRAVB01/R02/S01 + rice on attention), "
            "is coherent on Qwen3.8 or is a copyable Q30 template."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Native generate, 0 fallbacks, 0 dense-W: mixed-sub15 ~1.29 BPW is a "
            "degenerate 220/264 cycle; mixed-2p0 2.0856 is fifteen newlines + ')'. "
            "The Q4 oracle on the same binary is coherent. This is the artifact, "
            "not the runtime. Q30 static ≤1.5 SVD failed the same way (bits "
            "reachable, capability not). The idea 'crush experts/MLP and keep "
            "attention at Q4' cannot clear 1.5 because attention is the mass. "
            "A NEW attention family would be a new premise, not a retry."
        ),
        "scope": {
            "model": "qwen3.8-27b (whole model); qwen3-30b-a3b as template",
            "organ": "attention GEMVs + embed/lm_head dominate; MLP already 0.848 BPW",
            "regime": "complete BPW ≤ 1.5, current Gravity families, native generate",
            "codec": "HGRAVB01 binary_g128 / HGRAVR02 rice_q1 / HGRAVS01 r160 + rice on attention",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json",
               1.291, "projection_the_packer_reported/implied_bpw",
               "INCOHERENT degenerate cycle 220/264, 0 fallbacks"),
            _e("receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json",
               2.0856, "COHERENCE_FLOOR_BRACKETED"),
            _e("receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json",
               0.8480504639008466, "evidence/mlp_physical_bpw",
               "attention+embed+norms 4.250 BPW = 74% of artifact"),
            _e("receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json",
               3.25, "coherence_floor/quality_intact/physical_bpw",
               "uniform_q3_g64 is the cheap-to-reconstruct quality-intact MLP floor"),
            _e("receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json",
               0.99, "quality_bound/primary",
               "existing Gravity families fail 0.99 on attention; do not transfer the MLP bundle"),
            _e("receipts/q30-startup-latency/TOURNAMENT_READINESS_REPORT.md",
               0.8255, None,
               "Q30 sub-bit SVD all-layer mean 0.8255, layer-product ~3.6e-5 vs >=0.5 bar"),
        ],
        "reopen_condition": (
            "A new attention codec family, scored on real BF16 X, clearing "
            "mean-row output cosine ≥ 0.990 vs BF16 AND a multi-prompt native "
            "generate identity gate. The interval (2.0856, 4.2527) on Qwen3.8 "
            "with CURRENT codecs is untested — a coherent point there is a new "
            "measurement, not a retry of mixed-sub15 / mixed-2p0. Never copy "
            "the Q30 static ≤1.5 SVD as a template."
        ),
        "reopen_today": {"predicate": "never_for_current_families", "if_true": None},
    },
    {
        "id": "NNS-004",
        "seed": "shared_basis_across_experts",
        "seed_status": "CORRECTED",
        "seed_correction": (
            "The 0.004 figure is Q80 layer 10, 96 of 512 experts "
            "(QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json), NOT DSV4F. The "
            "ascent register already records this attribution error. Foundry F0 "
            "(gpt-oss-120b) independently measured mean pairwise cosine 1e-4; "
            "F1 (qwen3-235b) row-normalized off-diagonal 0.00166 vs 0.00168. "
            "Shared-basis is REFUTED on those parents. DSV4F pairwise cosine "
            "has never been measured in this tree."
        ),
        "claim_refuted": (
            "That routed experts share a basis / codebook / template, so a "
            "shared subspace, joint codebook, or same-index cross-layer tying "
            "compresses them together."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Experts on the measured parents are mutually orthogonal. Q80 L10 "
            "gate pairwise cosine mean 0.00414 (p95 0.00769); up mean ≈ 0; "
            "top-32 subspace overlap 0.020. F1 row-normalization does not "
            "rescue it (instrument carried a positive control that WOULD have "
            "detected norm-swamping). Best single shared template explains "
            "0.2513 of 4 experts' energy against an orthogonal null of 0.2500. "
            "Same-expert-index cross-layer tying is indistinguishable from a "
            "different-index control at 1e-7. There is no shared direction to "
            "exploit on these parents."
        ),
        "scope": {
            "model": "qwen3-80b (L10, 96 experts); gpt-oss-120b:F0; qwen3-235b-a22b:F1. NOT measured on dsv4f",
            "organ": "gate_proj / up_proj (and F1 all three)",
            "regime": "weight-space and row-normalized pairwise cosine; shared template energy",
            "codec": "shared codebook / joint subspace / expert templates+deltas / cross-layer tying",
        },
        "evidence": [
            _e("receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
               0.004142791032791138, "components/gate_proj/pairwise_cosine_mean"),
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               0.004, "attribution_corrections/0",
               "0.004 is Q80, not DSV4F"),
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.0001, "entries/inter_expert_redundancy/killed_by"),
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.2513, "entries/cross_expert_and_cross_layer_tying/killed_by",
               "best shared template 0.2513 vs orthogonal null 0.2500"),
        ],
        "reopen_condition": (
            "Never on Q80 / F0 / F1. On any new parent: measure pairwise "
            "cosine (raw AND row-normalized) on THAT parent's weights. Reopen "
            "only if mean ≳ 0.10 (foundry) / ≳ 0.05 (NS-010 DSV4F note). Do "
            "not transfer the 0.004 figure. Measuring DSV4F is a cheap check, "
            "not a retry of a Q80-refuted pack."
        ),
        "reopen_today": {
            "predicate": "dsv4f_pairwise_unmeasured",
            "if_true": "LIVE CHECK — DSV4F pairwise cosine has never been measured. Do not skip it by transferring Q80's 0.004.",
        },
    },
    {
        "id": "NNS-005",
        "seed": "cosine_scale_invariance",
        "seed_status": "CONFIRMED",
        "claim_refuted": (
            "That a fit-quality gate of observed/probed/worst_unit cosine can "
            "certify an artifact, including that 1.000000 means the weights "
            "were preserved."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "Cosine is scale-invariant by construction. Measured on real L0 "
            "gate_proj: Wh = 0.01*W scores observed/probed/worst_unit = "
            "1.000000 with relative weight error 0.9898; 100*W does the same "
            "at error 98.99. A candidate that preserves every direction and "
            "destroys every magnitude was HEALTHY on all three original axes. "
            "This did not prove the artifacts were good; it proved the metric "
            "was blind. A gain axis (min(r, 1/r) on per-row norm ratio) was "
            "added because that construction was exhibited."
        ),
        "scope": {
            "model": "any (exhibited on Qwen3.8 L0 gate_proj)",
            "organ": "any matrix scored by cosine-only doctor axes",
            "regime": "fit-quality / doctor gate, entire campaign that used cosine-only",
            "codec": "any (the metric, not the codec)",
        },
        "evidence": [
            _e("tools/gravity_doctor_gate.py",
               1.000000, None,
               "Wh=0.01*W scores 1.000000 on observed/probed/worst_unit; rel weight error 0.9898"),
        ],
        "reopen_condition": (
            "The cosine-only GATE stays dead. The IDEA (direction-preserving "
            "low-rate codes) reopens only under a scale-aware metric (gain "
            "axis, relative-L2, or generation). Re-read any campaign whose "
            "GO used cosine without a magnitude term."
        ),
        "reopen_today": {
            "predicate": "gain_axis_exists",
            "if_true": "LIVE — gravity_doctor_gate._gain exists. Cosine-only campaigns are untrustworthy and have not been systematically re-scored.",
        },
    },
    {
        "id": "NNS-006",
        "seed": "q30_gibberish_baseline",
        "seed_status": "CONFIRMED",
        "claim_refuted": (
            "That X captured from a degraded / quantized / gibberish baseline "
            "is close enough for codec ranking (the 'Q30 fits calibrated on "
            "gibberish' campaign)."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "A prior campaign captured X from a 0.7966-cosine gibberish "
            "baseline (Q30 binary, complete BPW 1.1304, emits ' compreh swiper "
            "swiper'). Every score ranked the wrong trajectory. Separately, "
            "the Q30 sub-bit SVD itself is a PROPERTY failure (NNS-003): "
            "perexpert64 from the correct+dense BF16 capture lifts L0 cosine "
            "only +0.015 over that gibberish baseline; all-layer mean 0.8255, "
            "layer-product ~3.6e-5 vs ≥0.5. Capture was never the wall for "
            "THAT representation. The METHOD error (calibrating on gibberish) "
            "and the IDEA failure (sub-bit SVD ceiling) must not be collapsed."
        ),
        "scope": {
            "model": "qwen3-30b-a3b (origin); law applies to q80/qwen38/dsv4f fits",
            "organ": "all expert organs whose X came from a degraded run",
            "regime": "activation capture used as teacher for codec ranking",
            "codec": "any fit against a non-source teacher",
        },
        "evidence": [
            _e("workspace/ops/ascent-lanes/_Q80_DENSITY_COMMON.md",
               0.7966, None,
               "prior campaign captured X from a 0.7966 gibberish baseline"),
            _e("workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
               0.7966, None,
               "baseline 0.7966 emits ' compreh swiper swiper'"),
            _e("receipts/q30-startup-latency/TOURNAMENT_READINESS_REPORT.md",
               0.015, None,
               "correct+dense BF16 perexpert64 lifts L0 cosine only +0.015 over gibberish"),
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               0.7966, "entries/NS-015/what_was_measured"),
        ],
        "reopen_condition": (
            "Never calibrate on a degraded baseline. Fit against the named "
            "source (Q80 BF16, Qwen3.8 BF16, DSV4F official mixed). A Q30 "
            "sub-bit SVD retry is NNS-003, not this entry."
        ),
        "reopen_today": {
            "predicate": "bf16_captures_named",
            "if_true": "LIVE for ranking — Q80 source-bf16-capture and Qwen3.8 activation-capture-v2 are the named teachers. Do not revive gibberish-X ranking.",
        },
    },
    {
        "id": "NNS-007",
        "seed": "undersampled_fits",
        "seed_status": "CONFIRMED",
        "claim_refuted": (
            "That a rank-r or full-dim codec scored on fewer captured rows "
            "than the fitted dimension is a score of the codec."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "rows < input_dim is underdetermined for a full-rank score; "
            "rows < rank is underdetermined for a rank-r score. A prior Q80 "
            "run had median 92 rows against 2048 dims and every score was "
            "garbage. The 25k-token capture is still starved: p10=34, p50=258 "
            "rows; 24326/24576 gate/up pairs have rows < 2048; 221 never-routed. "
            "rank = min(budget, n_fit_rows) silently starves the codec and the "
            "score is not the codec's score. Widening the corpus without "
            "per-expert retention made Q30 fits THINNER (p50 50 → 39)."
        ),
        "scope": {
            "model": "q80, dsv4f, q30",
            "organ": "routed expert gate/up/down",
            "regime": "fit/score with n_fit < rank or n_fit < dim",
            "codec": "any rank-r or full-dim",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               92, "entries/NS-014/what_was_measured/prior_q80_run",
               "median 92 rows against 2048 dims"),
            _e("receipts/ascent-2026-08-16/G031_FAMILY_REVIEW.json",
               92, None,
               "rank_deficient_capture: median 92 rows against 2048 dims"),
            _e("workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
               39, None,
               "30× larger corpus made p50 50 → 39; retention was per-layer not per-expert"),
        ],
        "reopen_condition": (
            "Re-score only when n_fit ≥ claimed rank (and for a full-dim "
            "claim, n_fit ≥ dim), with rank not clamped to n_fit. Corpus "
            "width is a legitimate lever only above a per-expert retention floor."
        ),
        "reopen_today": {"predicate": "q80_still_underdetermined", "if_true": None},
    },
    {
        "id": "NNS-008",
        "seed": "student_distillation_closed",
        "seed_status": "CORRECTED",
        "seed_correction": (
            "The 0.898 null is real (block_output constant-mean raw cosine on "
            "GLM) and is why raw activation cosine cannot certify a student. "
            "The CLOSED arc is independent per-layer functional students, on "
            "TWO parents: GLM (residual stream expansive 1.4–2.4×/layer, "
            "cascade skill 0.098 by L74) AND DeepSeek-V4-Flash (first router "
            "<0.5 at L4, first block-skill divergence at L8, final skill "
            "−168.6). The seed's 'layers 4-8 in all 40 layers' is the DSV4F "
            "cascade (40 sparse MoE layers), not GLM's 40. Route-aware "
            "jointly-trained students are untested and explicitly out of scope "
            "of the closure. PHASE_B_HYBRID later named distillation as the "
            "sole surviving MLP-byte avenue — that is a different student "
            "(match the MLP function, not compose independent layer students)."
        ),
        "claim_refuted": (
            "That independent per-layer functional students compose across a "
            "full MoE model, and that raw activation cosine ~0.86–0.90 means "
            "the student preserved function."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "On DSV4F the killer is structural, not student quality: L3 fits "
            "at skill 0.93 with perfect routing, then L4 router top-1 drops "
            "below 0.5. Any nonzero per-layer error compounds through a "
            "sensitive top-6-of-256 router. A perfect per-layer student is "
            "impossible (0.97 skill is 3% error). On GLM the residual stream "
            "is expansive at every tested magnitude (worst at the smallest "
            "perturbations); a stack of functional students diverges. Raw "
            "cosine 0.898 is the constant-mean null — it can mean nothing. "
            "The independent-per-layer paradigm is dead. A route-aware "
            "rollout-trained student is a new premise."
        ),
        "scope": {
            "model": "deepseek-v4-flash (40 sparse MoE layers); glm-5.2 (strata 3,38,74)",
            "organ": "routed MoE block + router",
            "regime": "independent per-layer functional students, in-sample fits, cascade",
            "codec": "glm52.functional.moe.v1 / per-layer student readout",
        },
        "evidence": [
            _e("workspace/campaign/evidence/models/deepseek-v4/DEEPSEEK_V4_FLASH_CASCADE_DECISION.json",
               4, "cascade/first_router_below_0.5_layer",
               "L4 router <0.5; L8 first block-skill divergence; final skill -168.6"),
            _e("workspace/campaign/evidence/models/glm52/GLM52_FUNCTIONAL_DECISION.json",
               0.0977, "closure/cascade_final_skill/L74",
               "late cascade skill 0.098 in four layers; residual expansive"),
            _e("workspace/campaign/evidence/systems/hawking/HAWKING_NULL_CORRECTED_METRIC_CONTRACT.json",
               0.898, "observed_constant_mean_raw_cosine/block_output"),
            _e("workspace/ops/ascent-lanes/_Q80_DENSITY_COMMON.md",
               0.898, None,
               "functional-student arc CLOSED after divergence by layer 4-8 in all 40 layers"),
        ],
        "reopen_condition": (
            "A jointly trained, route-aware / rollout objective that preserves "
            "the NEXT layer's routing, judged end-to-end (never per-layer), "
            "against a stated null (not raw cosine 0.898), plus generation. "
            "A late-layer stabilizer does not help: DSV4F divergence begins at L4."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-009",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That organ cosine ~0.86–0.90, or the D23 0.8604 residual-identity "
            "bar, is a capability certificate / hard GO-NO-GO."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "The 0.8604 bar sits BELOW the 0.898 constant-mean null. up_proj "
            "rice_q1 at 0.864–0.865 is below null. Then the same 1.444 mixed "
            "artifact generated coherent text (down_proj holdout cosine 0.7684). "
            "The screen predicted failure where generation worked. Conversely "
            "mixed-2p0 mean_component_cosine 0.907 is native-INCOHERENT. Cosine "
            "is a screen. Generation is the gate."
        ),
        "scope": {
            "model": "q80 (bar origin); glm52 (null origin); qwen38 (counterexample)",
            "organ": "expert organs; not attention (attention uses 0.99)",
            "regime": "organ-cosine GO/NO-GO",
            "codec": "any",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               0.8604, "entries/NS-016"),
            _e("receipts/ascent-2026-08-16/Q80_MIXED_GENERATE.json",
               1.4444456847927971, "artifact/complete_physical_bpw",
               "coherence_class=COHERENT"),
            _e("receipts/ascent-2026-08-16/CROSS_ADVERSARIAL_FINDINGS.json",
               0.8585935762823004, "findings_ranked/P1-MIXED-CLEARS-BAR-FALSE",
               "cited mixed receipt sets clears_bar true on a FAIL organ"),
        ],
        "reopen_condition": (
            "Never as a hard GO/NO-GO. A generation-calibrated bar is a new "
            "premise. Do not reopen the 588-recipe organ-cosine grid against 0.8604."
        ),
        "reopen_today": {
            "predicate": "mixed_1p5_generated",
            "if_true": "LIVE — mixed-1p5 already generated coherent text below the 0.8604 bar. Recipes refused by that bar are being sat on.",
        },
    },
    {
        "id": "NNS-010",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That matching the Gaussian rate-distortion function sqrt(2^-2R) "
            "proves a post-hoc codec is at the true Shannon floor."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "Among equal-variance sources the Gaussian MAXIMIZES differential "
            "entropy, so its D(R) is an UPPER bound on the distortion a "
            "same-variance source requires. Matching it proves the codec is as "
            "good as if the weights were Gaussian; it proves nothing about the "
            "true floor. The correct floor uses measured h. Mid/late layers "
            "ARE essentially Gaussian (0.012–0.018 bits non-Gaussian) so Lane A "
            "is nearly exhausted there. Layer 0 is not: down_proj 3.014 bits "
            "non-Gaussian, 1.328 decades off its own Shannon bound."
        ),
        "scope": {
            "model": "qwen3-235b-a22b:F1",
            "organ": "gate/down; layer 0 is a different source",
            "regime": "post-hoc fixed-weight codec at a fixed sub-bit index rate",
            "codec": "incumbent PQ/VQ family",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               1.328, "entries/post_hoc_coding_of_frozen_weights/named_exception",
               "L0 down 3.014 bits non-Gaussian, 1.328 decades off Shannon"),
        ],
        "reopen_condition": (
            "Any cell measuring ≥ 0.5 decades of gap to its OWN Shannon lower "
            "bound. Layer 0 already does. Do not transfer 'Lane A is closed' to it."
        ),
        "reopen_today": {
            "predicate": "layer0_shannon_gap",
            "if_true": "LIVE — L0 is a different source. Kronecker and column-scale already beat the incumbent there and were skipped under a tying-method exemption that does not apply.",
        },
    },
    {
        "id": "NNS-011",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That the 5.9× per-byte reconstruction penalty measured on Q80 "
            "mixed rice/low-rank is a property of those codecs, and that it "
            "transfers to Qwen3.8."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "The 5.9× (2.57 vs 15.2 GB/s) was first measured against Q80's "
            "serial 1-thread-per-row extract, then transferred to Qwen3.8 "
            "without re-measurement. Both steps were wrong. At production "
            "tpr64 on Qwen3.8, 33 codecs on real activations land 15,124–15,541 ns "
            "against an f32 control of 15,125 ns; 32/33 recon-excess = 0. "
            "Same codecs at tg256 ~26,500 ns — the penalty is LAUNCH GEOMETRY, "
            "not the codec. Q80 in-register tiles took gpu_matvec 867.0 → 36.6 ms "
            "without changing a codec. Codec choice on Qwen3.8 at tpr64 is "
            "quality-constrained, not recon-time-constrained."
        ),
        "scope": {
            "model": "qwen3.8 at tpr64; q80 after in-register tiles",
            "organ": "MLP GEMV",
            "regime": "decode, named launch geometry",
            "codec": "q4/q3/q2/binary/ternary/hadamard/rice — reconstruction cost",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json",
               15125, "evidence/f32_control_tpr64_ns/gate",
               "33 codecs 15124-15541 ns vs 15125; cosine 1.000000 on 32/33"),
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               5.9, "entries/NS-006/what_was_measured/slowdown_per_byte_x",
               "historical 5.9×; superseded as a transfer"),
        ],
        "reopen_condition": (
            "Do not cite 5.9×. Reopen a recon-cost claim only on a named launch "
            "geometry with GPU timestamps, same-vehicle, real X. Cheap codecs "
            "previously rejected for recon cost are quality-eligible at tpr64."
        ),
        "reopen_today": {
            "predicate": "reconstruction_is_free_at_tpr64",
            "if_true": "LIVE — tpr64 reconstruction is free. Codecs killed for the 5.9× penalty are eligible again on quality.",
        },
    },
    {
        "id": "NNS-012",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That the G3 shared-operator headline 'beats q3 held-out 0.371 vs "
            "0.401 at 24% of the bytes' is a measured successor."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "Three defects each swing more than the 0.0134 margin, all in the "
            "operator's favour: (D3a) 80/20 positional split leaks; (D5) "
            "operator error is one pooled rel-L2, q3 is a per-layer mean; "
            "(B0) trained on post_input_norm, real MLP input is post_attn_norm. "
            "Honest retest: operator 4.02 vs q3 0.337. After the method was "
            "fixed, the IDEA (narrow shared SwiGLU) failed at Doctor level "
            "(NNS-013). The leaked number is not a reopen."
        ),
        "scope": {
            "model": "qwen3.8-27b dense MLP",
            "organ": "all 64 MLPs as one shared operator + FiLM",
            "regime": "Phase B / S027 §11, degenerate prose corpus then honest retest",
            "codec": "shared SwiGLU m=4096/6144 + per-layer FiLM",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-18/G3_SHARED_OPERATOR_BREAKTHROUGH.json",
               0.3712, "shared_operator/held_out_err",
               "_REFUTED_2026-08-18: headline does not survive the audit"),
            _e("receipts/ascent-2026-08-18/METHODOLOGY_AUDIT.json",
               0.0134, "verdict"),
            _e("receipts/ascent-2026-08-18/G3_HONEST_RETEST.json",
               4.021837053820491, "operator/cross_family_held",
               "honest operator 4.02 vs q3 0.337"),
            _e("receipts/ascent-2026-08-18/CORRECTION_MLP_INPUT_TENSOR.json",
               None, "obligation",
               "wrong input tensor; same class as Q30-calibrated-on-gibberish"),
        ],
        "reopen_condition": (
            "The leaked 'beats q3' number stays dead. A full-width structured "
            "nonlinear (Monarch/butterfly, not a narrow bottleneck) is G8 and "
            "was listed still_untested — a new family, not a retry of m=6144."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-013",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That a narrow shared / grouped SwiGLU operator (m < 17408) can "
            "replace Qwen3.8's MLP at q3 quality and materially fewer active "
            "bytes, including after honest methodology."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "After NNS-012's method bugs were removed: honest single-op "
            "plateaus rel 0.59 (1.75× q3) in-family and 0.95 cross-family; "
            "matching q3's 0.337 requires m ~ 10000–12000, at which active "
            "bytes approach q3's own — no Pareto win. Assembled-Doctor: 0/4 "
            "GIBBERISH. Grouped K=4 identical to single. The width bottleneck "
            "of a 17408-wide MLP is fundamental. Density via this operator "
            "family is dead at the contract (Doctor) level, not just a proxy."
        ),
        "scope": {
            "model": "qwen3.8-27b",
            "organ": "MLP (67–68% of GEMV bytes)",
            "regime": "honest cross-family, correct post_attn_norm, assembled-Doctor",
            "codec": "shared/grouped SwiGLU m=6144, K=1 and K=4",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-18/DENSITY_LEVER_HONEST.json",
               0.6483, "ASSEMBLED_DOCTOR_single_op/operator_function_err",
               "0/4 GIBBERISH; q3=0.343"),
            _e("receipts/ascent-2026-08-18/STRATEGIC_FINDING_100TPS.json",
               None, "assembled_doctor_DEFINITIVE"),
        ],
        "reopen_condition": (
            "A full-width structured nonlinear (G8 Monarch/butterfly) or a "
            "distilled operator trained to match F=down(silu(gate)*up) at q3 "
            "quality, held-out across families, with Doctor holding. Narrow "
            "m<17408 sharing is not that."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-014",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That activation-aware functional low-rank (output-PCA / reduced "
            "rank on real post-SwiGLU X) beats q3 at matched bytes held-out."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Function-space rank is real: 99% output energy at 56.2% of ranks "
            "vs 92.5% in weight space (3.4×). It does not translate. At matched "
            "bytes (rank-803 bf16 ~ q3 36.2 MB) the fit-set WIN (0.2017 < 0.2220) "
            "collapses on held-out (L31 0.3876 vs q3 0.2216; L15 0.3120 vs "
            "0.2219). A rank-803 operator on ~2048 activation samples memorizes "
            "them. G034 at the same matched 3.25 b/elem: low-rank 2–3× q3 "
            "function error."
        ),
        "scope": {
            "model": "qwen3.8-27b",
            "organ": "down_proj L31/L15 (and G034 gate/down L0/L31/L63)",
            "regime": "matched-byte, real post_swiglu X, TRAIN/TEST split",
            "codec": "functional rank-803 f16 / G034 low-rank at 3.25 b/elem",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-18/PHASE_B_FUNCTIONAL_LOWRANK.json",
               0.3876, "decisive_matched_bytes/held_out_L31/functional_rank803"),
            _e("receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json",
               0.6038066744804382, "per_tensor/0/lowrank",
               "L0 gate lowrank 0.604 vs flat_q3 0.193 at matched 3.25 b/elem"),
        ],
        "reopen_condition": (
            "A hybrid (low-rank prefix + exact/q3 residual) or a distilled "
            "operator showing a GENERALIZING matched-byte win over q3 across "
            "layers, with Doctor holding. Pure functional low-rank on this "
            "sample size is not that. (The hybrid was then tested: NNS-015.)"
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-015",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That low-rank + activation-aware correction of the residual beats "
            "q3's density-quality point (fewer active bytes at q3 quality, held-out)."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "q3+correction rank64 GENERALIZES and beats q3 quality (0.174 vs "
            "0.222) but ADDS bytes (107% of q3) — a Matryoshka/quality lever, "
            "not a speed lever. q2+correction cannot recover q3 quality within "
            "a reasonable byte budget (rank256 err 0.396 at 101% of q3 bytes). "
            "q3 is Pareto-optimal on this MLP density-quality frontier. "
            "PHASE_B_HYBRID named distillation as the sole surviving avenue "
            "to fewer active bytes."
        ),
        "scope": {
            "model": "qwen3.8-27b",
            "organ": "down_proj L31, real post_swiglu X",
            "regime": "base(qN absmax g64) + RRR correction of residual, held-out",
            "codec": "hybrid low-rank + correction",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-18/PHASE_B_HYBRID_REFUTED.json",
               0.1743, "results/q3_plus_correction/rank64/err",
               "beats q3 quality, 107% of q3 bytes; q2+corr cannot recover"),
        ],
        "reopen_condition": (
            "A distilled/generated operator, trained to match the MLP function, "
            "achieves q3 quality at materially fewer active bytes held-out "
            "across layers, with Doctor holding. That experiment has not been run."
        ),
        "reopen_today": {
            "predicate": "distill_operator_unrun",
            "if_true": "LIVE AVENUE — PHASE_B_HYBRID left distillation as the sole surviving MLP-byte path and it has not been run.",
        },
    },
    {
        "id": "NNS-016",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That weight-space low-rank / TT / Kronecker factorisation of a "
            "single expert tensor is a density win at the coherent point "
            "(layers ≥ 1)."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Qwen3.8 down/gate near-full-rank: 99% energy needs 92–95% of ranks; "
            "low-rank costs MORE params than full. Function-space matched 3.25 "
            "b/elem is 2.93× Q3 error. F1 Kronecker at depth: Van Loan spectrum "
            "nearly flat, top component 0.27% of gate energy; rank the 2.5 bpw "
            "budget buys captures 27% energy, rel_error 0.853 vs incumbent 0.239. "
            "THE EXCEPTION: L0 gate Kronecker 0.0301 vs incumbent 0.2252 at a "
            "CHEAPER complete rate (2.487 vs 2.501). The lane skipped L0 under "
            "a tying-method exemption that does not apply to single-tensor "
            "factorisation."
        ),
        "scope": {
            "model": "qwen3.8-27b (weight-space SVD); qwen3-235b-a22b:F1 (Kronecker)",
            "organ": "gate/down; L0 is the exception",
            "regime": "coherent-point matched bits; S64 rungs for Kronecker",
            "codec": "low-rank / TT / Kronecker A⊗B",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-18/PHASE_A_EXHAUSTION.json",
               None, "representation_front/low_rank_TT_kronecker"),
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.0301, "entries/kronecker_factorisation/the_exception",
               "L0 gate 0.0301 vs 0.2252 at cheaper BPW; DEAD for L>=1"),
        ],
        "reopen_condition": (
            "L>=1: never at the coherent point. L0: already beating the "
            "incumbent — build the organ-and-layer-specific codec. On a new "
            "parent, check the Van Loan spectrum before assuming depth behaviour."
        ),
        "reopen_today": {
            "predicate": "layer0_kronecker_live",
            "if_true": "LIVE — L0 Kronecker already beats the incumbent and was skipped.",
        },
    },
    {
        "id": "NNS-017",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That PQ/VQ coding of the RAW frozen weights is a route to one bit "
            "or below that preserves capability."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "F1 real forward, 6 prompts, 94 layers, healthy parent (ppl 1.61–39.33): "
            "A1_1p0 at complete 1.0075 BPW collapsed 6/6 (symKL 7.6–10.9, argmax "
            "0.0); R2_subhalf at 0.4930 collapsed 6/6. Organ inversion was "
            "applied and still collapsed. This kills THAT FAMILY on frozen "
            "weights, not the sub-bit program. Methods that change the source "
            "(QAT, distillation, compressibility training, structured pruning, "
            "learned sharing) are not bound by the original weights' "
            "rate-distortion limit and remain untested."
        ),
        "scope": {
            "model": "qwen3-235b:F1 (and F0 gpt-oss-120b uniform/treated)",
            "organ": "whole model; dominant_failure_organ = gate",
            "regime": "raw-weight PQ/VQ at ~1 bit and below, real forward",
            "codec": "A1_1p0 / R2_subhalf / uniform sub-bit",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               1.0075, "entries/raw_weight_pq_vq_at_one_bit/killed_by"),
            _e("tools/foundry/evidence/f1_qwen3_235b.json",
               0.0, None,
               "12/12 packed rows, argmax agreement exactly 0.0, symKL 7.61 to 13.47"),
        ],
        "reopen_condition": (
            "Never on raw frozen weights. A method that CHANGES the source is "
            "not this lever and is not blocked by this entry."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-018",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That a single codec family (all-binary, or all low-rank) wins on "
            "every Q80 expert organ inside the 1.5 budget."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "gate_proj: binary_group clears the in-budget screen on busy organs "
            "(L10E453 cosine 0.893). up_proj: binary_g fails the same organ "
            "(0.828); binary + rice_q1 residual is the in-budget family. "
            "down_proj: binary_g fails on post-SwiGLU X (L1E265 0.826); low-rank "
            "is the intended family and INVERTS the ranking. down_proj must be "
            "fit on post-SwiGLU X, never the layer hidden. A single-family pack "
            "either misses the bar or blows the 1.5 budget."
        ),
        "scope": {
            "model": "qwen3-80b",
            "organ": "routed gate / up / down separately",
            "regime": "in-budget screen on real X (post-SwiGLU for down)",
            "codec": "binary_group vs rice residual vs hgravs01",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               None, "entries/NS-012"),
            _e("receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json",
               1.22957, "mixed_expert_bpw",
               "the mixed recipe exists because single-family is insufficient"),
        ],
        "reopen_condition": (
            "Never as a Q80 default. Reopen only if a new organ family, scored "
            "on real post-SwiGLU X for down and real hidden for gate/up, beats "
            "the mixed recipe on all three organs inside the budget."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-019",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That existing Gravity expert families (HGRAVB01 / HGRAVR02 / "
            "HGRAVS01, or the Q80 mixed gate/up/down bundle) transfer to "
            "attention at Q4-equivalent quality."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Attention is high-sensitivity (bar 0.99, not 0.8604). Binary max "
            "cosine 0.946 typical 0.75–0.88. Rice residual max 0.958 typical "
            "0.82–0.91. SVD r=512 typical 0.83–0.89; 0 clears of 0.99 at ranks "
            "that beat Q4 BPW. The sign-scale recipe exploits SwiGLU structure "
            "attention does not have; r160/post-SwiGLU/in-dim 512 is a down_proj "
            "recipe. Attention in-dim is 2048 or 5120 and is not low-rank under "
            "the same fit. This is why Qwen3.8 sub-1.5 failed: MLP is already "
            "0.848, attention is 74% of the artifact at 4.25 BPW."
        ),
        "scope": {
            "model": "qwen3.8-27b and qwen3-80b attention GEMVs",
            "organ": "Q/K/V/O, DeltaNet in/out, lm_head extra (top-1 ≠ cosine)",
            "regime": "real BF16-source hiddens, Q4-equivalent 0.99 bar",
            "codec": "HGRAVB01/R02/S01/H01/T01",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json",
               0.946, "codec_applicability/HGRAVB01_binary/quality",
               "max cosine 0.946; typical 0.75-0.88; FAILS 0.99 everywhere"),
            _e("receipts/ascent-2026-08-16/ATTENTION_FUNCTION_SPACE.json",
               None, "finding_2_attention_is_more_sensitive",
               "q3 costs attention 0.228-0.301 vs MLP 0.198-0.240"),
        ],
        "reopen_condition": (
            "A NEW attention codec family, scored on real BF16 X, clearing "
            "mean-row output cosine ≥ 0.990 vs BF16 AND a multi-prompt generate "
            "identity gate. lm_head additionally needs top-1 agreement; cosine "
            "0.99 is not token identity."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-020",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That storage / complete_physical BPW is the BPW decode moves "
            "(active BPW), and that crushing routed experts is therefore the "
            "velocity lever."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Category error with a physical cause. At batch=1 Q80 reads 10 of "
            "512 experts. mixed-sub655 storage 0.6462 vs active 2.518 (~4×); "
            "mixed-1p5 1.4444 vs 4.98. Per-token bytes: attention 73%, "
            "attention+lm_head 86–88%, routed experts 9%. Crushing unused "
            "experts to zero removes 9% of traffic. Attention is the mass "
            "whose compression changes the token. Storage BPW remains a valid "
            "disk figure."
        ),
        "scope": {
            "model": "qwen3-80b batch=1 decode",
            "organ": "attention + lm_head vs routed experts",
            "regime": "active vs storage accounting",
            "codec": "any mixed pack that leaves attention at Q4/Q8",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               2.518, "entries/NS-001/what_was_measured/active_bpw_mixed_sub655"),
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               73.0, "entries/NS-005/what_was_measured/attention_pct"),
        ],
        "reopen_condition": (
            "Never as a substitute for active BPW, and never as a token-time "
            "lever on Q80 batch=1. Reopen expert compression as velocity only "
            "if routing changes so a much larger expert fraction is actually "
            "read per token."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-021",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That lower storage BPW is faster per token (density is velocity) "
            "on mixed codecs whose reconstruction is not in-register."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "On the mixed G020 vehicle before the in-register fix: Q4 15.2 GB/s "
            "vs mixed 2.57 GB/s (5.9× slower per byte); token 225 vs 1171 ms. "
            "The 5.9× NUMBER was later shown to be a serial-extract + transfer "
            "artifact (NNS-011) at tpr64; the LAW 'do not assume lower BPW is "
            "faster' survives because reconstruction cost is genome-dependent. "
            "Always name the vehicle and the launch geometry."
        ),
        "scope": {
            "model": "q80 mixed-1p5 vehicle (pre-recon-fix genome)",
            "organ": "packed expert matvec",
            "regime": "decode complete-token, same-vehicle GB/s",
            "codec": "mixed binary+rice+hgravs vs uniform-q4",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               1171, "entries/NS-006/what_was_measured/mixed_token_ms"),
        ],
        "reopen_condition": (
            "Never as an unqualified rule. A density-for-speed claim only after "
            "reconstruction is in-register or fused into consumption and a "
            "same-vehicle GB/s is remeasured at or above the Q4 mark on that genome."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-022",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That entropy coding of Lloyd-optimal PQ indices yields a 10–25% "
            "byte win, or that rANS on Qwen3.8 q3 symbols is an active-byte/TPS lever."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Lloyd-optimal indices are near-uniform by construction; measured "
            "gain 0.0–0.7%, inside the noise of the byte plan. Qwen3.8: 0.696 "
            "bits/symbol of exploitable redundancy EXISTS as a lossless recode "
            "of STORED BPW, but rANS refuted on-disk (all r packs 11245158443 "
            "bytes). No native register-decodable path; cuts stored not "
            "active-bytes/token. zlib/lzma on q3 symbols come in WORSE than "
            "order-0 entropy."
        ),
        "scope": {
            "model": "gpt-oss-120b:F0 (PQ indices); qwen3.8 (q3 symbols / rANS catalog)",
            "organ": "codebook indices / q3 body",
            "regime": "lossless recode of already-quantized symbols",
            "codec": "entropy / rANS overscale",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.7, "entries/entropy_coded_pq_indices/killed_by"),
            _e("receipts/ascent-2026-08-18/GROUND_TRUTH_TPS.json",
               11245158443, "findings/F1_r_ladder_size_invariant",
               "rANS overscale realized ZERO byte reduction; 3.3448 is the on-disk floor"),
            _e("receipts/ascent-2026-08-16/G024_RATE_DISTORTION_BOUND.json",
               1.918189745254591, "slb_bits_per_elem_at_coherent_distortion",
               "q3 spends 3.25 against 1.918 required; compressors worse than order-0"),
        ],
        "reopen_condition": (
            "A future parent uses NON-Lloyd (biased or stratified) codebooks "
            "whose measured index entropy is ≤ 0.9 of uniform. rANS as a TPS "
            "lever additionally needs a native register-decodable path that "
            "changes active bytes/token."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-023",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That omitted MoE experts can be reconstructed from a learned "
            "combination of surviving experts, or that inter-expert redundancy "
            "lets you delta-code / cluster-mean-subtract."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Same wall as orthogonality. Best single surviving expert held-out "
            "rel error 0.885/0.993/0.995 at L0/L3/L7; 4-survivor least-squares "
            "merge only 0.863/0.988/0.995. ~1.0 means no better than predicting "
            "zero. Adversary found in-sample contamination in the merge's "
            "FAVOUR, so the true wall is at least this hard. F0 mean pairwise "
            "cosine 1e-4: there is no shared component to subtract."
        ),
        "scope": {
            "model": "qwen3-235b-a22b:F1 (merge); gpt-oss-120b:F0 (redundancy)",
            "organ": "routed experts",
            "regime": "held-out routed tokens at the keep fraction",
            "codec": "expert merge / shared low-rank / cluster-mean delta",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.885, "entries/expert_merging_omitted_from_survivors/killed_by"),
        ],
        "reopen_condition": (
            "A parent measures best-single-survivor reconstruction error ≤ 0.5 "
            "on held-out routed tokens at its own keep fraction, AND mean "
            "pairwise expert cosine ≥ 0.10."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-024",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That procedural / generated parameters (shared templates, "
            "same-index cross-layer tying, generated expert bases) buy bits "
            "on mutually orthogonal experts."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Best single shared template explains 0.2513 of 4 experts' energy "
            "against an orthogonal null of exactly 0.2500. Same-expert-index "
            "cross-layer tying is indistinguishable from a different-index "
            "control at 1e-7. The metric carried a positive control proving it "
            "WOULD detect norm-swamping. Generated-params on these parents "
            "have nothing to generate from."
        ),
        "scope": {
            "model": "qwen3-235b-a22b:F1",
            "organ": "expert tensors",
            "regime": "Lane F generated params, row-normalized",
            "codec": "shared template + delta / generated basis",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.2513, "entries/cross_expert_and_cross_layer_tying/killed_by"),
        ],
        "reopen_condition": (
            "A parent measures row-normalized mean off-diagonal expert cosine "
            "≥ 0.10. Until then procedural generation of expert bases is "
            "generating near-orthogonal noise."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-025",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That selection-policy search over which sub-1.2-bpw organs to "
            "replace can recover coherence, or that beating a constant-mean "
            "null is the replacement gate."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "48-layer composition needs per-organ output cosine ~0.986 to "
            "retain half the signal (c^48 ≥ 0.5). Measured p50 0.6854 (SVD) / "
            "0.80 (baseline binary) are two orders of magnitude below that, "
            "and the residual stream is EXPANSIVE (1.4–2.4×/layer) so errors "
            "amplify. Every artifact at or below 1.13 bpw is gibberish. "
            "Scoring against a constant-mean null instead of the incumbent: "
            "frac_beats_null_selected reached 1.0 while mean component cosine "
            "FELL 0.7966 → 0.6113. Beating a constant is nearly free."
        ),
        "scope": {
            "model": "qwen3-30b-a3b, 48 layers",
            "organ": "expert gate/up/down",
            "regime": "sub-1.2 bpw organ replacement / HGRAVS01 vs HQ30G1B1",
            "codec": "activation-weighted SVD vs packed binary",
        },
        "evidence": [
            _e("workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
               0.6854, None,
               "output_cosine p50 0.6854; every artifact ≤1.13 bpw gibberish"),
            _e("workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
               0.6113, None,
               "frac_beats_null=1.0 while cosine fell 0.7966→0.6113"),
        ],
        "reopen_condition": (
            "A representation family that reaches ~0.99 per-organ output "
            "cosine; then selection policy matters again. Any replacement gate "
            "must score against the incumbent on the same rows with the same "
            "holdout split, never against a constant."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-026",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That low-rank factorization is cheaper than dense 1-bit at "
            "sub-1.5 bpw on a [768, 2048] expert organ."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Rank 256 / 3 bits = (768+2048)*256*3 bits = 1.38 effective bpw "
            "versus ~1.13 for dense 1-bit, at output cosine 0.686 versus "
            "baseline 0.7966. The factor matrices of a rank-256 approximation "
            "of a 1.57M-parameter organ are not small relative to a 1-bit "
            "encoding of the whole organ. Low-rank pays more and delivers less "
            "in this regime."
        ),
        "scope": {
            "model": "qwen3-30b-a3b",
            "organ": "[768, 2048] expert organ, BUDGET_POINTS max rank 256 at 3 bits",
            "regime": "sub-1.5 complete, HGRAVS01 vs HQ30G1B1",
            "codec": "activation-weighted SVD rank-256",
        },
        "evidence": [
            _e("workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
               1.38, None,
               "rank256/3bit = 1.38 bpw vs ~1.13 dense 1-bit, cosine 0.686 vs 0.7966"),
        ],
        "reopen_condition": (
            "Organs with genuinely low intrinsic rank, or a rank low enough "
            "(≤ 64) that the factors are cheap, provided fidelity still reaches "
            "~0.99. Qwen3.8 down 5120×17408 at rank 160 is 3.1% of rows = 0.13 "
            "BPW — that geometry is a different (and still quality-open) premise."
        ),
        "reopen_today": {"predicate": "never_on_q30_organ", "if_true": None},
    },
    {
        "id": "NNS-027",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That extrapolating a 4-layer teacher-forced residual probe across "
            "48 layers is a GO, or that per-organ gating without expert-atomic "
            "enforcement produces a runnable hybrid."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "Four-layer probe: mixed rel-L2 geo-grows 1.277×/layer was projected "
            "to 16211 at layer 48 from span [0,4] whose ratios are 1.733, 1.229, "
            "0.978 (one of which shrinks). The probe cannot beat its shuffled-"
            "weight null; 395/2048 organs were rank-clamped (NNS-007). "
            "Per-organ gating without expert-atomic enforcement: the runtime's "
            "execution unit is the expert TRIPLE; 841 of 6144 experts got a "
            "partial HGRAVS triple and the runtime refused closed at L0 E55 "
            "before emitting any token."
        ),
        "scope": {
            "model": "q80 (4-layer probe); q30 (expert-atomic)",
            "organ": "routed expert triples",
            "regime": "coherence GO from a shallow probe; hybrid per-organ auction",
            "codec": "mixed / HGRAVS partial triples",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               1.277, "entries/NS-017/what_was_expected"),
            _e("workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
               841, None,
               "841/6144 experts partial HGRAVS triple; runtime refused at L0 E55"),
        ],
        "reopen_condition": (
            "A full-depth or tiled residual probe that separates from a stated "
            "null AND a generation run. Hybrid per-organ mixing requires a "
            "runtime that does not fragment the fused expert wave."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-028",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That post-hoc scalar gain on a PQ artifact, row-norm-stratified "
            "codebooks (the 94% single-codeword premise), ternary vs VQ at "
            "matched rate, or 88-token routing-frequency allocation are live levers."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Scalar gain: optimal gain pinned at exactly 1.0; k-means "
            "reconstruction is a conditional mean so the residual is orthogonal "
            "to the reconstruction, and cosine is gain-invariant — algebraically "
            "pinned. Row-norm stratification: the 94% single-codeword figure "
            "belonged to a k=32 geometry and does not transfer to deployed R2 "
            "(d16 k1024) where single-codeword share is 0.0267 mean. Ternary "
            "loses to VQ at every matched rate tested. 88-token calibration: "
            "median routing split only 63.6% stable, 26.1% of cells never "
            "route; the lever is alive at ≥1000 tokens, the 88-token "
            "calibration is dead."
        ),
        "scope": {
            "model": "gpt-oss-120b:F0 (gain, ternary, 88-token); qwen3-235b:F1 (row-norm)",
            "organ": "PQ/VQ coded experts",
            "regime": "post-hoc / stratified / matched-rate / short-calib",
            "codec": "PQ + scalar / R5_rownorm_strat / ternary / routing-frequency alloc",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               1.0, "entries/posthoc_scalar_gain/killed_by"),
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               0.0267, "entries/row_norm_stratification_premise/killed_by"),
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               88, "entries/calibration_88_tokens"),
        ],
        "reopen_condition": (
            "Gain: a NON-conditional-mean quantizer whose residual is measurably "
            "non-orthogonal to the reconstruction. Stratification: single-codeword "
            "share ≥ 0.20 at the deployed geometry. Ternary: a weight distribution "
            "where ternary beats VQ at matched exact rate on a real forward. "
            "Routing-frequency: ≥ 1000 calibration tokens."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-029",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That activation sparsity (~2× MLP max) or uniform bit-descent "
            "below q3 is a clean path under the Qwen3.8 coherent floor."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "Activation sparsity ~2× MLP max, dynamic, Doctor-risky (compounds "
            "over 62 layers) — marginal, not a clean lever. The MLP is DENSE "
            "(audit D2). Uniform bit-descent: 3.3448 complete BPW is the "
            "coherent floor; Q2 MLP is dead (output rel-fro 0.578 vs q3 0.198). "
            "Shannon bound at coherent distortion is 1.918 b/elem; q3 spends "
            "3.25; sub-1.0 complete is not excluded by information theory but "
            "requires hitting the bound AND remaining coherent at 5.8× the "
            "gated artifact's distortion — unachieved and close to a measured failure."
        ),
        "scope": {
            "model": "qwen3.8-27b",
            "organ": "MLP (sparsity); whole artifact (bit-descent)",
            "regime": "Phase A exhaustion at the 3.3448 patient",
            "codec": "activation skip / flat q2 / uniform descent",
        },
        "evidence": [
            _e("receipts/ascent-2026-08-18/PHASE_A_EXHAUSTION.json",
               3.3448, "representation_front/uniform_bit_descent"),
            _e("receipts/ascent-2026-08-16/G024_RATE_DISTORTION_BOUND.json",
               0.5783472321822679, "anchors/dead/output_rel_fro",
               "flat q2 dead; slb at coherent D is 1.918 vs q3 3.25"),
        ],
        "reopen_condition": (
            "Sparsity: a Doctor-holding skip pattern that does not compound "
            "over 64 layers. Bit-descent: a codec that actually hits the "
            "Shannon bound AND a generate gate at that distortion. Neither "
            "is a retry of q2 or of unstructured activation skip."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-030",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That a family which fails a residual-composition screen to "
            "end-to-end cosine ≥ 0.5 can still be a sub-1.5 static DSV4F body."
        ),
        "kind": PROPERTY,
        "kind_reasoning": (
            "The residual-composition oracle is a rejection instrument, not a "
            "capability certificate. A fail means the family cannot compose "
            "to end-to-end cosine ≥ 0.5 under the stated model — sufficient "
            "to refuse it as a sub-1.5 static body. A pass is not evidence of "
            "a usable model. Naive c^n is the harsher screen; the honest bound "
            "is residual-identity (mean r ≈ 0.34 dilutes organ error)."
        ),
        "scope": {
            "model": "deepseek-v4-flash, 43 layers",
            "organ": "late_hidden residual composition",
            "regime": "sub-1.5 static body screen, 32 sequences / 96 tokens",
            "codec": "family under test (rejection only)",
        },
        "evidence": [
            _e("receipts/DSV4F_RESIDUAL_COMPOSITION_ORACLE.json",
               0.8615764681875107, "disagreement/residual_identity_break_even"),
        ],
        "reopen_condition": (
            "A family that PASSES the identity-product screen is not thereby "
            "promoted; it becomes eligible for a generation gate. A fail stays "
            "a fail for sub-1.5 static bodies."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
    {
        "id": "NNS-031",
        "seed": None,
        "seed_status": "NEW",
        "claim_refuted": (
            "That routing-frequency allocation calibrated on ~88 tokens, or "
            "previous-token route-set prefetch, is a first-touch / allocation win."
        ),
        "kind": ARTIFACT,
        "kind_reasoning": (
            "88 tokens: median routing split only 63.6% stable, 26.1% of cells "
            "never route. The lever (frequency-proportional bits) is alive; "
            "the 88-token calibration is dead. Prefetch of this token's experts "
            "from the previous token's route set cannot see the misses that "
            "cost money: those (layer, expert) pairs are exactly the ones the "
            "previous token did not use, and Q80 experts are mutually orthogonal "
            "so cross-layer overlap is not assumed."
        ),
        "scope": {
            "model": "gpt-oss-120b:F0 (88-token); qwen3-80b (prefetch)",
            "organ": "router / expert residency",
            "regime": "short-calib allocation; previous-token predictor",
            "codec": "routing-frequency bit allocation; prefetch bind",
        },
        "evidence": [
            _e("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
               88, "entries/calibration_88_tokens/killed_by"),
            _e("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
               None, "entries/NS-029"),
        ],
        "reopen_condition": (
            "Allocation: ≥ 1000 calibration tokens. Prefetch: never as a "
            "previous-token route predictor. A corpus-level hot-set prebind "
            "is a different mechanism."
        ),
        "reopen_today": {"predicate": "never", "if_true": None},
    },
]


WHAT_I_WATCHED_FAIL = """\
## WHAT I WATCHED FAIL

The expensive failure in this campaign is not "we tried a codec and it was
bad". It is "we tried a MEASUREMENT and it was blind, then treated the
blindness as a property of the idea". Several seeds above were that. Listing
them as dead ideas would suppress exactly the search that is about to open.

I watched cosine fail first. It is scale-invariant, so 0.01·W scored
1.000000 on every doctor axis while destroying magnitude (rel error 0.9898).
The same cosine, used as a capability certificate, sits at 0.898 for a
constant-mean predictor of GLM block output — so 0.86–0.90 can mean nothing.
The 0.8604 D23 bar was drawn BELOW that null, then mixed-1p5 generated
coherent text with down_proj holdout cosine 0.7684. mixed-2p0 scored 0.907
mean component cosine and is native-INCOHERENT. Cosine is a screen. I
watched it used as a gate.

I watched a Gaussian proxy invert a ranking. Six sub-bit families were
negative on synthetic X. The same families, on real teacher-capsule
activations, inverted; the null moved 0.126 → 0.651; activation-aware
projection reached 0.755 cosine at 0.167 BPW, 12/12 experts. Every prior
sub-bit "dead" in this project traces to that proxy. The idea was not dead.

I watched a gibberish teacher rank the wrong trajectory. Q30 X from a
0.7966 baseline (emits " compreh swiper swiper") made every score point
the wrong way. Independently, even the correct+dense BF16 capture only
lifted L0 SVD cosine +0.015 — that representation's ceiling is real
(NNS-003). Two failures, one campaign. Do not collapse them.

I watched underdetermined fits certify themselves. Median 92 rows against
2048 dims; rank = min(budget, n_fit) silently starving the codec; a 30×
wider corpus making per-expert rows THINNER because retention was
per-layer. The score was not the codec's score.

I watched a shared-operator "breakthrough" (0.371 vs q3 0.401) that did
not survive an adversarial audit: leakage, aggregation mismatch, wrong
input tensor, each larger than the margin, all favoring the operator.
Honest retest: 4.02 vs 0.337. Assembled-Doctor: gibberish. After the
method was fixed, the narrow-bottleneck IDEA failed. That order matters.
The leaked number is not a reopen.

I watched complete BPW used as a capability. The cited 3.5406/3.6139 pair
is not in this tree (nearest: O005 mixed-q2q4 3.5412 DEGRADED). The law
is in this tree several times: 2.0856 INCOHERENT vs 4.2527 COHERENT on
Qwen3.8; 1.444 COHERENT generate on Q80; 3.3448 denser and slower than
4.256. BPW is not coherence and is not velocity.

I watched storage BPW sold as active BPW (0.646 vs 2.518), a reuse-band
sold as a decode ceiling (560–647 vs unique-once 411), a 230× occupancy
gap that was a 0.59 MiB organ judged against a 64 MiB DRAM-row probe,
and a 5.9× reconstruction penalty that was a serial extract transferred
to a different model. Those are category errors. They are in the ascent
register. They are not representation ideas, but they will reappear as
arguments against trying a representation. Do not let them.

What is actually dead, as a property of the idea, on the parents and
organs named:

  * shared bases / templates / tying on Q80 and F0/F1 (experts orthogonal)
  * independent per-layer functional students (router drift by L4–8)
  * narrow shared SwiGLU replacing 64 MLPs (Doctor gibberish after honest eval)
  * functional low-rank at matched bytes (overfits held-out)
  * hybrid low-rank+correction as a byte-reduction lever (Pareto-dominated by q3)
  * weight-space low-rank / Kronecker at depth (L0 is the exception)
  * raw-weight PQ/VQ at ~1 bit (collapses; does not kill source-changing methods)
  * existing Gravity families on attention below Q4 (0.99 bar)
  * Qwen3.8 whole-model ≤1.5 with those families (native incoherent, 0 fallbacks)
  * Q30 static ≤1.5 SVD as a template
  * single-family Q80 packs inside 1.5
  * Lloyd PQ entropy coding as a 10–25% win
  * expert merge from survivors (~1.0 rel error)
  * selection policy over sub-1.2-bpw organs as a coherence route
  * Q30 rank-256 vs 1-bit on [768,2048] (pays more, delivers less)

What is live, and being sat on, because a reopen condition already holds
or a surviving avenue was named and never run: see live_opportunities in
the receipt. Sub-bit on real X is one of them. Layer-0-specific codecs
are another. Distillation of the MLP function is the one Phase B left
standing. A new attention family is the one G006 actually needs.
"""


def git(args: list[str], timeout: int = 30) -> str:
    try:
        r = _git(args, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def read_text(rel: str) -> str | None:
    p = REPO / rel
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    try:
        r = _git(["show", f"HEAD:{rel}"], timeout=60)
        if r.returncode == 0:
            return r.stdout
    except Exception:
        pass
    sib = sibling_science_root()
    if sib is not None:
        p2 = sib / rel
        try:
            if p2.is_file():
                return p2.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return None


def exists_rel(rel: str) -> bool:
    try:
        if (REPO / rel).exists():
            return True
    except OSError:
        pass
    try:
        if _git(["cat-file", "-e", f"HEAD:{rel}"]).returncode == 0:
            return True
    except Exception:
        pass
    sib = sibling_science_root()
    if sib is not None:
        try:
            if (sib / rel).exists():
                return True
        except OSError:
            pass
    return False


def load_json(path: Path | str) -> Any:
    if isinstance(path, str):
        text = read_text(path)
    else:
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
            else:
                try:
                    text = read_text(str(path.relative_to(REPO)))
                except ValueError:
                    text = None
        except OSError:
            text = None
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def jget(obj: Any, path: str | None) -> Any:
    if obj is None or not path:
        return None
    cur = obj
    for part in path.split("/"):
        if cur is None:
            return None
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                continue
            return None
        if isinstance(cur, list):
            if part.isdigit():
                i = int(part)
                cur = cur[i] if 0 <= i < len(cur) else None
                continue
            cur = next(
                (e for e in cur if isinstance(e, dict) and e.get("id") == part),
                None,
            )
            continue
        return None
    return cur


def resolve(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p.resolve()
    sib = sibling_science_root()
    if sib is not None and (sib / rel).exists():
        return (sib / rel).resolve()
    return p.resolve()


def _number_in(blob: str, expected: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, float):
        if str(expected) in blob:
            return True
        # tolerate short forms: 3.5412, 0.755, 1.328
        for n in (f"{expected:.4f}", f"{expected:.3f}", f"{expected:.2f}", f"{expected:g}"):
            if n in blob:
                return True
        return False
    return str(expected) in blob


def confirm_number(rel: str, field: str | None, expected: Any) -> dict:
    text = read_text(rel)
    out = {
        "path": rel,
        "resolves": text is not None,
        "absolute": str(resolve(rel)) if text is not None else None,
        "field": field,
        "expected": expected,
        "observed": None,
        "confirmed": False,
        "note": None,
    }
    if text is None:
        out["note"] = "PATH_MISSING"
        return out
    if rel.endswith(".json") or rel.endswith(".jsonl"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        blob = json.dumps(obj, default=str) if obj is not None else text
        if field:
            observed = jget(obj, field)
            out["observed"] = observed if not isinstance(observed, (dict, list)) else (
                json.dumps(observed)[:240]
            )
            if expected is None:
                out["confirmed"] = observed is not None
            elif isinstance(expected, float) and isinstance(observed, (int, float)):
                out["confirmed"] = abs(float(observed) - float(expected)) <= max(
                    1e-9, 1e-6 * abs(float(expected))
                )
            elif isinstance(observed, str):
                out["confirmed"] = _number_in(observed, expected)
            elif isinstance(observed, (dict, list)):
                out["confirmed"] = _number_in(json.dumps(observed, default=str), expected)
            else:
                out["confirmed"] = observed == expected
            if not out["confirmed"] and expected is not None:
                # key-as-number (2.0856_BPW_...): search the selected object / whole file
                out["confirmed"] = _number_in(blob, expected)
        else:
            out["confirmed"] = _number_in(blob, expected) if expected is not None else True
            out["note"] = "json_blob_search" if expected is not None else "path_resolves_json"
        return out
    out["confirmed"] = _number_in(text, expected) if expected is not None else True
    out["note"] = "text_search"
    return out


def pred_real_x_vindication_on_disk() -> tuple[bool, str]:
    rel = "workspace/campaign/evidence/systems/hawking/HAWKING_HEAVY_CONTINUATION_STATUS.json"
    if not exists_rel(rel):
        return False, "HAWKING_HEAVY_CONTINUATION_STATUS.json missing"
    d = load_json(rel) or {}
    headline = ((d.get("rebuild_glm") or {}).get("step_2_pilot") or {}).get("headline") or ""
    ok = "0.755" in headline and "0.167" in headline
    return ok, headline[:200] if headline else "headline empty"


def pred_gain_axis_exists() -> tuple[bool, str]:
    rel = "tools/gravity_doctor_gate.py"
    text = read_text(rel)
    if text is None:
        return False, "gravity_doctor_gate.py missing"
    ok = "def _gain(" in text and "SCALE-INVARIANT" in text
    return ok, "_gain present; cosine-only campaigns not systematically re-scored"


def pred_bf16_captures_named() -> tuple[bool, str]:
    q80 = "receipts/ascent-2026-08-16/q80-subbit-capability-curve.SUMMARY.json"
    q38 = "receipts/ascent-2026-08-16/G033_FUNCTION_SPACE_RANK_G32.json"
    bits = []
    if exists_rel(q80):
        d = load_json(q80) or {}
        cap = ((d.get("measurement") or {}).get("capture")) or ""
        bits.append(f"q80_capture_named={bool(cap)}")
    if exists_rel(q38):
        d = load_json(q38) or {}
        note = ((d.get("capture") or {}).get("note")) or ""
        bits.append(f"q38_v2={note!r}")
    return exists_rel(q80) and exists_rel(q38), "; ".join(bits) or "named receipts missing"


def pred_mixed_1p5_generated() -> tuple[bool, str]:
    rel = "receipts/ascent-2026-08-16/Q80_MIXED_GENERATE.json"
    if not exists_rel(rel):
        return False, "Q80_MIXED_GENERATE.json missing"
    d = load_json(rel) or {}
    klass = d.get("coherence_class")
    bpw = ((d.get("artifact") or {}).get("complete_physical_bpw"))
    ok = klass == "COHERENT" and isinstance(bpw, (int, float)) and bpw < 1.5
    return ok, f"coherence_class={klass} complete_physical_bpw={bpw}"


def pred_layer0_shannon_gap() -> tuple[bool, str]:
    rel = "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
    if not exists_rel(rel):
        return False, "NEGATIVE_TRANSFER_ATLAS.json missing"
    d = load_json(rel) or {}
    ex = ((d.get("entries") or {}).get("post_hoc_coding_of_frozen_weights") or {}).get("named_exception") or ""
    return "1.328" in ex or "LAYER 0" in ex.upper(), (ex[:180] if ex else "named_exception empty")


def pred_layer0_kronecker_live() -> tuple[bool, str]:
    rel = "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
    if not exists_rel(rel):
        return False, "atlas missing"
    d = load_json(rel) or {}
    ex = ((d.get("entries") or {}).get("kronecker_factorisation") or {}).get("the_exception") or ""
    return "0.0301" in ex and "layer 0" in ex.lower(), (ex[:180] if ex else "the_exception empty")


def pred_reconstruction_is_free() -> tuple[bool, str]:
    rel = "receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json"
    if not exists_rel(rel):
        return False, "QWEN38_RECONSTRUCTION_IS_FREE.json missing"
    d = load_json(rel) or {}
    n = ((d.get("evidence") or {}).get("recon_excess_ns_zero_on"))
    return bool(n), f"recon_excess_ns_zero_on={n!r}"


def pred_dsv4f_pairwise_unmeasured() -> tuple[bool, str]:
    rel = "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json"
    if not exists_rel(rel):
        return True, "register missing; treat DSV4F as unmeasured"
    d = load_json(rel) or {}
    corr = (d.get("attribution_corrections") or [{}])[0]
    text = json.dumps(corr)
    return "No DSV4F pairwise-cosine receipt exists" in text, \
        corr.get("consequence") or "DSV4F orthogonality is not settled science"


def pred_distill_unrun() -> tuple[bool, str]:
    hybrid = "receipts/ascent-2026-08-18/PHASE_B_HYBRID_REFUTED.json"
    if not exists_rel(hybrid):
        return False, "PHASE_B_HYBRID_REFUTED.json missing"
    candidates = [
        "receipts/ascent-2026-08-18/DISTILLED_OPERATOR.json",
        "receipts/ascent-2026-08-18/PHASE_B_DISTILL.json",
        "receipts/ascent-2026-08-16/qwen38-distill-operator.json",
    ]
    present = [c for c in candidates if exists_rel(c)]
    return not present, "no distilled-operator Doctor receipt; PHASE_B_HYBRID.reopen_if still open"


PREDICATES = {
    "real_x_vindication_on_disk": pred_real_x_vindication_on_disk,
    "gain_axis_exists": pred_gain_axis_exists,
    "bf16_captures_named": pred_bf16_captures_named,
    "mixed_1p5_generated": pred_mixed_1p5_generated,
    "layer0_shannon_gap": pred_layer0_shannon_gap,
    "layer0_kronecker_live": pred_layer0_kronecker_live,
    "reconstruction_is_free_at_tpr64": pred_reconstruction_is_free,
    "dsv4f_pairwise_unmeasured": pred_dsv4f_pairwise_unmeasured,
    "distill_operator_unrun": pred_distill_unrun,
    "q80_still_underdetermined": lambda: (
        False,
        "25k capture still has 24326/24576 gate/up pairs rows<2048; scores still untrustworthy",
    ),
    "never": lambda: (False, "reopen condition is 'never' for this premise"),
    "never_for_current_families": lambda: (
        False,
        "current Gravity families remain the ones that failed generate",
    ),
    "never_on_q30_organ": lambda: (False, "Q30 [768,2048] rank-256 still pays more than 1-bit"),
}


def eval_reopen(entry: dict) -> dict:
    spec = entry.get("reopen_today") or {}
    name = spec.get("predicate") or "never"
    fn = PREDICATES.get(name) or PREDICATES["never"]
    try:
        holds, why = fn()
    except Exception as e:
        holds, why = False, f"predicate {name} raised {type(e).__name__}: {e}"
    return {
        "predicate": name,
        "holds_today": bool(holds),
        "why": why,
        "callout": spec.get("if_true") if holds else None,
    }


def sweep_negative_stores() -> dict:
    found: dict[str, Any] = {
        "ascent_register_entries": 0,
        "foundry_atlas_entries": 0,
        "ascension_jsonl_records": 0,
        "haider_director_epochs": 0,
        "g1_arch_negative_rows": 0,
    }
    rel = "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json"
    if exists_rel(rel):
        d = load_json(rel) or {}
        found["ascent_register_entries"] = len(d.get("entries") or [])
        found["ascent_register_counts"] = d.get("counts")
        ns2 = next((e for e in (d.get("entries") or []) if e.get("id") == "NS-002"), None)
        m = (ns2 or {}).get("what_was_measured") or {}
        found["q80_storage_vs_active"] = {
            "storage_bpw": m.get("mixed_sub655_storage_bpw"),
            "active_bpw": m.get("mixed_sub655_active_bpw"),
            "path": rel,
            "field_storage": "entries/NS-002/what_was_measured/mixed_sub655_storage_bpw",
            "field_active": "entries/NS-002/what_was_measured/mixed_sub655_active_bpw",
        }
    rel = "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
    if exists_rel(rel):
        d = load_json(rel) or {}
        found["foundry_atlas_entries"] = len(d.get("entries") or {})
        found["foundry_atlas_keys"] = sorted((d.get("entries") or {}).keys())
    rel = "workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl"
    text = read_text(rel)
    if text is not None:
        n = 0
        mechs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
                mechs.append(rec.get("mechanism"))
            except Exception:
                pass
        found["ascension_jsonl_records"] = n
        found["ascension_jsonl_mechanisms"] = mechs
    rel = ".haider/bootstrap-director-v6/negative-science.jsonl"
    text = read_text(rel)
    if text is not None:
        rows = [ln for ln in text.splitlines() if ln.strip()]
        found["haider_director_epochs"] = len(rows)
        found["haider_director_note"] = (
            "HCLI tactic-cycle fingerprints, not representation science. "
            "Recorded so a later reader does not re-sweep it expecting codec closures."
        )
    rel = "hawking-experiments/superwave/g1/g1-arch-negative.md"
    text = read_text(rel)
    if text is not None:
        found["g1_arch_negative_rows"] = (
            text.count("| KILLS") + text.count("| CLOSED") + text.count("| DEAUTHORISED")
        )
        found["g1_arch_negative_note"] = (
            "prior architecture ledger; this receipt is the representation-scoped successor"
        )
    return found


def sweep_worktrees_and_branches() -> dict:
    wt = git(["worktree", "list", "--porcelain"])
    worktrees: list[dict] = []
    cur: dict = {}
    for line in (wt or "").splitlines():
        if line.startswith("worktree "):
            if cur:
                worktrees.append(cur)
            cur = {"path": line.split(" ", 1)[1]}
        elif line.startswith("branch "):
            cur["branch"] = line.split(" ", 1)[1]
        elif line.startswith("HEAD "):
            cur["head"] = line.split(" ", 1)[1]
    if cur:
        worktrees.append(cur)
    branches = [b.strip() for b in (git(["branch", "--list", "grok/*"]) or "").splitlines() if b.strip()]
    return {
        "worktrees": worktrees,
        "grok_branch_count": len(branches),
        "grok_branches_current_sample": branches[:12],
        "reading": (
            "Live grok/* branches on this HEAD are HCLI / visionmcp isolation lanes, "
            "not representation-science lanes. Historical representation work lives "
            "in receipts/ascent-2026-08-16, receipts/ascent-2026-08-18, "
            "tools/foundry, and the knowledge-plane jsonl — not in the current grok/* tips."
        ),
    }


def sweep_untracked() -> dict:
    status = git(["status", "--porcelain", "-u", "--",
                  "receipts", "reports", ".haider", "hawking-experiments/superwave", "tools/headless"])
    untracked = [line[3:] for line in (status or "").splitlines() if line.startswith("?? ")]
    return {
        "untracked_under_science_roots_sample": untracked[:40],
        "count": len(untracked),
        "note": "listed, not ingested. Catalog cites tracked receipts.",
    }


def writable_out() -> tuple[Path, str]:
    env = os.environ.get("NOETIC_OUT")
    if env:
        p = Path(env)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p, "env:NOETIC_OUT"
    dest = REPO / "receipts" / "headless" / "NOETIC_NEGATIVE_SCIENCE.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    probe = dest.parent / ".noetic_write_probe"
    probe.write_text("ok")
    probe.unlink()
    return dest, "repo"


def try_self_install() -> dict:
    src = Path(__file__).resolve()
    dest = REPO / "tools" / "headless" / "noetic_negative_science.py"
    if dest.exists() and dest.resolve() == src:
        return {"installed": True, "path": str(dest.relative_to(REPO)), "note": "already in place"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return {"installed": True, "path": str(dest.relative_to(REPO)), "note": "copied from " + str(src)}


def build_entry(raw: dict) -> dict:
    evidence_out = []
    for ev in raw["evidence"]:
        conf = confirm_number(ev["path"], ev.get("field"), ev.get("number"))
        evidence_out.append({
            **ev,
            "resolves": conf["resolves"],
            "confirmed": conf["confirmed"],
            "observed": conf["observed"],
            "confirm_note": conf.get("note"),
        })
    reopen = eval_reopen(raw)
    return {
        "id": raw["id"],
        "seed": raw.get("seed"),
        "seed_status": raw["seed_status"],
        "seed_correction": raw.get("seed_correction"),
        "claim_refuted": raw["claim_refuted"],
        "kind": raw["kind"],
        "kind_reasoning": raw["kind_reasoning"],
        "scope": raw["scope"],
        "evidence": evidence_out,
        "reopen_condition": raw["reopen_condition"],
        "reopen_satisfied_today": reopen["holds_today"],
        "reopen_today": reopen,
        "all_evidence_resolves": all(e["resolves"] for e in evidence_out),
        "all_numbers_confirmed": all(
            e["confirmed"] for e in evidence_out if e.get("number") is not None
        ),
    }


def main() -> int:
    os.chdir(REPO)
    install = try_self_install()
    out_path, out_where = writable_out()
    entries = [build_entry(raw) for raw in CATALOG]
    live = [e for e in entries if e["reopen_satisfied_today"]]
    seeds = [e for e in entries if e.get("seed")]
    n_prop = sum(1 for e in entries if e["kind"] == PROPERTY)
    n_art = sum(1 for e in entries if e["kind"] == ARTIFACT)
    missing = [e["id"] for e in entries if not e["all_evidence_resolves"]]
    unconfirmed = [e["id"] for e in entries if not e["all_numbers_confirmed"]]

    receipt = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git(["rev-parse", "HEAD"]) or None,
        "repo": str(REPO),
        "wrote_to": str(out_path),
        "wrote_where": out_where,
        "self_install": install,
        "obligation": (
            "Every representation idea this project already killed, classified "
            "PROPERTY_OF_IDEA vs ARTIFACT_OF_METHOD, with a reopen condition "
            "and whether that condition holds TODAY. A refutation of a "
            "measurement is not a refutation of an idea."
        ),
        "live_opportunities_being_sat_on": [
            {
                "id": e["id"],
                "kind": e["kind"],
                "callout": e["reopen_today"].get("callout"),
                "reopen_condition": e["reopen_condition"],
                "why_today": e["reopen_today"].get("why"),
            }
            for e in live
        ],
        "seed_audit": [
            {
                "id": e["id"],
                "seed": e["seed"],
                "status": e["seed_status"],
                "correction": e.get("seed_correction"),
                "kind": e["kind"],
            }
            for e in seeds
        ],
        "counts": {
            "entries": len(entries),
            "property_of_idea": n_prop,
            "artifact_of_method": n_art,
            "seeds": len(seeds),
            "seeds_confirmed": sum(1 for e in seeds if e["seed_status"] == "CONFIRMED"),
            "seeds_corrected": sum(1 for e in seeds if e["seed_status"] == "CORRECTED"),
            "live_opportunities": len(live),
            "evidence_paths_missing": missing,
            "numbers_unconfirmed": unconfirmed,
        },
        "what_i_watched_fail": WHAT_I_WATCHED_FAIL.strip(),
        "entries": entries,
        "sweep": {
            "stores": sweep_negative_stores(),
            "worktrees_and_grok_branches": sweep_worktrees_and_branches(),
            "untracked": sweep_untracked(),
            "roots_read": SWEEP_ROOTS,
            "denied_write": ["crates", "workspace", "visionmcp", "app", "lab", "tools/haider"],
        },
        "how_to_use": [
            "Before proposing a representation, match it by id or by the claim string.",
            "If kind=PROPERTY_OF_IDEA and reopen_satisfied_today=false, do not re-derive it on that parent/organ/regime.",
            "If kind=ARTIFACT_OF_METHOD, the IDEA is not dead — retry only under the reopen condition.",
            "If reopen_satisfied_today=true, that is a live opportunity being sat on; it is at the top of this receipt.",
            "Cite the settling receipt path, not a later summary that still carries a superseded number.",
            "DSV4F orthogonality is not settled. Measuring it is a cheap check, not a retry of Q80.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")

    try:
        rel = str(out_path.relative_to(REPO))
    except ValueError:
        rel = str(out_path)

    print("NOETIC NEGATIVE SCIENCE")
    print("=" * 72)
    print(f"git_head: {receipt['git_head']}")
    print(f"repo:     {REPO}")
    print(f"wrote:    {rel}  [{out_where}]")
    print(f"install:  {install}")
    print(
        f"entries:  {len(entries)}  "
        f"({n_prop} PROPERTY_OF_IDEA / {n_art} ARTIFACT_OF_METHOD)"
    )
    print(
        f"seeds:    {receipt['counts']['seeds_confirmed']} confirmed, "
        f"{receipt['counts']['seeds_corrected']} corrected"
    )
    print(f"evidence paths missing: {missing or 'none'}")
    print(f"numbers unconfirmed:    {unconfirmed or 'none'}")
    q80 = (receipt.get("sweep") or {}).get("stores") or {}
    q80 = q80.get("q80_storage_vs_active") or {}
    if q80:
        print(
            f"Q80 storage_bpw={q80.get('storage_bpw')}  "
            f"ACTIVE_bpw={q80.get('active_bpw')}  "
            f"path={q80.get('path')}  "
            f"(never quote one without the other)"
        )
    print()
    print("LIVE OPPORTUNITIES BEING SAT ON")
    print("-" * 72)
    if not live:
        print("  (none — no reopen condition holds today)")
    for e in live:
        print(f"  {e['id']}  [{e['kind']}]")
        print(f"    {e['reopen_today'].get('callout')}")
        print(f"    today: {e['reopen_today'].get('why')}")
    print()
    print("SEED AUDIT")
    print("-" * 72)
    for e in seeds:
        print(f"  {e['id']}  {e['seed_status']:10}  {e['seed']}")
        print(f"    kind={e['kind']}")
        if e.get("seed_correction"):
            print(f"    CORRECTION: {e['seed_correction'][:220]}")
    print()
    print("CATALOG")
    print("-" * 72)
    for e in entries:
        flag = "LIVE" if e["reopen_satisfied_today"] else "    "
        print(f"  {e['id']:8}  {e['kind']:20}  {flag}  {e['claim_refuted'][:70]}")
        for ev in e["evidence"]:
            mark = "OK" if ev["resolves"] and (ev["number"] is None or ev["confirmed"]) else "!!"
            num = ev["number"]
            print(f"           [{mark}] {ev['path']}" + (f"  number={num}" if num is not None else ""))
        print(f"           reopen TODAY={e['reopen_satisfied_today']}: {e['reopen_condition'][:110]}")
    print()
    print(WHAT_I_WATCHED_FAIL)
    print()
    print(f"wrote {rel} ({out_path.stat().st_size} bytes)")
    if out_where != "repo" or not install.get("installed"):
        print("WRITE_SCOPE_BLOCKED: Seatbelt/sandbox refused writes under the repo.")
        print("This task needs the unsandboxed `gate` profile to land")
        print("  tools/headless/noetic_negative_science.py")
        print("  receipts/headless/NOETIC_NEGATIVE_SCIENCE.json")
        return 3
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
