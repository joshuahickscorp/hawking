"""EXPERT_BANK_SCHOOL — generators for structured expert STORAGE sharing
and COMPUTE sharing. Naive global similarity, trivial shared basis, and
unchanged archetypes are recorded-dead and refused.

This module emits candidates and their cheapest falsifiers. It does not
fit real weights. Specimens are out of scope; that work belongs to a
later funnel stage. Every receipt is STATIC_ONLY / bench UNKNOWN.

    python3 tools/future/expert_bank_school.py --build
    python3 tools/future/expert_bank_school.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
from pathlib import Path
from typing import Any

from tools.future._common import write_receipt, load_json, REPO

RECEIPT = "EXPERT_BANK_SCHOOL.json"
SCHEMA = "hawking.future.expert_bank_school.v1"

INDEX_PATH = REPO / "receipts" / "future" / "NEGATIVE_SCIENCE_INDEX.json"

# Atlas / scar files a sibling index would have summarized. Consulted on
# disk when present; never imported as Python modules.
ATLAS_PATHS = (
    "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
    "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
    "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
    "receipts/headless/EXPERT_FAMILY_GENOME.json",
    "receipts/headless/C1SHAREDBASIS_DESIGN.json",
    "receipts/headless/SHARED_BASIS_COHERENT.json",
    "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json",
)

NAMED_FLASH_ABSENT = (
    "receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json",
    "tools/flash_route_conditioned_shared_basis.py",
    "tools/flash_route_archetype_sparse_screen.py",
    "tools/flash_expert_bank_profile.py",
    "tools/flash_doctor_bank_screen.py",
)

STORAGE_FIELDS = (
    "id",
    "axis",
    "kind",
    "mechanism",
    "why_it_might_work",
    "native_execution_concept",
    "forbids_dense_rematerialization",
    "cheapest_falsifier",
    "expected_byte_effect",
    "expected_compute_effect",
    "capability_risk",
    "scar_distance",
    "status",
    "evidence_class",
)
COMPUTE_FIELDS = STORAGE_FIELDS + (
    "repeated_computation",
    "why_currently_repeated",
)

REQUIRED_STORAGE_KINDS = (
    "common_left_subspaces",
    "common_right_subspaces",
    "expert_specific_small_cores",
    "tensor_decomposition",
    "clustered_subspaces",
    "dictionary_families",
    "route_conditioned_archetypes",
    "expert_embeddings_generators",
    "shared_input_transforms",
    "shared_output_latent_spaces",
    "cross_layer_expert_prediction",
    "conditional_residuals",
    "capability_sensitive_expert_islands",
)
REQUIRED_COMPUTE_KINDS = (
    "one_hidden_vector_many_experts",
    "shared_xb_then_skinny",
    "latent_weighted_reduction",
    "one_output_expansion",
    "shared_representation_decode",
    "shared_projections_across_organs",
    "cross_layer_reused_transforms",
)

DEAD_FAMILY_RAW = "RAW_GLOBAL_SIMILARITY"
DEAD_FAMILY_BASIS = "TRIVIAL_SHARED_BASIS"
DEAD_FAMILY_ARCHETYPE = "UNCHANGED_ARCHETYPE"
DEAD_FAMILIES = (DEAD_FAMILY_RAW, DEAD_FAMILY_BASIS, DEAD_FAMILY_ARCHETYPE)


class DeadHypothesisError(ValueError):
    """Raised when a candidate matches a recorded-dead hypothesis."""

    def __init__(self, candidate_id: str, scar: dict[str, Any]):
        self.candidate_id = candidate_id
        self.scar = scar
        super().__init__(
            f"REFUSED {candidate_id}: dead hypothesis {scar['id']} "
            f"({scar.get('title') or scar.get('family')})"
        )


class CandidateSchemaError(ValueError):
    """Raised when a live candidate is missing a required field."""


# ---------------------------------------------------------------------------
# Recorded-dead families. Recovered from disk / git during authoring; encoded
# here so a sparse checkout still refuses them. Phrases are long and specific
# so structured cousins (route-conditioned archetypes, one-sided subspaces)
# are not blanket-killed.
# ---------------------------------------------------------------------------

BUILTIN_SCARS: tuple[dict[str, Any], ...] = (
    {
        "id": "SCAR-RAW-GLOBAL-SIMILARITY",
        "family": DEAD_FAMILY_RAW,
        "title": "raw global expert similarity",
        "phrases": (
            "raw global expert similarity",
            "raw global similarity",
            "inter-expert redundancy",
            "flatten-and-cosine sharing",
            "mean pairwise cosine sharing",
            "shared mean expert subtraction",
            "cluster-mean subtraction across experts",
        ),
        "why_dead": (
            "Q80 L10 n=96: gate pairwise cosine mean 0.00414, up ~0; "
            "top-32 right-subspace overlap 0.025/0.020; k90=84/96. "
            "Qwen3-30B-A3B L2 n=32: mean cosine 0.0024/5.6e-5/-1e-6; "
            "subtracting the mean expert leaves residual norm 0.984. "
            "Foundry atlas inter_expert_redundancy: mean pairwise cosine 1e-4. "
            "NS-010 / NNS-004: shared cross-expert basis/codebook/template "
            "is REFUTED on Q80/F0/F1."
        ),
        "settled_by": (
            "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
            "receipts/headless/EXPERT_FAMILY_GENOME.json",
            "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json#inter_expert_redundancy",
            "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json#NS-010",
            "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#NNS-004",
        ),
        "reopen_condition": (
            "Never on Q80 / F0 / F1. On a NEW parent, measure pairwise cosine "
            "(raw AND row-normalized) on THAT parent's weights and reopen "
            "only if mean >= 0.10."
        ),
        "scope": "MODEL_SPECIFIC on Q80/F0/F1; DSV4F pairwise cosine is still unmeasured",
    },
    {
        "id": "SCAR-TRIVIAL-SHARED-BASIS",
        "family": DEAD_FAMILY_BASIS,
        "title": "trivial shared basis",
        "phrases": (
            "trivial shared basis",
            "unconditioned shared weight basis",
            "one shared weight basis for all experts",
            "shared bf16 basis nf4 residual",
            "unconditioned shared codebook across routed experts",
        ),
        "why_dead": (
            "C1SHAREDBASIS_DESIGN: sharing lost on FIDELITY, not because a "
            "kernel rematerialized dense W (associativity identity holds). "
            "Verdict NOT_WORTH_BUILDING. SHARED_BASIS_COHERENT: no coherent "
            "shared-basis point beats q2f on both density and ns "
            "(died_at=held_out_activation). Flash bounded slice: shared "
            "bf16 basis + nf4 residual was 256000 bytes vs independent q4 "
            "174080 bytes (not smaller). NNS-004 / NS-010 kill the "
            "unconditioned shared-basis/codebook family on Q80."
        ),
        "settled_by": (
            "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            "receipts/headless/SHARED_BASIS_COHERENT.json",
            "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json",
            "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#NNS-004",
        ),
        "reopen_condition": (
            "Never as an unconditioned weight-space shared basis on Q80/Qwen3.8. "
            "A one-sided, clustered, route-conditioned, or activation-PCA "
            "object is a different hypothesis and is not this scar."
        ),
        "scope": "unconditioned weight-space shared basis / shared codebook",
    },
    {
        "id": "SCAR-UNCHANGED-ARCHETYPE",
        "family": DEAD_FAMILY_ARCHETYPE,
        "title": "unchanged archetype",
        "phrases": (
            "unchanged archetype",
            "static unchanged archetype",
            "archetype used unchanged",
            "same-index cross-layer weight tying",
            "procedural generated expert bases on orthogonal experts",
        ),
        "why_dead": (
            "Foundry atlas cross_expert_and_cross_layer_tying: best single "
            "shared template explains 0.2513 of 4 experts' energy against an "
            "orthogonal null of exactly 0.2500; same-index cross-layer tying "
            "is indistinguishable from a different-index control at 1e-7. "
            "NNS-024: procedural / generated parameters (shared templates, "
            "same-index tying, generated expert bases) do not buy bits on "
            "mutually orthogonal experts. No file named "
            "flash_route_archetype_sparse_screen.py exists in this tree; "
            "the scar is the atlas/NNS template family."
        ),
        "settled_by": (
            "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json#cross_expert_and_cross_layer_tying",
            "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#NNS-024",
        ),
        "reopen_condition": (
            "Never as an unchanged template / same-index weight tie on the "
            "measured parents. A route-conditioned or residual-adapted "
            "archetype is a different hypothesis."
        ),
        "scope": "unchanged templates and same-index weight tying",
    },
)


# ---------------------------------------------------------------------------
# Storage-sharing candidates. Each is a structured cousin of a dead family,
# not a rerun. Native path must consume factors without rematerializing W.
# ---------------------------------------------------------------------------

def _store(
    kind: str,
    cid: str,
    mechanism: str,
    why: str,
    native: str,
    falsifier: str,
    byte_effect: str,
    compute_effect: str,
    risk: str,
    scar_distance: str,
) -> dict[str, Any]:
    return {
        "id": cid,
        "axis": "STORAGE",
        "kind": kind,
        "mechanism": mechanism,
        "why_it_might_work": why,
        "native_execution_concept": native,
        "forbids_dense_rematerialization": True,
        "cheapest_falsifier": falsifier,
        "expected_byte_effect": byte_effect,
        "expected_compute_effect": compute_effect,
        "capability_risk": risk,
        "scar_distance": scar_distance,
        "status": "HYPOTHESIS_UNFITTED",
        "evidence_class": "STATIC_ONLY",
    }


STORAGE_CANDIDATES: tuple[dict[str, Any], ...] = (
    _store(
        "common_left_subspaces",
        "STORE-COMMON-LEFT-SUBSPACE",
        "common left subspaces: share an output-side factor U across experts; "
        "keep expert-specific right factors C_e. Algebra y = U @ (C_e @ x) "
        "without forming W_e = U @ C_e.",
        "Q80's pairwise cosine kill is on flattened W, not on a joint SVD of "
        "the stacked left spaces. Left (output) subspaces were never the "
        "measured object in QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE (that "
        "receipt reports right-subspace overlap of two experts). Geometry "
        "of down_proj is tall-output: sharing U amortizes the wide side.",
        "Bind one U[d_model, r] per layer-organ and gathered C_e[r, d_in] "
        "for the selected experts. Compute z_e = C_e @ x (skinny), then "
        "y = U @ sum_e w_e z_e (or U @ z_e per expert if U is only partly "
        "shared). Never write W_e into DRAM.",
        "Instrument first on a planted U_shared @ C_e stack (must recover "
        "shared energy) and an orthogonal control (must not). Then, at a "
        "later funnel stage that owns a specimen, one layer / one organ / "
        "8 experts: joint SVD of stacked W_e W_e^T energy in top-r. Kill "
        "if top-r energy is indistinguishable from the orthogonal null. "
        "Do not flatten-and-cosine. Do not fit the full bank in this lane.",
        "Amortizes the wide output factor across E experts; per-expert "
        "payload shrinks to skinny C_e. Magnitude UNKNOWN pending specimen.",
        "One shared expansion GEMV per token plus selected skinny maps. "
        "ns UNKNOWN (no GPU lease).",
        "If left spaces are as disjoint as Q80 right spaces, U captures "
        "noise and every expert loses the same complement. Gate is the "
        "historical first-break organ.",
        "Not raw global similarity (does not flatten W or use pairwise "
        "cosine). Not a trivial shared basis (one-sided, not U_shared @ "
        "V_shared for all experts).",
    ),
    _store(
        "common_right_subspaces",
        "STORE-COMMON-RIGHT-SUBSPACE",
        "common right subspaces: share an input-side factor V across experts "
        "in WEIGHT space of the stacked bank; expert-specific left cores "
        "C_e. Algebra y_e = C_e @ (V @ x).",
        "A joint right factor of the expert STACK is a different object "
        "from pairwise top-32 overlap of expert 0 vs 1 (Q80 overlap 0.025 "
        "is a risk, not a joint-SVD kill). If a low-rank joint V exists, "
        "every selected expert reuses one V @ x.",
        "Bind one V[r, d_in] and gathered C_e[d_out, r]. Compute z = V @ x "
        "once per organ (or once per layer if gate/up share V), then "
        "y_e = C_e @ z. Never form W_e = C_e @ V.",
        "Planted shared-V stack vs orthogonal control (instrument). Then "
        "one-layer joint SVD of stacked W_e^T W_e. Kill if top-r energy "
        "matches the orthogonal null at the r the byte plan needs. Pairwise "
        "overlap of two experts is not this test.",
        "Amortizes the wide input factor; payload is skinny C_e. Magnitude "
        "UNKNOWN. Q80 pairwise overlap 0.025 is a prior that this may die.",
        "One shared V @ x plus selected skinny maps. ns UNKNOWN.",
        "Q80 L10 pairwise right-overlap is already near null; a joint V "
        "may still exist and still be useless at the r that saves bytes. "
        "Do not treat a pairwise kill as this test, and do not treat a "
        "joint-V miss as a license to rerun flatten-and-cosine.",
        "Not trivial shared basis (one-sided joint factor of the stack, "
        "not one unconditioned basis plus nf4 residual on raw W). Not "
        "activation-PCA (that is shared_input_transforms).",
    ),
    _store(
        "expert_specific_small_cores",
        "STORE-EXPERT-SMALL-CORE",
        "expert-specific small cores: W_e ≈ U @ C_e @ V with C_e a small "
        "r x r (or r x s) core unique to the expert; U, V shared.",
        "Orthogonality of flattened W does not forbid a shared sandwich "
        "with a mixing core. The core is the expert's identity; U and V "
        "are the bank's geometry. This is a parameterization, not a claim "
        "that experts look alike.",
        "Bind U, V once and C_e for selected experts. Path: z = V @ x, "
        "u_e = C_e @ z, y_e = U @ u_e (or fused). Three GEMVs, none of "
        "which materializes W_e. C_e stays in registers if r is small.",
        "Planted sandwich recovers C_e energy; orthogonal W_e must force "
        "r ≈ min(d_out, d_in) to fit. Kill if the r that beats independent "
        "coding is not small enough to cut bytes after storing U and V once.",
        "Payload per expert is C_e; U,V amortized over E. Win only if r "
        "is small. Magnitude UNKNOWN.",
        "Two shared GEMVs per token plus a tiny core GEMV per selected "
        "expert. ns UNKNOWN.",
        "If C_e must be nearly square at full rank, this is a byte loss "
        "(U,V plus cores exceed independent W). Capability follows the "
        "core rank: too-small C_e is a systematic, all-expert failure.",
        "Not an unchanged archetype (the core changes per expert). Not a "
        "trivial shared basis (sandwich with a mixing core, not residual "
        "on a frozen shared W).",
    ),
    _store(
        "tensor_decomposition",
        "STORE-TENSOR-DECOMPOSITION",
        "tensor decomposition of the expert BANK as a 3-way tensor "
        "(expert × out × in): CP / Tucker / tensor-train on the stack, "
        "consumed factor-wise.",
        "Single-tensor Kronecker of one expert at layers >= 1 is dead "
        "(flat Van Loan spectrum; NNS-016) with a named L0 exception. "
        "A 3-way decomposition of the bank can exist even when each slice "
        "is full-rank and pairwise cosine is ~0, because the expert mode "
        "is a new axis.",
        "Store CP atoms (a_r, u_r, v_r) or Tucker (G, A, U, V). Native: "
        "y = sum_r a_e,r * (u_r @ (v_r @ x)) or the Tucker contraction "
        "along in, then mix by expert row of A. Never reconstruct a dense "
        "slice W_e.",
        "Do not rerun single-tensor Kronecker at depth. Falsify the BANK "
        "decomposition: CP rank needed for a planted 3-way vs for real "
        "stack energy (later funnel). Kill if CP rank scales like E * "
        "min(d_out, d_in) (then there is no expert-mode structure).",
        "If the expert mode has low CP/Tucker rank, bytes scale as "
        "R*(E + d_out + d_in) not E*d_out*d_in. Magnitude UNKNOWN. "
        "L0 single-tensor Kronecker remaining live is a different object.",
        "R shared rank-1 (or Tucker) contractions instead of E independent "
        "matvecs. ns UNKNOWN.",
        "Depth layers may have a flat expert-mode spectrum the way they "
        "have a flat Van Loan spectrum. Do not transfer the L0 Kronecker "
        "win to the bank, and do not transfer the depth Kronecker kill "
        "to the bank without measuring the expert mode.",
        "Not NNS-016 (that is one expert tensor, Kronecker/TT/low-rank, "
        "layers >= 1). The expert index is an additional mode.",
    ),
    _store(
        "clustered_subspaces",
        "STORE-CLUSTERED-SUBSPACES",
        "clustered subspaces: partition experts into K << E clusters; "
        "each cluster shares a left or right factor; members keep a "
        "skinny residual. All experts remain present.",
        "Global one-space sharing can fail while K cluster spaces succeed. "
        "Q80 k90=84/96 in expert-space energy says a SINGLE shared space "
        "needs almost E components; that does not bound K-way clustered "
        "spaces. This is not merging: omitted experts are not reconstructed.",
        "Bind K shared factors and a cluster-id table (log2 K bits/expert). "
        "Selected expert e in cluster k uses V_k @ x then C_e. Gather "
        "V_k by cluster id, not by reconstructing W_e.",
        "Kill if the K that beats the orthogonal null is ~E (then clustering "
        "is a rename of independent experts). Kill if the method reconstructs "
        "a held-out expert from cluster mates (that is dead expert-merging, "
        "atlas expert_merging_omitted_from_survivors / NNS-023).",
        "K shared wide factors plus skinny members. Win vs one-space only "
        "if K is small. Magnitude UNKNOWN.",
        "Up to min(K_selected, top-k) shared transforms per token. ns UNKNOWN.",
        "Atlas expert-merging is DEAD (best survivor rel-err ~0.89-1.0). "
        "A cluster that silently becomes a merge will inherit that wall. "
        "Capability islands may not match cosine clusters.",
        "Not raw global similarity (K spaces, not one). Not expert merging "
        "(every expert is stored; no omitted-expert reconstruction). Not "
        "an unchanged archetype (cluster factor plus member residual).",
    ),
    _store(
        "dictionary_families",
        "STORE-DICTIONARY-FAMILIES",
        "dictionary families: a shared dictionary of linear atoms; each "
        "expert is a sparse combination of those atoms, executed as a sum "
        "of gathered low-rank factors.",
        "A single shared template is null (0.2513 vs 0.2500). A dictionary "
        "of many atoms with sparse codes can still span orthogonal experts "
        "the way a codebook spans orthogonal residuals, without claiming "
        "experts are similar. Source must not be raw-weight PQ (NS-raw-PQ "
        "dead at ~1 and ~0.49 BPW).",
        "Keep dictionary atoms as GEMM factors in registers/SRAM (tiny K). "
        "Expert e holds sparse indices and scales. y_e = sum_{k in supp(e)} "
        "s_{e,k} * (atom_k @ x). Lookup-free FMA. Never densify W_e from "
        "the atom sum. Large LUT gathers are a recorded dead lever.",
        "Kill if the K needed to hit a planted reconstruction is so large "
        "the dictionary plus codes exceed independent coding. Kill if atoms "
        "are applied by densifying W_e. Do not entropy-code Lloyd indices "
        "as the byte win (atlas entropy_coded_pq_indices).",
        "Atoms amortized; per-expert payload is sparse codes. Win only at "
        "tiny K and lookup-free consume. Magnitude UNKNOWN.",
        "nnz(e) atom applications per selected expert. ns UNKNOWN.",
        "Raw-weight PQ/VQ at one bit collapsed on real forwards. Dictionary "
        "atoms that are just a PQ codebook on frozen W inherit that kill. "
        "Apple GPU large-LUT gather is also a dead lever.",
        "Not NS-010 unconditioned shared codebook on raw routed-expert "
        "weights. Not a single unchanged template. Not SB6-large-LUT.",
    ),
    _store(
        "route_conditioned_archetypes",
        "STORE-ROUTE-CONDITIONED-ARCHETYPES",
        "route-conditioned archetypes: a small set of archetypal operators "
        "specialized by the live router score (or top-k identity), producing "
        "an adapted expert without storing a full W_e.",
        "Unchanged templates failed because they ignore the route. The "
        "router already computed a cheap score per expert; using that score "
        "as a conditioner is information the dead template threw away. "
        "Adaptation can be a scale, a mixing of K archetypes, or a low-rank "
        "delta gated by the score.",
        "Bind K archetypal factors. Given router scores π, form a mixed "
        "core C(π) = sum_k g_k(π) C_k (or a sparse top-k mix) and apply "
        "C(π) @ (V @ x) natively. The kernel reads π and the archetypal "
        "factors; it does not write a dense adapted W.",
        "Kill if g(π)=constant on held-out routes (then this collapses to "
        "the unchanged-archetype scar). Kill if mixing K archetypes cannot "
        "beat a single archetype by more than the orthogonal-null margin "
        "that already killed templates (0.0013 energy).",
        "K archetypes plus a conditioner instead of E full experts. "
        "Magnitude UNKNOWN.",
        "One conditioned mix plus the shared transform. ns UNKNOWN.",
        "If the router score does not predict the needed adaptation, this "
        "becomes the dead template with extra math. Over-mixing can smear "
        "capability across experts the router meant to keep distinct.",
        "Not an unchanged archetype: the operator depends on the live route. "
        "The named Flash route-conditioned / archetype-sparse tools are "
        "absent from this tree; this candidate is the untested structured "
        "form, not a rerun of a missing screen.",
    ),
    _store(
        "expert_embeddings_generators",
        "STORE-EXPERT-EMBEDDING-GENERATOR",
        "expert embeddings / generators: each expert is a small embedding "
        "h_e; a shared generator G maps h_e to the factors a kernel consumes "
        "(cores, codes, or skinny maps), not to a dense W_e.",
        "NNS-024 killed generated bases that buy bits BECAUSE experts are "
        "similar. A generator as a compressed parameterization can represent "
        "orthogonal experts if G is expressive; the embedding is identity, "
        "not a similarity exploit. Bytes live in G once plus E embeddings.",
        "At bind time, G(h_e) emits C_e (or sparse codes) into the same "
        "buffers the small-core / dictionary kernel already consumes. The "
        "token path never calls G if cores are materialized as skinny "
        "factors; G is a storage generator, not a per-token network. Never "
        "emit dense W_e from G.",
        "Kill if ||G(h_e) - target_factors|| requires h_e or G large enough "
        "that embeddings+G exceed independent skinny factors. Kill if G "
        "emits dense W. Do not score this by pairwise cosine of W.",
        "G stored once; payload is h_e. Win only if dim(h_e)+|G| << E*|W|. "
        "Magnitude UNKNOWN.",
        "Token path identical to the emitted-factor kernel; G is bind-time. "
        "ns UNKNOWN on the token path by construction.",
        "An underfit G is a systematic capability failure across the bank. "
        "NNS-024 remains in force if the generator is just a shared template "
        "in disguise.",
        "Not NNS-024 procedural generated expert bases on orthogonal "
        "experts: this does not assume similarity and does not emit W. "
        "Not an unchanged archetype (h_e differs; G is shared).",
    ),
    _store(
        "shared_input_transforms",
        "STORE-SHARED-INPUT-TRANSFORM",
        "shared input transforms: one activation-space transform z = V @ x "
        "from the real residual stream (PCA / Hessian-weighted), then "
        "expert-specific skinny maps on z. Experts may be orthogonal as W "
        "and still share input covariance.",
        "The surviving premise of the shared-basis packet: the residual "
        "stream is one vector. Weight-space tying is dead; sharing the "
        "transform of x does not require W cosine. This is SB1 in that "
        "packet, never fitted here, still untested on a Doctor path.",
        "Compute z = V @ x once per layer (serves gate+up if they share V). "
        "Selected experts apply skinny C_e to z. Down: skinny then shared "
        "U. Lookup-free GEMM. Forbidden: materialize W_e = C_e V then qmm.",
        "One layer, one organ, real X (never Gaussian). Uncoded V from "
        "activation PCA; reconstruct C_e (V x) vs W_e x. Kill if output "
        "cosine is no better than an orthogonal-V control at the r the "
        "byte plan needs. If uncoded already dies, quantizing C_e cannot "
        "save it. Do not run on a specimen in this lane.",
        "Skinny payload scales as r/d_in of the expert mass plus amortized "
        "V. Algebraic identity only; magnitude UNKNOWN pending specimen.",
        "One shared V @ x plus top-k skinny maps, vs top-k full matvecs. "
        "ns UNKNOWN.",
        "If XX^T energy at r is low, every expert loses the same complement "
        "(systematic). Gate/up historically break first. Gaussian X is a "
        "recorded artifact (NS-009 / NNS-001) and is forbidden as the "
        "falsifier's input.",
        "Not trivial shared basis (activation-space V, not a shared weight "
        "basis plus residual on W). Not raw global similarity (does not "
        "use W cosine).",
    ),
    _store(
        "shared_output_latent_spaces",
        "STORE-SHARED-OUTPUT-LATENT",
        "shared output latent spaces: each expert emits a small latent "
        "u_e; a shared U expands to d_model. Combine can happen in the "
        "latent. Dual of shared input transforms, on the output side.",
        "down_proj is the historically tolerant organ (hgravs01 live on "
        "Q80 down as a per-expert low-rank screen). Sharing the expansion "
        "U is the storage form of 'one output expansion' on the compute "
        "axis. Combine-in-latent is cheaper than expand-then-sum.",
        "Selected experts write u_e into an r-vector. Kernel does "
        "u = sum_e π_e u_e then y = U @ u, or y_e = U @ u_e if U is only "
        "partly shared. U is bound once. Never store or form W_down,e.",
        "Kill if a shared U at the r that saves bytes cannot match "
        "per-expert expansions on a planted sandwich; then on one layer "
        "down_proj at a later stage. Kill if combine-in-latent disagrees "
        "with expand-then-sum by more than the combiner's linearity allows "
        "(it must be identical when U is fully shared).",
        "Amortizes tall down expansion. Magnitude UNKNOWN.",
        "One expansion GEMV instead of top-k expansions. ns UNKNOWN.",
        "If experts need distinct output directions, a shared U smears "
        "them. Combining in latent is algebraically exact only when U is "
        "shared; a hybrid (clustered U) must not pretend otherwise.",
        "Not a trivial shared basis (output-side only, combine-in-latent "
        "is part of the object). Not mean-expert subtraction.",
    ),
    _store(
        "cross_layer_expert_prediction",
        "STORE-CROSS-LAYER-EXPERT-PREDICTION",
        "cross-layer expert prediction: predict layer L+1's expert factors "
        "from layer L's (same index or learned map); store only the residual "
        "of that prediction. Weights are not tied.",
        "Same-index WEIGHT tying is dead (control at 1e-7). Predicting a "
        "skinny core or code, and storing the residual, is a delta-code of "
        "factors, not a tie of W. Adjacent layers see related residual "
        "streams even when W cosine is ~0.",
        "Bind predictors P that map C_e,L -> C_e,L+1_hat as a small linear "
        "map on factors. Token path uses the residual-corrected factors "
        "already in the same skinny buffers. No dense W_L or W_{L+1}.",
        "Kill if residual energy of predicted factors is ~1 (then prediction "
        "is a no-op, matching the tying control). The test is on factors, "
        "not on flattened W. Do not reopen same-index W tying.",
        "Stores residual of predicted factors, not a second full bank. "
        "Win only if the residual is small. Magnitude UNKNOWN.",
        "Token path unchanged if residuals are fused at bind. ns UNKNOWN.",
        "A bad predictor adds a residual as large as the original factor "
        "(byte loss). Do not confuse a small W-cosine with a small factor "
        "residual; they are different measurements.",
        "Not same-index cross-layer weight tying (atlas / NNS-024). The "
        "stored object is a factor residual, not tied W.",
    ),
    _store(
        "conditional_residuals",
        "STORE-CONDITIONAL-RESIDUAL",
        "conditional residuals: y = shared_op(x) + residual_e(x), where "
        "shared_op is a learned or architecture-native shared operator "
        "(not the mean expert) and residual_e is skinny / sparse / gated.",
        "Subtracting the mean expert leaves residual norm 0.984 "
        "(EXPERT_FAMILY_GENOME noetic REFUTED). A learned or native shared "
        "op (Flash/Qwen shared expert, or a fitted shared SwiGLU) can still "
        "leave a small residual even when the mean is useless. SB4 in the "
        "shared-basis packet is this idea for models that already have a "
        "shared expert; it was not fitted here.",
        "Compute y_shared = shared_op(x) once. Selected experts add "
        "R_e @ z (skinny) or a sparse residual. Sum. Never form "
        "W_e = W_shared + R_e as a dense matrix.",
        "Kill if ||residual_e|| / ||expert_e|| is ~1 on the named shared_op "
        "(mean-expert already showed this). The falsifier must name the "
        "shared_op; 'the mean' is not an allowed shared_op. Do not merge "
        "omitted experts from residuals (NNS-023).",
        "Shared op stored once; per-expert payload is the residual. Win "
        "iff residual is small. Magnitude UNKNOWN.",
        "One shared_op plus top-k skinny residuals. ns UNKNOWN.",
        "If shared_op is the mean, this IS the dead noetic hypothesis. If "
        "residual is dense, bytes do not move. Capability dies when the "
        "residual cannot carry the expert's distinct function.",
        "Not shared-mean subtraction (explicitly forbidden as shared_op). "
        "Not expert merging. Not an unchanged archetype.",
    ),
    _store(
        "capability_sensitive_expert_islands",
        "STORE-CAPABILITY-ISLANDS",
        "capability-sensitive expert islands: partition the bank by "
        "capability / organ sensitivity (gate vs down, layer-band, tool "
        "vs language), and share structure only inside an island. Islands "
        "may use different mechanisms.",
        "NS-uniform-subbit / NNS-018: organs do not fail together; a "
        "single family across gate/up/down is INSUFFICIENT. Sharing that "
        "ignores the split will re-kill on the sensitive organ. Islands "
        "make that split the partition, so sharing is only attempted where "
        "the organ is tolerant.",
        "Tag experts/organs with an island id. Each island binds its own "
        "shared factors and native path (e.g. down island uses shared U, "
        "gate island stays independent or uses a milder skinny). The kernel "
        "dispatches per island; it does not densify a mixed W.",
        "Kill if island labels derived from a capability screen do not "
        "reduce per-island joint-SVD null-gap vs the global null (then "
        "the partition is cosmetic). Kill if the partition is used to "
        "justify a single codec family across organs (NNS-018).",
        "Sharing confined to tolerant islands; sensitive islands may store "
        "independent. Net byte effect UNKNOWN and possibly small.",
        "Per-island native paths; extra dispatch topology. ns UNKNOWN and "
        "possibly worse if islands fragment the wave.",
        "Wrong island labels mix a sensitive expert into a tolerant codec "
        "and fail capability first. Uniform sub-bit across islands is the "
        "dead lever this exists to avoid.",
        "Not raw global similarity (partitioned on purpose). Not a single "
        "unconditioned shared basis across the bank.",
    ),
)


# ---------------------------------------------------------------------------
# Compute-sharing candidates. Distinct axis: work repeated only because
# source matrices were treated as independent. Not a storage encoding.
# ---------------------------------------------------------------------------

def _compute(
    kind: str,
    cid: str,
    mechanism: str,
    why: str,
    native: str,
    falsifier: str,
    byte_effect: str,
    compute_effect: str,
    risk: str,
    scar_distance: str,
    repeated: str,
    why_repeated: str,
) -> dict[str, Any]:
    row = _store(
        kind, cid, mechanism, why, native, falsifier,
        byte_effect, compute_effect, risk, scar_distance,
    )
    row["axis"] = "COMPUTE"
    row["repeated_computation"] = repeated
    row["why_currently_repeated"] = why_repeated
    return row


COMPUTE_CANDIDATES: tuple[dict[str, Any], ...] = (
    _compute(
        "one_hidden_vector_many_experts",
        "COMPUTE-ONE-X-MANY-EXPERTS",
        "one hidden vector entering many experts: keep x register/SRAM "
        "resident and stream selected expert rows against it in one wave, "
        "fused with SiLU(gate)*up. Not a stacked dense table of independent W_e.",
        "Decode routes one token to k experts and every one multiplies THE "
        "SAME activation. ACCELERATOR_EXPERT_BATCH already showed that "
        "stacking independent W_e against shared x IS a taller matvec "
        "(exact 0.0 residual) and that at realistic decode top-k the "
        "representation-native stacked path is at best marginal. The "
        "untested remainder is fused register-resident x with packed "
        "payloads and organ fusion, without forming a stacked dense table.",
        "Load x once into threadgroup memory. Each threadgroup owns an "
        "intermediate row of a selected expert's packed gate/up and dots "
        "against resident x. Packed values are consumed in-register. No "
        "dense W_e, no stacked dense table.",
        "Do not rerun the B-sweep of ACCELERATOR_EXPERT_BATCH as if it "
        "were new. Falsify the FUSED remainder: a kernel that keeps x "
        "resident and reads packed rows vs the existing per-expert "
        "qwen80_routed_expert_wave_gate_up dispatches, on the same packed "
        "payload, under a protected lease (later lane). Kill if the fused "
        "wave matches the independent-dispatch complete-token within noise.",
        "None by itself (payloads unchanged). Byte effect is UNKNOWN/none "
        "unless packed layout also changes.",
        "Removes k independent loads of x and k separate gate/up launches. "
        "Magnitude UNKNOWN; stacked-dense cousin was marginal at decode "
        "top-k in ACCELERATOR_EXPERT_BATCH.",
        "Prefill / multi-token (different x per row) does not collapse and "
        "needs a different kernel. Occupancy vs bytes is the live wall, "
        "not gather-vs-sequential (NS-032).",
        "Not naive expert-table batching (already measured). Not a trivial "
        "shared basis (no shared W).",
        "k independent (gate_proj x) and (up_proj x) against the same x, "
        "plus k independent loads of x.",
        "Source stores one matrix per expert and the runtime honors that "
        "independence with one dispatch family per expert, even though the "
        "activation is identical.",
    ),
    _compute(
        "shared_xb_then_skinny",
        "COMPUTE-SHARED-XB-THEN-SKINNY",
        "shared xB transform with small expert-specific transforms: z = x @ B "
        "once; y_e = z @ C_e. Compute dual of shared input / common right.",
        "If a shared B exists (activation-PCA or joint right factor), the "
        "repeated full matvec is almost entirely duplicated work on x. "
        "This is the execution of STORE-SHARED-INPUT-TRANSFORM / "
        "STORE-COMMON-RIGHT-SUBSPACE, listed on the compute axis because "
        "the win can be FLOPs even when stored bytes barely move.",
        "One GEMV z = B @ x in SRAM. Selected C_e stream against z. No "
        "W_e = C_e B. Gate and up may share the same z.",
        "Same cheapest falsifier as shared input transforms (uncoded V, "
        "real X). Kill on function before claiming a compute win. A "
        "protected complete-token comparison is a later lane.",
        "Follows the paired storage candidate; may be ~0 if B is stored "
        "in addition to W. UNKNOWN.",
        "Replaces k full matvecs with 1 wide + k skinny. Algebraic "
        "reduction only; ns UNKNOWN.",
        "If B does not capture XX^T, quality dies before compute pays. "
        "Skinny maps that are not actually skinny (r ~ d_in) are a compute "
        "loss (extra round trip).",
        "Not trivial shared basis (B is a transform of x, not a shared W). "
        "Not ACCELERATOR_EXPERT_BATCH stacking of independent W_e.",
        "Each selected expert applies its own full input projection to x.",
        "Source matrices B_e are independent, so the compiler has no "
        "license to hoist a common transform of x.",
    ),
    _compute(
        "latent_weighted_reduction",
        "COMPUTE-LATENT-WEIGHTED-REDUCTION",
        "latent weighted reduction: apply router weights in a small latent "
        "and reduce there, instead of weighting full d_model outputs.",
        "The combiner is linear. If experts emit aligned latents, "
        "sum_e π_e (U u_e) = U (sum_e π_e u_e). The reduction moves before "
        "the expansion. This is free algebra when U is shared; it is the "
        "compute reason to want STORE-SHARED-OUTPUT-LATENT.",
        "Each selected expert writes an r-vector. Weighted sum in r. One "
        "U @ u. Exact when U is shared. No dense down_proj.",
        "Identity check: shared-U expand-then-sum vs reduce-then-expand "
        "must match at 0.0 residual (planted). Kill the candidate if a "
        "real organ requires per-expert U to meet the function screen "
        "(then the identity does not apply).",
        "None extra; depends on storing shared U. UNKNOWN.",
        "Cuts k-1 expansions of size d_model. ns UNKNOWN.",
        "Forcing a shared U so this identity applies can be a capability "
        "regression. Do not apply the identity when U is not shared.",
        "Not expert merging (all selected experts still run). Not mean "
        "subtraction.",
        "k independent down_proj expansions to d_model, then a weighted sum "
        "in d_model.",
        "Source down_proj is per-expert to full hidden, so the combiner is "
        "written after expansion.",
    ),
    _compute(
        "one_output_expansion",
        "COMPUTE-ONE-OUTPUT-EXPANSION",
        "one output expansion: a single shared expansion of a reduced "
        "intermediate, rather than k independent down_proj GEMVs.",
        "Same algebra as latent weighted reduction, named separately "
        "because the repeated work is the expansion itself even when "
        "weights are applied after. Shared U is the storage partner.",
        "Bind one U. Consume the reduced latent. One GEMV to d_model. "
        "Never k packed down_proj payloads applied independently unless "
        "the function screen forbids sharing U.",
        "Planted shared-U identity (must be exact). Then one-layer down "
        "function screen vs per-expert U. Kill if per-expert U is required.",
        "k down payloads collapse to one U plus skinny latents if storage "
        "follows. UNKNOWN.",
        "One expansion GEMV. ns UNKNOWN.",
        "down_proj is tolerant relative to gate, not invulnerable. A "
        "single U can smear expert-specific output directions.",
        "Not a trivial shared basis of whole W. Not same-index tying.",
        "k independent (down_proj @ intermediate_e) expansions.",
        "Source has one down_proj per expert; the runtime launches them "
        "as independent organs.",
    ),
    _compute(
        "shared_representation_decode",
        "COMPUTE-SHARED-REPRESENTATION-DECODE",
        "shared representation decode: a tiny shared codebook / scale "
        "family lives in registers; per-expert payloads are indices "
        "consumed fused with the matvec. Decode is not a per-expert "
        "prologue that writes dense W.",
        "Independent per-expert unpack + scale is repeated work on the "
        "same code family. Atlas large-LUT and entropy-coded Lloyd "
        "indices are dead; tiny lookup-free shared CB is the remaining "
        "decode-sharing object. NS-012 / NNS-018 forbid assuming one "
        "family across organs, so codebooks stay organ-tagged.",
        "Codebook in registers (K tiny). Fused FMA of codebook entries "
        "with x. No densify-then-GEMM. Organ-specific codebooks allowed.",
        "Kill if K is large enough to become a gather LUT (dead lever: "
        "learned codebook LUT). Kill if the kernel writes or caches a "
        "dense W (NS-018: caching decoded W restores the footprint the "
        "representation exists to remove). Kill if the codebook is "
        "raw-weight PQ at ~1 BPW (dead family). A planted tiny shared "
        "CB plus fused FMA must beat per-expert unpack on a synthetic "
        "packed payload before any specimen is touched.",
        "Codebook amortized; payloads are indices. Magnitude UNKNOWN. "
        "Raw-weight 1-bit PQ is dead; this does not reopen it.",
        "Removes per-expert unpack prologues. ns UNKNOWN. Reconstruction "
        "penalty is vehicle-specific (NNS-011); do not cite a measured "
        "factor here.",
        "Large codebook gathers are dead. A shared decode that is the "
        "raw-PQ family at one bit will collapse on a real forward. Organ "
        "split still applies.",
        "Not NS-010 unconditioned shared codebook on raw routed-expert "
        "weights. Not entropy-coded Lloyd indices. Not NS-018 "
        "cache-decoded-W. Not NNS-018 single-family-across-organs "
        "(organ-specific tiny CBs are allowed).",
        "Per-expert dequant / unpack / scale application before or during "
        "each expert matvec.",
        "Source payloads were packed independently, so decode is written "
        "as an expert-local prologue rather than a bank-level codebook "
        "in registers.",
    ),
    _compute(
        "shared_projections_across_organs",
        "COMPUTE-SHARED-PROJECTIONS-ACROSS-ORGANS",
        "shared projections across organs: hoist a common transform of x "
        "that both gate and up (and possibly the shared expert) can reuse, "
        "instead of two independent input projections per expert.",
        "Per-expert fused gate-up already exists "
        "(qwen80_routed_expert_wave_gate_up dots gate and up against x in "
        "one kernel). That is fusion, not sharing: the two matrices are "
        "still independent. The untested share is a common right factor "
        "across gate and up, so z = V @ x serves both organs.",
        "One V @ x. Skinny gate core and skinny up core against z, fused "
        "SiLU product. Independent per-organ paths remain legal for the "
        "sensitive organ (gate). No densified W_gate or W_up.",
        "Kill if a shared V for gate+up fails the function screen on gate "
        "(dominant_failure_organ historically). The fused-but-unshared "
        "kernel is the control, not a missing fusion.",
        "May cut stored input factors if V is shared; UNKNOWN.",
        "One shared input GEMV for two organs instead of two. ns UNKNOWN.",
        "NNS-018 / uniform-subbit: organs do not fail together. Sharing "
        "V across gate and up can kill gate to save a projection that "
        "down never needed. Islands may have to keep gate independent.",
        "Not already-built fused gate-up (that still uses two matrices). "
        "Not a trivial shared basis of the whole expert.",
        "gate_proj @ x and up_proj @ x as two full projections, even when "
        "fused in one kernel.",
        "Source has two independent matrices per expert; fusion multiplies "
        "both by x but does not hoist a shared factor.",
    ),
    _compute(
        "cross_layer_reused_transforms",
        "COMPUTE-CROSS-LAYER-REUSED-TRANSFORM",
        "cross-layer reused transforms: reuse a transform of the residual "
        "stream (V_L @ x_L) as an initialization for layer L+1, with a "
        "small adapter, instead of independently projecting at every layer.",
        "Layers are independent modules in source, but x_{L+1} is a "
        "residual update of x_L, so V @ x_{L+1} ≈ V @ x_L + V @ Δ. Same-"
        "index WEIGHT tying is dead; reusing the TRANSFORM of x is a "
        "compute hoist, not a W template. SB9 in the shared-basis packet "
        "was table-trim of V across a block, not this compute reuse.",
        "Keep z_L = V @ x_L. At L+1 compute z_{L+1} = z_L + V @ Δ (or a "
        "small adapter on z_L). Token path uses z, never W. V may be "
        "per-block (storage SB9) or per-layer with reuse of the product.",
        "Kill if ||V @ x_{L+1} - adapter(z_L)|| is not smaller than "
        "||V @ x_{L+1}|| on a real residual stream (then reuse is a no-op). "
        "Do not score by W cosine across layers (dead tying metric).",
        "May let V be stored per-block instead of per-layer. UNKNOWN.",
        "Avoids a full V @ x at L+1 when Δ is cheap. ns UNKNOWN.",
        "A large residual update (attention + MLP) can make Δ as expensive "
        "as x, so reuse does not pay. Do not reopen weight tying to make "
        "the adapter look good.",
        "Not same-index cross-layer weight tying. Not an unchanged "
        "archetype copied across layers.",
        "Independent V_L @ x_L at every layer, including when x_L is a "
        "small residual update of x_{L-1}.",
        "Source layers are separate tensors with no residual-product "
        "identity expressed in the graph, so each layer projects x from "
        "scratch.",
    ),
)


# ---------------------------------------------------------------------------
# Scar consultation and refusal
# ---------------------------------------------------------------------------

def _blob(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("id") or ""),
        str(candidate.get("mechanism") or ""),
        str(candidate.get("family") or ""),
        str(candidate.get("dead_family") or ""),
        str(candidate.get("title") or ""),
        " ".join(str(a) for a in (candidate.get("aliases") or ())),
    ]
    return " ".join(parts).lower()


def _on_disk(rel: str) -> bool:
    return (REPO / rel).is_file()


def consult_scar_sources() -> dict[str, Any]:
    """Locate the index if a sibling lane wrote it; else atlas files on disk.

    Does not import a sibling module. Missing files in a sparse checkout
    are recorded, not treated as proof they do not exist in git.
    """
    index_present = INDEX_PATH.is_file()
    atlas = []
    for rel in ATLAS_PATHS:
        atlas.append({"path": rel, "on_disk": _on_disk(rel), "role": "scar_source"})
    named_flash = []
    for rel in NAMED_FLASH_ABSENT:
        named_flash.append({"path": rel, "on_disk": _on_disk(rel)})
    return {
        "index_path": "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
        "index_present": index_present,
        "consult_rule": (
            "If NEGATIVE_SCIENCE_INDEX.json is present, extra DEAD/REFUTED "
            "phrases from it are unioned into the matcher. Else atlas files "
            "are read when on disk. Builtin scars always apply so a sparse "
            "checkout still refuses the three named dead families."
        ),
        "atlas": atlas,
        "named_flash_tools": named_flash,
    }


def _index_extra_scars(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = doc.get("entries") or doc.get("scars") or doc.get("killed_hypotheses") or []
    if isinstance(rows, dict):
        rows = [
            {"id": k, **v} if isinstance(v, dict) else {"id": str(k), "mechanism": str(v)}
            for k, v in rows.items()
        ]
    extra: list[dict[str, Any]] = []
    dead_verdicts = {"DEAD", "REFUTED", "KILLED", "REFUSE", "PROPERTY_OF_IDEA"}
    for e in rows:
        if not isinstance(e, dict):
            continue
        verdict = str(e.get("verdict") or e.get("class") or e.get("status") or "").upper()
        family = str(e.get("family") or "").upper()
        phrases = tuple(e.get("phrases") or ())
        mech = str(e.get("mechanism") or e.get("claim_refuted") or e.get("title") or "")
        if family in DEAD_FAMILIES or any(
            p.lower() in mech.lower() for scar in BUILTIN_SCARS for p in scar["phrases"]
        ) or verdict in dead_verdicts and phrases:
            extra.append(
                {
                    "id": str(e.get("id") or f"INDEX-{len(extra)}"),
                    "family": family or "INDEX",
                    "title": mech or str(e.get("id")),
                    "phrases": phrases or ((mech.lower(),) if mech else ()),
                    "why_dead": str(e.get("why_dead") or e.get("why_it_failed") or verdict),
                    "source": "NEGATIVE_SCIENCE_INDEX",
                }
            )
    extra.sort(key=lambda r: r["id"])
    return extra


def load_extra_scars() -> list[dict[str, Any]]:
    if not INDEX_PATH.is_file():
        return []
    try:
        doc = load_json(INDEX_PATH)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(doc, dict):
        return []
    return _index_extra_scars(doc)


def all_scars(extra_scars: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = [dict(s) for s in BUILTIN_SCARS]
    seen = {s["id"] for s in rows}
    for s in list(extra_scars or []) + load_extra_scars():
        if s["id"] not in seen:
            rows.append(s)
            seen.add(s["id"])
    rows.sort(key=lambda r: r["id"])
    return rows


def match_scar(
    candidate: dict[str, Any],
    extra_scars: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the first matching recorded-dead scar, else None.

    Targeted: a live structured cousin must not match. Matching is on
    family tags and long phrases, not on the words 'shared' or 'expert'.
    """
    blob = _blob(candidate)
    fam = str(candidate.get("family") or candidate.get("dead_family") or "").upper()
    for scar in all_scars(extra_scars):
        scar_fam = str(scar.get("family") or "").upper()
        if fam and fam in DEAD_FAMILIES and fam == scar_fam:
            return scar
        for ph in scar.get("phrases") or ():
            if ph and ph.lower() in blob:
                return scar
    return None


def _required_fields(candidate: dict[str, Any]) -> tuple[str, ...]:
    if candidate.get("axis") == "COMPUTE":
        return COMPUTE_FIELDS
    return STORAGE_FIELDS


def admit_candidate(
    candidate: dict[str, Any],
    extra_scars: list[dict[str, Any]] | None = None,
    require_schema: bool = True,
) -> dict[str, Any]:
    """Admit a live candidate, or raise DeadHypothesisError.

    Scar matching runs BEFORE schema checks so a minimal dead probe still
    fires the guard.
    """
    scar = match_scar(candidate, extra_scars=extra_scars)
    if scar is not None:
        raise DeadHypothesisError(str(candidate.get("id") or "<no-id>"), scar)
    if require_schema:
        missing = [f for f in _required_fields(candidate) if f not in candidate]
        if missing:
            raise CandidateSchemaError(
                f"{candidate.get('id')}: missing fields {missing}"
            )
        if candidate.get("forbids_dense_rematerialization") is not True:
            raise CandidateSchemaError(
                f"{candidate.get('id')}: native path must forbid dense rematerialization"
            )
        if candidate.get("evidence_class") != "STATIC_ONLY":
            raise CandidateSchemaError(
                f"{candidate.get('id')}: evidence_class must be STATIC_ONLY"
            )
        if candidate.get("status") != "HYPOTHESIS_UNFITTED":
            raise CandidateSchemaError(
                f"{candidate.get('id')}: this lane does not fit weights"
            )
    return dict(candidate)


def generate_storage(
    extra_scars: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [admit_candidate(c, extra_scars=extra_scars) for c in STORAGE_CANDIDATES]


def generate_compute(
    extra_scars: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [admit_candidate(c, extra_scars=extra_scars) for c in COMPUTE_CANDIDATES]


def generate(
    extra_scars: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = generate_storage(extra_scars) + generate_compute(extra_scars)
    rows.sort(key=lambda r: r["id"])
    return rows


DEAD_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "PROBE-RAW-GLOBAL-SIMILARITY",
        "mechanism": "raw global expert similarity",
        "family": DEAD_FAMILY_RAW,
    },
    {
        "id": "PROBE-TRIVIAL-SHARED-BASIS",
        "mechanism": "trivial shared basis",
        "family": DEAD_FAMILY_BASIS,
    },
    {
        "id": "PROBE-UNCHANGED-ARCHETYPE",
        "mechanism": "unchanged archetype",
        "family": DEAD_FAMILY_ARCHETYPE,
    },
)


def refusal_controls(
    extra_scars: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Watch the guard fail on the three named dead families."""
    rows = []
    for probe in DEAD_PROBES:
        try:
            admit_candidate(probe, extra_scars=extra_scars, require_schema=False)
        except DeadHypothesisError as exc:
            rows.append(
                {
                    "probe_id": probe["id"],
                    "refused": True,
                    "scar_id": exc.scar["id"],
                    "scar_family": exc.scar.get("family"),
                    "reason": str(exc),
                }
            )
            continue
        rows.append(
            {
                "probe_id": probe["id"],
                "refused": False,
                "error": "GUARD_FAILED_TO_FIRE",
            }
        )
    return rows


def assert_guards_fire(extra_scars: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = refusal_controls(extra_scars)
    failed = [r for r in rows if not r.get("refused")]
    if failed:
        raise RuntimeError(f"scar refusal did not fire: {failed}")
    return rows


def _recover_qwen80() -> dict[str, Any]:
    rel = "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json"
    p = REPO / rel
    if not p.is_file():
        return {"path": rel, "on_disk": False}
    d = load_json(p)
    gate = d.get("components", {}).get("gate_proj", {})
    up = d.get("components", {}).get("up_proj", {})
    return {
        "path": rel,
        "on_disk": True,
        "layer": d.get("layer"),
        "n_experts": d.get("n_experts"),
        "gate_pairwise_cosine_mean": gate.get("pairwise_cosine_mean"),
        "up_pairwise_cosine_mean": up.get("pairwise_cosine_mean"),
        "gate_subspace_overlap_top32": gate.get("subspace_overlap_top32"),
        "up_subspace_overlap_top32": up.get("subspace_overlap_top32"),
        "gate_k90": gate.get("k90"),
        "reading": (
            "Experts in this slice are mutually near-orthogonal as flattened "
            "W; a rank-32 shared weight subspace is not present. This kills "
            "raw global similarity, not one-sided / clustered / activation-"
            "space cousins."
        ),
    }


def recovered_implementation(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        "note": (
            "Naive expert sharing is already dead science. This module does "
            "not rerun it. Existing measurement tools fit specimens; this "
            "module generates unfitted structured candidates plus a scar "
            "guard. The named Flash doctor / route-conditioned / archetype "
            "screens are not in this tree."
        ),
        "qwen80_cross_expert_negative": _recover_qwen80(),
        "measurement_tools": [
            {
                "path": "lab/operators/q80_cross_expert_structure.py",
                "role": "produced the Q80 pairwise cosine / subspace-overlap negative",
            },
            {
                "path": "tools/odyssey/expert_family_genome.py",
                "role": "noetic hypothesis Expert_i ~= shared mean + delta; REFUTED (residual 0.984)",
                "receipt": "receipts/headless/EXPERT_FAMILY_GENOME.json",
            },
            {
                "path": "tools/headless/negative_science.py",
                "role": "failure store; no keyed retrieval (frontier F009)",
                "receipt": "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            },
            {
                "path": "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
                "role": "inter_expert_redundancy, cross_expert_and_cross_layer_tying, expert_merging DEAD; layer_zero LIVE; kronecker DEAD at depth LIVE at L0",
            },
            {
                "path": "tools/headless/c1sharedbasis_design.py",
                "role": "sharing lost on FIDELITY; NOT_WORTH_BUILDING as trivial shared basis",
                "receipt": "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            },
            {
                "path": "tools/headless/shared_basis_coherent.py",
                "role": "no coherent shared-basis point beats q2f on density and ns",
                "receipt": "receipts/headless/SHARED_BASIS_COHERENT.json",
            },
            {
                "path": "tools/headless/shared_basis_kernel.py",
                "role": "competent native kernel for shared-binary-basis MLP (dense, not MoE bank)",
                "receipt": "receipts/headless/SHARED_BASIS_KERNEL.json",
            },
            {
                "path": "hcli/agentos/flash_router_representation_ab.py",
                "role": "bounded Flash shared-basis vs independent q4 slice; shared was larger",
                "receipt": "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json",
            },
            {
                "path": "workspace/campaign/odyssey/evidence/sub1/shared-basis.md",
                "role": "SB1-SB9 proposals that do not assume expert redundancy; not a generator",
            },
            {
                "path": "receipts/headless/ACCELERATOR_EXPERT_BATCH.json",
                "role": "shared-x stacked independent W_e is a taller matvec; decode top-k marginal",
            },
            {
                "path": "crates/hawking-core/shaders/qwen80_routed_expert_wave.metal",
                "role": "current native path: per-expert packed GEMV, independent matrices",
            },
        ],
        "named_paths_absent_from_this_worktree": [
            r["path"] for r in sources["named_flash_tools"] if not r["on_disk"]
        ],
        "not_duplicating": (
            "expert_family_genome.py and q80_cross_expert_structure.py MEASURE "
            "a specimen. shared-basis.md PROPOSES mechanisms in prose. "
            "Neither emits a sealed candidate catalog with compute-sharing "
            "as a first-class axis and a watched-fail scar guard. This "
            "module extends that science; it does not fork a second genome."
        ),
        "scar_sources": sources,
    }


def gaps_closed() -> list[str]:
    return [
        "queryable generator of structured (not naive) expert-bank STORAGE hypotheses, each with a native consume path that forbids densifying W",
        "first-class COMPUTE-sharing axis: repeated work that exists only because source matrices were treated as independent",
        "scar refusal for raw global similarity, trivial shared basis, and unchanged archetypes that actually fires (watched-fail probes in the receipt)",
        "sealed STATIC_ONLY receipt with recovered implementation, gaps, and negative findings",
        "explicit scar_distance on every candidate so structured cousins are not mistaken for the dead families",
    ]


def negative_findings(sources: dict[str, Any]) -> list[str]:
    findings = [
        "Did not fit any candidate to real weights (specimens are out of scope for this lane).",
        "Did not take a hardware measurement; every expected effect is qualitative or algebraic. Complete-token ns, tps, and joules remain UNKNOWN.",
        "receipts/future/NEGATIVE_SCIENCE_INDEX.json is not present; builtin scars plus on-disk atlas probes were used. A sibling index will be consulted if it appears.",
        "DSV4F pairwise cosine is still unmeasured (NNS-004 live check). Q80 orthogonality is not transferred.",
        "tools/accelerator/, tools/headless/, and hcli/agentos/ are not materialized in this sparse checkout; they were recovered via git show during authoring.",
    ]
    missing_flash = [
        r["path"] for r in sources["named_flash_tools"] if not r["on_disk"]
    ]
    if missing_flash:
        findings.append(
            "Named Flash doctor / route-conditioned / archetype / bank-screen "
            "paths are absent from this worktree (and from HEAD): "
            + ", ".join(missing_flash)
        )
    missing_atlas = [r["path"] for r in sources["atlas"] if not r["on_disk"]]
    if missing_atlas:
        findings.append(
            "Atlas files not on disk in this sparse checkout (recovered via "
            "git show during authoring, encoded as builtin scars): "
            + ", ".join(missing_atlas)
        )
    return findings


def build() -> Path:
    sources = consult_scar_sources()
    extra = load_extra_scars()
    controls = assert_guards_fire(extra)
    storage = generate_storage(extra)
    compute = generate_compute(extra)
    kinds_storage = [c["kind"] for c in storage]
    kinds_compute = [c["kind"] for c in compute]
    if tuple(kinds_storage) != REQUIRED_STORAGE_KINDS:
        raise RuntimeError(f"storage kinds drifted: {kinds_storage}")
    if tuple(kinds_compute) != REQUIRED_COMPUTE_KINDS:
        raise RuntimeError(f"compute kinds drifted: {kinds_compute}")
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Generate structured expert STORAGE-sharing and COMPUTE-sharing "
            "candidates that have not been tested, and refuse recorded-dead "
            "naive sharing (raw global similarity, trivial shared basis, "
            "unchanged archetypes). Candidates only; no specimen fit; no "
            "hardware claim."
        ),
        "era_vocabulary": {
            "eras": [
                "I Genesis of the Laboratory",
                "II Compounding Civilization",
                "III Autonomous Science Civilization",
                "IV Synthetic Machine Civilization",
                "V Released Hawking Civilization",
            ],
            "odysseys": [
                "I WHAT IS TRUE?",
                "II WHAT DID HAWKING ALREADY LEARN?",
                "III WHERE IS HAWKING WRONG?",
            ],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is_not_its_own_civilization": True,
        },
        "fit_policy": "NOT_FIT",
        "native_execution_law": (
            "A kernel must consume factors / packed payloads / latents "
            "without rematerializing dense expert weights. Densify-then-GEMM "
            "is a fake share (C1SHAREDBASIS_DESIGN separated that failure "
            "from fidelity; fidelity still killed the trivial basis)."
        ),
        "killed_hypotheses": [
            {
                "id": s["id"],
                "family": s["family"],
                "title": s["title"],
                "why_dead": s["why_dead"],
                "settled_by": list(s.get("settled_by") or ()),
                "reopen_condition": s.get("reopen_condition"),
                "scope": s.get("scope"),
            }
            for s in BUILTIN_SCARS
        ],
        "scar_sources_consulted": sources,
        "extra_scars_from_index": [s["id"] for s in extra],
        "refusal_controls": controls,
        "storage_candidates": storage,
        "compute_candidates": compute,
        "counts": {
            "storage": len(storage),
            "compute": len(compute),
            "killed_families": len(BUILTIN_SCARS),
            "refusal_controls_fired": sum(1 for r in controls if r.get("refused")),
            "extra_scars_from_index": len(extra),
        },
        "recovered_implementation": recovered_implementation(sources),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(sources),
    }
    return write_receipt(RECEIPT, doc, "tools/future/expert_bank_school.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
