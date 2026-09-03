"""NEGATIVE_SCIENCE_INDEX — query the scars before proposing.

Hawking already recorded a large corpus of things that do not work.
Nothing queried that corpus before a new experiment was proposed, so
rediscovery was free. This module is the keyed index and the refusal path.

It extends the existing stores (tools/headless/negative_science.py,
noetic_negative_science.py, foundry/doctor atlases, campaign JSONL) — it
does not restate them and it does not write onto the Codex surface.

    python3 tools/future/negative_index.py --build
    python3 tools/future/negative_index.py --query --model qwen3-235b-a22b --hypothesis-family cross_expert_structure
    python3 tools/future/negative_index.py --refuse '{"model":"qwen3-235b-a22b","hypothesis_family":"cross_expert_structure"}'

Public API (what a downstream generator calls):

    query(model=..., organ=..., representation=..., hypothesis_family=..., machine=...)
        -> list[dict]  ranked by specificity, each carrying source_path
    refuse_if_dead(proposal) -> dict | None
        proposal is a dict with the same keys (hypothesis_family may also
        arrive as technique / mechanism / lever). Returns a refusal citing
        the scar, or None. MODEL_SPECIFIC scars do not prune a different
        named parent.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from tools.future._common import HARDWARE_FIELDS, REPO, git, write_receipt

RECEIPT = "NEGATIVE_SCIENCE_INDEX.json"
SCHEMA = "hawking.future.negative_index.v1"
UNRECORDED = "unrecorded"
PARSED = "PARSED"
UNPARSED = "UNPARSED"

# Named corpus from the lane contract, plus the extra stores the recover
# sweep found. discover() unions this with a git ls-tree name scan.
SEED_SOURCES: tuple[str, ...] = (
    "tools/headless/negative_science.py",
    "tools/headless/noetic_negative_science.py",
    "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
    "receipts/ascent-2026-08-16/Q80_LM_HEAD_NEGATIVE.json",
    "receipts/ascent-2026-08-16/Q80_GK_SIMD_NEGATIVE.json",
    "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
    "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
    "workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
    "workspace/campaign/evidence/research/doctor/DOCTOR_NEGATIVE_TRANSFER_ATLAS.json",
    "research/hawking-experiments/superwave/g1/g1-arch-negative.md",
    "workspace/campaign/evidence/systems/hawking/HAWKING_EXPERT_WAVE_NEGATIVE.json",
    "workspace/campaign/evidence/systems/hawking/HAWKING_RESIDENT_STATE_NEGATIVE.json",
    "workspace/campaign/records/ascension-sandbox/physical/qwen-family/dual-gravity/ASCENSION_NEGATIVE_SCIENCE.jsonl",
    "workspace/campaign/records/ascension-sandbox/physical/qwen80/evolution/NEGATIVE_SCIENCE.jsonl",
    "workspace/docs/guides/dead_levers.md",
    "workspace/campaign/governance/odyssey/state/graveyard/GRAVEYARD.json",
    "receipts/ascent-2026-08-18/PHASE_B_HYBRID_REFUTED.json",
    "research/lab/operators/ascension_graveyard.py",
    # Emitted by tools/future/tps_falsifications.py. Named here because
    # SKIP_PREFIXES excludes receipts/future/ from the discovery sweep,
    # and SEED_SOURCES is appended after that filter.
    "receipts/future/TPS_FALSIFICATIONS.jsonl",
    # Emitted by tools/future/campaign_scars.py. Same reason as the
    # TPS_FALSIFICATIONS.jsonl row: SKIP_PREFIXES drops receipts/future/.
    "receipts/future/CAMPAIGN_SCARS.json",
    # SCIENCE scars landed by the 2026-08-31 Gravity wave. SKIP_PREFIXES excludes
    # receipts/future/ from the discovery sweep, so a scar landed there is
    # INVISIBLE to refuse_if_dead unless it is named here - and the whole point of
    # a scar is that the next proposer cannot rediscover it for free.
    #
    # This was not theoretical. The model-bearing torture (G015) failed with zero
    # launches because choose() advertised WU.DEAD.mlp_function_replacement as
    # policy: refuse_if_dead did not key MLP_FUNCTION_REPLACEMENT_CLOSED, so the
    # dead school was still on the menu 45 times running and the resident kept
    # picking it. Every school this wave closed is listed below.
    "receipts/future/MLP_STRUCTURED_OPERATOR.json",
    "receipts/future/MLP_SPARSE_RESIDUAL.json",
    "receipts/future/MLP_FUNCTIONAL_RANK.json",
    "receipts/future/MLP_NONLINEAR_PROGRAM.json",
    "receipts/future/MLP_SHARED_PROGRAM.json",
    "receipts/future/DELTANET_GENERATED_TRANSITION.json",
    "receipts/future/AUX_CAPABILITY_SCREEN.json",
    "receipts/future/AUX_U8_NATIVE.json",
    "receipts/future/AUX_U8_LUT.json",
    "receipts/future/MLP_STREAM_COUNT.json",
    "receipts/future/MLP_ISSUE_RATE_LADDER.json",
    # S025 §16-17. Not a representation school - a COST MODEL scar, and the one
    # that priced two of the schools above. Listed here so refuse_if_dead can
    # refuse the next candidate whose whole case is a byte count.
    "receipts/future/ECONOMICS_CALIBRATION.json",
    # G014. Landed by tools/sovereign/g014_negative_science.py, same SKIP_PREFIXES
    # reason as every receipts/future/ row above: discovery does not sweep it, so
    # without this line refuse_if_dead cannot key these scars at all.
    "receipts/future/SOVEREIGN_NEGATIVE_SCIENCE.json",
)

SKIP_PREFIXES = ("tools/future/", "receipts/future/", "crates/")
NAME_TOKENS = ("negativ", "dead_lever", "graveyard", "refuted")

# Longer / more specific model patterns first.
MODEL_RULES: tuple[tuple[str, str], ...] = (
    (r"qwen3[-_. ]?235b", "qwen3-235b-a22b"),
    (r"qwen3[-_ ]?80b|\bq80\b|qwen80|qwen3_next_hybrid|qwen3[-_]next", "qwen3-80b"),
    (r"qwen3[-_ ]?30b|\bq30\b|qwen30|qwen3_moe|qwen3-coder-30b", "qwen3-30b-a3b"),
    (r"qwen3(?:\.8|[-_]?38)|qwen[-_]?38\b|qwen3\.8", "qwen3.8-27b"),
    (r"gpt[-_ ]?oss", "gpt-oss-120b"),
    (r"glm[-_ ]?5\.?2|\bglm52\b", "glm-5.2"),
    (r"deepseek[-_ ]?v4|\bdsv4f\b", "deepseek-v4-flash"),
    (r"qwen14", "qwen14"),
    (r"qwen-?3b\b", "qwen3b"),
)

ORGAN_SLUGS = {
    "gate_proj": "gate",
    "gate": "gate",
    "mlp_gate_up": "gate",
    "mlp1": "gate",
    "up_proj": "up",
    "up": "up",
    "down_proj": "down",
    "down": "down",
    "mlp_down": "down",
    "mlp2": "down",
    "mlp": "mlp",
    "mlp_gate_up_mlp_down": "mlp",
    "gqa_attention": "attention",
    "gqa_attention_q": "attention",
    "hybrid_attention_q": "attention",
    "attention": "attention",
    "attn": "attention",
    "qkv": "attention",
    "lm_head": "lm_head",
    "lmhead": "lm_head",
    "embed": "embed",
    "router": "router",
    "moe_router": "router",
    "routed_experts": "routed_experts",
    "routed_expert": "routed_experts",
    "routed_expert_gate": "gate",
    "shared_expert": "mlp",
    "experts": "routed_experts",
    "kv": "kv",
    "kv_state": "kv",
    "deltanet": "deltanet",
    "gated_deltanet_convolution": "deltanet",
    "gated_deltanet_projection": "deltanet",
    "whole_model": "whole_model",
    "whole_artifact": "whole_model",
    "whole": "whole_model",
}

FAMILY_SLUGS = {
    "cross_expert_and_cross_layer_tying": "cross_expert_structure",
    "cross_expert_structure": "cross_expert_structure",
    "cross_expert": "cross_expert_structure",
    "inter_expert_redundancy": "cross_expert_structure",
    "generated_tied_params": "cross_expert_structure",
    "shared_basis_across_experts": "cross_expert_structure",
    "trivial_global_expert_sharing": "cross_expert_structure",
    "global_expert_sharing": "cross_expert_structure",
    "expert_templates": "cross_expert_structure",
    "expert_templates_deltas": "cross_expert_structure",
    "ns_cross_expert_and_cross_layer_tying": "cross_expert_structure",
    "expert_merging_omitted_from_survivors": "expert_merge",
    "expert_merge": "expert_merge",
    "expert_merging": "expert_merge",
    "kronecker_factorisation": "kronecker",
    "kronecker_factorization": "kronecker",
    "kronecker": "kronecker",
    "raw_weight_pq_vq_at_one_bit": "raw_weight_pq_vq",
    "raw_weight_pq_vq": "raw_weight_pq_vq",
    "layer_zero_is_a_different_source": "layer0_separate_source",
    "ternary_factorization": "ternary",
    "ternary_threshold_group128": "ternary",
    "calibration_88_tokens": "calibration_88_tokens",
    "entropy_coded_pq_indices": "entropy_coded_pq",
    "large_expert_cache": "large_expert_cache",
    "post_hoc_coding_of_frozen_weights": "post_hoc_frozen_codec",
    "posthoc_scalar_gain": "posthoc_scalar_gain",
    "row_norm_stratification_premise": "row_norm_stratification",
    "uniform_subbit_allocation": "uniform_subbit_allocation",
    "binary_sign_scale128": "binary_quantization",
    "binary_outlier_residual": "binary_quantization",
    "binary_quantization": "binary_quantization",
    "qn_binary_injury": "binary_quantization",
    "qn_binary_healing": "binary_quantization",
    "qn_binary_as_draft": "binary_draft",
    "qn_shared_basis_density": "shared_basis",
    "qn_shared_k_hybrid": "shared_basis",
    "qn_lowrank_healing": "low_rank",
    "qn_coordinate_transform": "coordinate_transform",
    "qn_head_redundancy": "head_sharing",
    "qn_state_merging": "state_merging",
    "sub_bit_synthetic_then_real": "synthetic_activation",
    "synthetic_activation_mismatch": "synthetic_activation",
    "complete_bpw_predicts_coherence": "bpw_is_not_capability",
    "shared_basis_across_experts_seed": "cross_expert_structure",
    "student_distillation_closed": "router_distill",
    "router_distill": "router_distill",
    "low_rank_residual": "low_rank",
    "teacher_low_rank_q3": "low_rank",
    "ns_global_dense_lowrank_qwen38": "global_dense_lowrank",
    "expert_wave": "expert_wave",
    "fused_expert_wave": "expert_wave",
    "resident_state": "resident_state",
    "gpu_resident_state": "resident_state",
    "gk_simd": "gk_simd",
    "lm_head_below_q8": "lm_head_precision",
    "uniform_q2_group64": "uniform_q2",
    "uniform_q3_group64": "uniform_q3",
    "uniform_q4_group64": "uniform_q4",
    "hadamard_lattice_q3_group128": "hadamard_lattice",
    "activation_corrected_rowwise_q3": "activation_corrected_q3",
    "additive_residual_codebook_q2x2": "residual_codebook",
    "packed_binary_simdgroup_template_parity": "binary_quantization",
}

MACHINE_SLUGS = {
    "m3 ultra 96gb / metal": "m3_ultra",
    "m3 ultra": "m3_ultra",
    "apple host cpu": "apple_host_cpu",
    "apple silicon": "apple_silicon",
    "metal": "metal",
}

DEAD_WORDS = frozenset(
    {
        "dead",
        "negative",
        "buried",
        "kills",
        "closed",
        "refuted",
        # A lane that kills its own hypothesis records FALSIFIED. That is the
        # project's most precise dead verdict and the index did not know it.
        "falsified",
        "unreachable",
        "deauthorised",
        "deauthorized",
        "nogo",
        "insufficient",
        "blocked",
        "failed",
        "incoherent",
        "rejected",
        "category_error",
        "categoryerror",
    }
)
LIVE_PHRASES = (
    "live and convergent",
    "not closed",
    "untested",
    "positive entry",
    "named exception",
)


_INDEX: list["Scar"] | None = None


def _slug(s: str) -> str:
    s = (s or "").strip().lower().replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _txt(v: Any, limit: int = 480) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        s = json.dumps(v, sort_keys=True, default=str)
    else:
        s = str(v)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def _pick(obj: dict[str, Any], *keys: str) -> str:
    for k in keys:
        if k in obj and obj[k] not in (None, "", [], {}):
            return _txt(obj[k])
    return ""


def extract_models(text: str) -> list[str]:
    blob = text or ""
    found: list[str] = []
    for pat, canon in MODEL_RULES:
        if re.search(pat, blob, flags=re.I) and canon not in found:
            found.append(canon)
    if found:
        return found
    # Never invent a model from a lever name or a prose sentence.
    return [UNRECORDED]


def canon_model(text: str) -> str:
    models = extract_models(text)
    return models[0] if models else UNRECORDED


def canon_organ(text: str) -> str:
    slug = _slug(text)
    if not slug:
        return UNRECORDED
    if slug in ORGAN_SLUGS:
        return ORGAN_SLUGS[slug]
    for key, canon in ORGAN_SLUGS.items():
        if key in slug or slug in key:
            return canon
    words = _words(text)
    if words & {"gate", "gateproj", "mlp1"}:
        return "gate"
    if words & {"attention", "attn", "gqa", "qkv"}:
        return "attention"
    if words & {"router"}:
        return "router"
    if words & {"lmhead", "lm"} and words & {"head"}:
        return "lm_head"
    return UNRECORDED


def extract_organs(text: str) -> list[str]:
    blob = text or ""
    if not blob.strip():
        return [UNRECORDED]
    found: list[str] = []
    for token in re.split(r"[,/;+\s]+|(?:\band\b)", blob):
        c = canon_organ(token)
        if c != UNRECORDED and c not in found:
            found.append(c)
    if found:
        return found
    c = canon_organ(blob)
    return [c]


def canon_machine(text: str) -> str:
    slug = _slug(text)
    if not slug:
        return UNRECORDED
    low = (text or "").strip().lower()
    if low in MACHINE_SLUGS:
        return MACHINE_SLUGS[low]
    if slug in MACHINE_SLUGS:
        return MACHINE_SLUGS[slug]
    if "m3" in slug and "ultra" in slug:
        return "m3_ultra"
    if "apple" in slug and "cpu" in slug:
        return "apple_host_cpu"
    if "metal" in slug:
        return "metal"
    if len(slug) > 48:
        return UNRECORDED
    return slug


def canon_family(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return UNRECORDED
    slug = _slug(raw)
    if slug in HARDWARE_FIELDS:
        slug = f"family_{slug}"
    if slug in FAMILY_SLUGS:
        return FAMILY_SLUGS[slug]
    if slug.startswith("binary"):
        return "binary_quantization"
    if "kronecker" in slug:
        return "kronecker"
    if "synthetic" in slug or "gaussian" in slug:
        return "synthetic_activation"
    if "expert_wave" in slug or slug.endswith("wave"):
        if "expert" in slug:
            return "expert_wave"
    if "resident" in slug and "state" in slug:
        return "resident_state"
    if "megakernel" in slug:
        return "megakernel"
    if "lm_head" in slug or slug.startswith("lmhead"):
        return "lm_head_precision"
    words = _words(raw)
    expertish = bool(words & {"expert", "experts"})
    if expertish and words & {"template", "templates", "tying", "sharing", "shared", "redundancy", "orthogonal"}:
        return "cross_expert_structure"
    if expertish and "cross" in words:
        return "cross_expert_structure"
    if "trivial" in words and expertish:
        return "cross_expert_structure"
    if {"global", "sharing"} <= words and expertish:
        return "cross_expert_structure"
    if "cross" in words and "layer" in words and words & {"tying", "tie", "tied"}:
        return "cross_expert_structure"
    if "shared" in words and "basis" in words and expertish:
        return "cross_expert_structure"
    if "shared" in words and "basis" in words:
        return "shared_basis"
    if words & {"merge", "merging"} and expertish:
        return "expert_merge"
    if "low" in words and "rank" in words:
        return "low_rank"
    if "speculative" in words or ("spec" in words and "decode" in words):
        return "spec_decode"
    if slug in HARDWARE_FIELDS:
        return f"family_{slug}"
    return slug


def canon_representation(text: str) -> str:
    slug = _slug(text)
    if not slug:
        return UNRECORDED
    if "binary" in slug or "1_bit" in slug or slug in {"onebit", "1bit"}:
        return "binary"
    if "kronecker" in slug:
        return "kronecker"
    if "shared_basis" in slug or ("shared" in slug and "basis" in slug):
        return "shared_basis"
    if "low_rank" in slug or "lora" in slug or "svd" in slug:
        return "low_rank"
    if slug.startswith("q8") or "bits_8" in slug:
        return "q8"
    if slug.startswith("q4"):
        return "q4"
    if slug.startswith("q3"):
        return "q3"
    if slug.startswith("q2"):
        return "q2"
    if "pq" in slug or "vq" in slug:
        return "pq_vq"
    if len(slug) > 64:
        return UNRECORDED
    return slug


def _verdict_eligible(verdict: str, status: str = "") -> bool:
    blob = f"{verdict} {status}".strip().lower()
    if not blob:
        return False
    if any(p in blob for p in LIVE_PHRASES):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", blob.replace("-", "")))
    # Type-1 dead, NO-GO, etc. collapse hyphens before the word set.
    collapsed = _slug(blob).replace("_", "")
    if "nogo" in collapsed or "type1" in collapsed:
        return True
    return bool(tokens & DEAD_WORDS)


@dataclass
class Scar:
    scar_id: str
    source_path: str
    source_origin: str
    parse_status: str
    unparsed_reason: str = ""
    model: str = UNRECORDED
    models: list[str] = field(default_factory=lambda: [UNRECORDED])
    organ: str = UNRECORDED
    organs: list[str] = field(default_factory=lambda: [UNRECORDED])
    representation: str = UNRECORDED
    machine: str = UNRECORDED
    hypothesis_family: str = UNRECORDED
    failure_mechanism: str = UNRECORDED
    verdict: str = UNRECORDED
    refuse_eligible: bool = False
    reopen_condition: str = UNRECORDED
    claim_refuted: str = UNRECORDED
    level: str = "MODEL_SPECIFIC"
    original_id: str = ""
    keys_filled: int = 0

    def finalize(self) -> "Scar":
        if not self.models:
            self.models = [self.model or UNRECORDED]
        if not self.organs:
            self.organs = [self.organ or UNRECORDED]
        self.model = self.models[0]
        self.organ = self.organs[0]
        self.models = sorted(set(self.models))
        self.organs = sorted(set(self.organs))
        filled = 0
        for name in (
            "model",
            "organ",
            "representation",
            "machine",
            "hypothesis_family",
            "failure_mechanism",
        ):
            v = getattr(self, name)
            if v and v != UNRECORDED:
                filled += 1
        self.keys_filled = filled
        if self.parse_status != PARSED:
            self.refuse_eligible = False
        if not self.original_id:
            self.original_id = self.scar_id
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.finalize())
        # Keep the receipt free of any hardware-field key, even if empty.
        for k in list(d):
            if k in HARDWARE_FIELDS:
                d.pop(k)
        return d


def _unparsed(rel: str, reason: str, origin: str) -> Scar:
    return Scar(
        scar_id=f"{rel}#UNPARSED",
        source_path=rel,
        source_origin=origin,
        parse_status=UNPARSED,
        unparsed_reason=reason,
        refuse_eligible=False,
        level="UNRECORDED",
    ).finalize()


def _scar(
    rel: str,
    origin: str,
    original_id: str,
    *,
    model_text: str = "",
    organ_text: str = "",
    representation_text: str = "",
    machine_text: str = "",
    family_text: str = "",
    mechanism: str = "",
    verdict: str = "",
    status: str = "",
    reopen: str = "",
    claim: str = "",
    level: str = "MODEL_SPECIFIC",
) -> Scar:
    models = extract_models(model_text) if model_text else [UNRECORDED]
    organs = extract_organs(organ_text) if organ_text else [UNRECORDED]
    family = canon_family(family_text or mechanism)
    mech = _txt(mechanism or family_text) or UNRECORDED
    verd = _txt(verdict or status) or UNRECORDED
    return Scar(
        scar_id=f"{rel}#{original_id}",
        source_path=rel,
        source_origin=origin,
        parse_status=PARSED,
        model=models[0],
        models=models,
        organ=organs[0],
        organs=organs,
        representation=canon_representation(representation_text),
        machine=canon_machine(machine_text),
        hypothesis_family=family,
        failure_mechanism=mech if mech else UNRECORDED,
        verdict=verd,
        refuse_eligible=_verdict_eligible(verd, status),
        reopen_condition=_txt(reopen) or UNRECORDED,
        claim_refuted=_txt(claim) or UNRECORDED,
        level=level if level in {"MODEL_SPECIFIC", "FAMILY", "GENERAL_PHYSICAL"} else "MODEL_SPECIFIC",
        original_id=str(original_id),
    ).finalize()


def read_text(rel: str) -> tuple[str | None, str]:
    """Disk first, then git show HEAD:<path> (sparse checkout is not absence)."""
    path = REPO / rel
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace"), "disk"
    except OSError:
        pass
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return r.stdout, "git"
    return None, "missing"


def discover_sources() -> list[str]:
    names = [ln for ln in git("ls-tree", "-r", "--name-only", "HEAD").splitlines() if ln]
    found: list[str] = []
    for n in names:
        if n.startswith(SKIP_PREFIXES):
            continue
        if "/test_" in f"/{n}" or n.endswith("/tests") or "/tests/" in n:
            continue
        low = n.lower()
        if any(tok in low for tok in NAME_TOKENS):
            found.append(n)
        elif "/negative-science/" in low:
            found.append(n)
    for s in SEED_SOURCES:
        if s not in found:
            found.append(s)
    # Lockfiles stay: they are unparseable scars, never dropped.
    return sorted(set(found))


def parse_jsonl(rel: str, text: str, origin: str) -> list[Scar]:
    out: list[Scar] = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            out.append(_unparsed(rel, f"line {i}: {e}", origin))
            continue
        if not isinstance(obj, dict):
            out.append(_unparsed(rel, f"line {i}: not an object", origin))
            continue
        rid = obj.get("record_id") or obj.get("id") or f"L{i}"
        geom = _pick(obj, "model_geometry") or _txt(obj.get("model"))
        mech = _pick(obj, "mechanism", "mechanism_key")
        out.append(
            _scar(
                rel,
                origin,
                str(rid),
                model_text=geom,
                organ_text=geom,
                family_text=mech,
                mechanism=mech,
                verdict=_pick(obj, "status", "verdict", "measured_outcome"),
                status=_pick(obj, "status"),
                reopen=_pick(obj, "reopen_condition", "reopen"),
                claim=_pick(obj, "failure_reason", "claim_boundary", "measured_outcome"),
            )
        )
    if not out:
        out.append(_unparsed(rel, "empty jsonl", origin))
    return out


def _parse_foundry(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    entries = obj.get("entries") or {}
    out = []
    if isinstance(entries, dict):
        items = sorted(entries.items())
        for key, rec in items:
            if not isinstance(rec, dict):
                out.append(_unparsed(rel, f"entry {key} not an object", origin))
                continue
            verd = _pick(rec, "verdict", "status")
            out.append(
                _scar(
                    rel,
                    origin,
                    key,
                    model_text=_pick(rec, "parent") or _pick(rec, "provenance") or "",
                    organ_text=_pick(rec, "organ", "lever"),
                    family_text=key + " " + _pick(rec, "lever"),
                    mechanism=_pick(rec, "lever") or key,
                    verdict=verd,
                    status=_pick(rec, "status"),
                    reopen=_pick(rec, "reopen_condition", "reopen"),
                    claim=_pick(rec, "killed_by", "provenance"),
                    representation_text=_pick(rec, "lever"),
                )
            )
    return out or [_unparsed(rel, "foundry atlas had no entries", origin)]


def _parse_register(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    machine = _txt((obj.get("machine") or {}).get("chip") if isinstance(obj.get("machine"), dict) else "")
    out = []
    for rec in obj.get("entries") or []:
        if not isinstance(rec, dict):
            continue
        models = rec.get("models") or []
        model_text = " ".join(str(m) for m in models) if isinstance(models, list) else _txt(models)
        oid = rec.get("id") or rec.get("mechanism") or "NS"
        out.append(
            _scar(
                rel,
                origin,
                str(oid),
                model_text=model_text,
                family_text=_pick(rec, "mechanism", "id"),
                mechanism=_pick(rec, "mechanism"),
                verdict=_pick(rec, "class"),
                status=_pick(rec, "class"),
                reopen=_pick(rec, "retry_when"),
                claim=_pick(rec, "why_it_failed", "what_was_expected"),
                machine_text=machine,
            )
        )
    return out or [_unparsed(rel, "register had no entries", origin)]


def _parse_noetic(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    out = []
    for rec in obj.get("entries") or []:
        if not isinstance(rec, dict):
            continue
        scope = rec.get("scope") if isinstance(rec.get("scope"), dict) else {}
        oid = rec.get("id") or rec.get("seed") or "NNS"
        kind = _pick(rec, "kind")
        reopen_today = rec.get("reopen_satisfied_today")
        verd = kind or "NEGATIVE"
        # PROPERTY_OF_IDEA stays refuse-eligible unless reopen already holds.
        eligible_override = None
        if reopen_today is True:
            eligible_override = False
            verd = "LIVE_REOPEN_HOLDS"
        scar = _scar(
            rel,
            origin,
            str(oid),
            model_text=_pick(scope, "model") or _pick(rec, "model"),
            organ_text=_pick(scope, "organ") or _pick(rec, "organ"),
            representation_text=_pick(scope, "codec", "regime") or _pick(rec, "representation"),
            family_text=_pick(rec, "seed", "id") + " " + _pick(rec, "claim_refuted"),
            mechanism=_pick(rec, "seed") or _pick(rec, "claim_refuted"),
            verdict=verd,
            status=kind,
            reopen=_pick(rec, "reopen_condition"),
            claim=_pick(rec, "claim_refuted", "kind_reasoning"),
        )
        if eligible_override is False:
            scar.refuse_eligible = False
        elif kind == "PROPERTY_OF_IDEA" and reopen_today is not True:
            scar.refuse_eligible = True
        elif kind == "ARTIFACT_OF_METHOD":
            # The idea is not dead; the method is. Do not blanket-refuse the idea.
            scar.refuse_eligible = False
        out.append(scar)
    return out or [_unparsed(rel, "noetic receipt had no entries", origin)]


def _parse_odyssey(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    out = []
    for rec in obj.get("entries") or []:
        if not isinstance(rec, dict):
            continue
        killed_on = rec.get("killed_on") or []
        model_text = " ".join(str(x) for x in killed_on) if isinstance(killed_on, list) else _txt(killed_on)
        oid = rec.get("id") or rec.get("mechanism") or "NS"
        out.append(
            _scar(
                rel,
                origin,
                str(oid),
                model_text=model_text or _pick(rec, "mechanism"),
                family_text=_pick(rec, "id", "mechanism"),
                mechanism=_pick(rec, "mechanism"),
                verdict=_pick(rec, "verdict"),
                reopen=_pick(rec, "reopen_if", "reopen_condition"),
                claim=_pick(rec, "killed_by", "premise"),
            )
        )
    return out or [_unparsed(rel, "odyssey store had no entries", origin)]


def _parse_campaign_scars(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    """Seven campaign scars: process/method defects keyed for refuse_if_dead."""
    rows = obj.get("scars") or obj.get("entries") or []
    out: list[Scar] = []
    if not isinstance(rows, list):
        return [_unparsed(rel, "campaign scars is not a list", origin)]
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        oid = rec.get("id") or rec.get("scar_id") or rec.get("hypothesis_family") or "CS"
        family = _pick(rec, "hypothesis_family", "id", "generalized_class")
        mech = _pick(rec, "generalized_class", "observed") or family
        verd = _pick(rec, "verdict", "status") or "FALSIFIED"
        claim = _pick(rec, "claim_refuted", "wrongly_concluded", "observed")
        out.append(
            _scar(
                rel,
                origin,
                str(oid),
                family_text=family or str(oid),
                mechanism=mech,
                verdict=verd,
                status=verd,
                reopen=_pick(rec, "reopen_condition", "reopen_if"),
                claim=claim,
                level=_pick(rec, "level") or "GENERAL_PHYSICAL",
            )
        )
    return out or [_unparsed(rel, "campaign scars had no entries", origin)]


def _parse_negative_science_v2(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    out = []
    for rec in obj.get("entries") or []:
        if not isinstance(rec, dict):
            continue
        oid = rec.get("id") or rec.get("technique") or "QN"
        out.append(
            _scar(
                rel,
                origin,
                str(oid),
                model_text=_pick(rec, "model"),
                organ_text=_pick(rec, "organ"),
                representation_text=_pick(rec, "representation"),
                machine_text=_pick(rec, "machine"),
                family_text=_pick(rec, "technique", "id"),
                mechanism=_pick(rec, "physical_reason", "technique"),
                verdict=_pick(rec, "capability") or "NEGATIVE",
                reopen=_pick(rec, "reopen_condition"),
                claim=_pick(rec, "capability", "physical_reason"),
                level=_pick(rec, "level") or "MODEL_SPECIFIC",
            )
        )
    return out or [_unparsed(rel, "negative_science.v2 had no entries", origin)]


def _parse_cross_expert_measurement(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    organs = " ".join(str(k) for k in comps.keys())
    claim_bits = []
    for name, stats in sorted(comps.items()):
        if not isinstance(stats, dict):
            continue
        # Cosine is a structure metric from the source receipt, cited as text.
        pcm = stats.get("pairwise_cosine_mean")
        k50 = stats.get("k50")
        n = stats.get("n")
        claim_bits.append(f"{name}: pairwise_cosine_mean={pcm} k50={k50}/{n}")
    n_exp = obj.get("n_experts")
    layer = obj.get("layer")
    claim = (
        f"Qwen80 L{layer} n_experts={n_exp}. "
        + "; ".join(claim_bits)
        + " — experts do not share a global template (near-orthogonal)."
    )
    return [
        _scar(
            rel,
            origin,
            "cross_expert_structure",
            model_text="qwen3-80b",
            organ_text=organs or "gate_proj up_proj",
            family_text="cross_expert_structure trivial global expert sharing",
            mechanism="trivial global expert sharing / shared expert template",
            verdict="NEGATIVE",
            status="NEGATIVE",
            reopen="a parent whose pairwise expert cosine is high enough that a shared template is cheaper than the bits it saves (foundry reopen: row-normalized mean off-diagonal cosine >= 0.10)",
            claim=claim,
            representation_text="shared expert template",
        )
    ]


def _parse_landed_science_scars(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    """Science scars carried in a tools/future receipt's own `scars` array.

    These receipts declare a well-formed machine-readable scar - family, level,
    mechanism, status, organ, parent, and often a `not` clause listing what the
    scar is NOT a retry of. Nothing read them, because SKIP_PREFIXES excludes
    receipts/future/ from the discovery sweep and only two files were seeded back.

    That gap had a measured cost. The model-bearing torture ran 30 minutes, made
    88 model calls and launched NOTHING, because choose() advertised
    WU.DEAD.mlp_function_replacement as policy: refuse_if_dead did not key
    MLP_FUNCTION_REPLACEMENT_CLOSED, so a school this campaign had closed hours
    earlier was still on the menu 45 times running and the resident kept picking
    it. A scar the index cannot see does not prune anything.
    """
    rows = obj.get("scars")
    if not isinstance(rows, list) or not rows:
        return []
    out: list[Scar] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        family = _pick(rec, "family", "hypothesis_family", "id")
        if not family:
            continue
        verd = _pick(rec, "status", "verdict") or "MEASURED_NEGATIVE"
        out.append(
            _scar(
                rel,
                origin,
                str(_pick(rec, "id", "family") or family),
                family_text=str(family),
                mechanism=_pick(rec, "mechanism", "why", "object") or str(family),
                verdict=verd,
                status=verd,
                reopen=_pick(rec, "reopen", "reopen_if", "reopen_condition"),
                claim=_pick(rec, "object", "claim_refuted", "not"),
                level=_pick(rec, "level") or "MODEL_SPECIFIC",
                model_text=_pick(rec, "parent", "model"),
                organ_text=_pick(rec, "organ"),
            )
        )
    return out


def parse_json(rel: str, text: str, origin: str) -> list[Scar]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return [_unparsed(rel, f"json decode: {e}", origin)]
    if not isinstance(obj, dict):
        return [_unparsed(rel, "json root is not an object", origin)]
    schema = str(obj.get("schema") or "")
    if schema.endswith("negative_transfer_atlas.v1") or "NEGATIVE_TRANSFER_ATLAS" in rel:
        if isinstance(obj.get("entries"), dict):
            return _parse_foundry(rel, obj, origin)
    if schema.endswith("negative_science_register.v1"):
        return _parse_register(rel, obj, origin)
    if schema.endswith("noetic_negative_science.v1"):
        return _parse_noetic(rel, obj, origin)
    if schema.endswith("negative_science.v2") and "entries" in obj:
        return _parse_negative_science_v2(rel, obj, origin)
    if schema.endswith("odyssey.negative_science.v1") or rel.endswith("odyssey/NEGATIVE_SCIENCE.json"):
        return _parse_odyssey(rel, obj, origin)
    if schema.endswith("campaign_scars.v1") or rel.endswith("CAMPAIGN_SCARS.json"):
        return _parse_campaign_scars(rel, obj, origin)
    if rel.startswith("receipts/future/") and isinstance(obj.get("scars"), list):
        landed = _parse_landed_science_scars(rel, obj, origin)
        if landed:
            return landed
    if "CROSS_EXPERT_STRUCTURE" in rel:
        return _parse_cross_expert_measurement(rel, obj, origin)
    if "n_experts" in obj and "components" in obj and "layer" in obj:
        return _parse_cross_expert_measurement(rel, obj, origin)
    return parse_json_generic(rel, obj, origin)


def parse_json_generic(rel: str, obj: dict[str, Any], origin: str) -> list[Scar]:
    # Single-result negatives (Q80 SIMD, lm_head, expert-wave, resident, phase B, …).
    verd = _pick(obj, "verdict", "result", "status", "outcome")
    if not verd:
        nested = obj.get("result")
        if isinstance(nested, dict):
            verd = _pick(nested, "verdict", "outcome", "status", "hypothesis")
            if _pick(nested, "outcome") == "REFUTED" or "REFUTED" in _txt(nested.get("outcome")):
                verd = "REFUTED"
    family = _pick(obj, "mechanism", "lever", "question", "hypothesis", "akb_registration")
    if not family:
        nested = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        family = _pick(nested, "hypothesis", "question")
    if verd and not family:
        family = rel
    if verd or family:
        model_text = _pick(obj, "model", "parent") or rel
        reopen = _pick(obj, "reopen_if", "reopen_condition", "retry_when", "next")
        claim = _pick(
            obj,
            "why",
            "why_it_failed",
            "what_this_falsifies",
            "my_error",
            "consequence",
            "killed_by",
        )
        if isinstance(obj.get("result"), dict) and not claim:
            claim = _pick(obj["result"], "hypothesis", "outcome")
        # Filename hints when the document is a named negative.
        low = rel.lower()
        if "lm_head" in low:
            family = family + " lm_head_precision"
            organ_text = "lm_head"
        elif "gk_simd" in low or "simd" in low:
            family = family + " gk_simd"
            organ_text = "mlp"
        elif "expert_wave" in low:
            family = family + " expert_wave"
            organ_text = "routed_experts"
        elif "resident_state" in low:
            family = family + " resident_state"
            organ_text = "kv"
        elif "hcli_coherence" in low:
            family = family + " hcli_coherence"
            organ_text = "whole_model"
            bind = obj.get("binding") if isinstance(obj.get("binding"), dict) else {}
            model_text = _pick(bind, "model_id") or model_text
        else:
            organ_text = _pick(obj, "organ")
        return [
            _scar(
                rel,
                origin,
                _slug(rel.split("/")[-1].replace(".json", "")) or "doc",
                model_text=model_text,
                organ_text=organ_text,
                family_text=family,
                mechanism=family,
                verdict=verd or "NEGATIVE",
                status=verd,
                reopen=reopen,
                claim=claim or _pick(obj, "claim_boundary"),
            )
        ]
    entries = obj.get("entries")
    if isinstance(entries, list):
        fake = {"schema": "hawking.odyssey.negative_science.v1", "entries": entries}
        return _parse_odyssey(rel, fake, origin)
    if isinstance(entries, dict):
        return _parse_foundry(rel, obj, origin)
    return [_unparsed(rel, "no keyed scar entries recognized", origin)]


def parse_markdown(rel: str, text: str, origin: str) -> list[Scar]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                continue
            current.append(cells)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    out: list[Scar] = []
    seen: set[str] = set()
    for table in tables:
        header = [_slug(h) for h in table[0]]
        idx = {name: i for i, name in enumerate(header)}

        def col(*names: str) -> int | None:
            for n in names:
                if n in idx:
                    return idx[n]
            return None

        i_mech = col("mechanism", "lever", "solution", "gate")
        i_verd = col("verdict", "status")
        i_reopen = col("reopen", "resurrection", "retry", "retry_when")
        i_ev = col("kill_number", "evidence_condensed", "evidence", "why", "note", "receipt")
        # Kill tables have a verdict/status column. Quantity/baseline tables do not.
        if i_mech is None or i_verd is None:
            continue
        for row_i, row in enumerate(table[1:], 1):
            def cell(i: int | None) -> str:
                if i is None or i >= len(row):
                    return ""
                return row[i]

            mech = cell(i_mech)
            verd = cell(i_verd)
            if not mech and not verd:
                continue
            oid = _slug(mech)[:80] or f"row{row_i}"
            if oid in seen:
                oid = f"{oid}-{row_i}"
            seen.add(oid)
            out.append(
                _scar(
                    rel,
                    origin,
                    oid,
                    family_text=mech,
                    mechanism=mech,
                    verdict=verd or "NEGATIVE",
                    status=verd,
                    reopen=cell(i_reopen),
                    claim=cell(i_ev) or mech,
                    model_text=f"{mech} {cell(i_ev)}",
                    organ_text=mech if extract_organs(mech) != [UNRECORDED] else "",
                )
            )
    if not out:
        return [_unparsed(rel, "markdown had no parseable tables", origin)]
    return out


def parse_python(rel: str, text: str, origin: str) -> list[Scar]:
    """Pull dict(id="QN-...", model=..., organ=..., technique=...) catalogs."""
    out: list[Scar] = []
    for m in re.finditer(
        r'dict\(\s*id="([^"]+)"\s*,\s*model="([^"]+)"\s*,\s*organ="([^"]+)"\s*,'
        r'\s*technique="([^"]+)"\s*,\s*representation="([^"]*)"',
        text,
    ):
        oid, model, organ, technique, representation = m.groups()
        window = text[m.end() : m.end() + 1800]
        cap = re.search(r'capability="([^"]+)"', window)
        reason = re.search(r'physical_reason="([^"]+)"', window)
        reopen = re.search(r'reopen_condition="([^"]+)"', window)
        machine = re.search(r'machine=([A-Z_]+)|machine="([^"]+)"', window)
        machine_text = ""
        if machine:
            machine_text = machine.group(2) or "M3 Ultra 96GB / Metal"
        out.append(
            _scar(
                rel,
                origin,
                oid,
                model_text=model,
                organ_text=organ,
                representation_text=representation,
                machine_text=machine_text,
                family_text=f"{oid} {technique}",
                mechanism=reason.group(1) if reason else technique,
                verdict="NEGATIVE",
                status="NEGATIVE",
                reopen=reopen.group(1) if reopen else "",
                claim=(cap.group(1) if cap else "") or technique,
            )
        )
    if out:
        return out
    # FailureClass enum members are a taxonomy, not measured scars.
    if "class FailureClass" in text or "CATALOG: list[dict]" in text:
        return [_unparsed(rel, "python implementation (catalog lives in its receipt; not re-extracted)", origin)]
    return [_unparsed(rel, "python file; no dict(id=...) catalog extracted", origin)]


def parse_source(rel: str) -> list[Scar]:
    text, origin = read_text(rel)
    if text is None:
        return [_unparsed(rel, "missing on disk and at HEAD", origin)]
    if rel.endswith(".lock") or not text.strip():
        return [_unparsed(rel, "empty or lock file", origin)]
    low = rel.lower()
    try:
        if low.endswith(".jsonl"):
            return parse_jsonl(rel, text, origin)
        if low.endswith(".json"):
            return parse_json(rel, text, origin)
        if low.endswith(".md"):
            return parse_markdown(rel, text, origin)
        if low.endswith(".py"):
            return parse_python(rel, text, origin)
    except Exception as e:  # never drop a source
        return [_unparsed(rel, f"parser raised {type(e).__name__}: {e}", origin)]
    return [_unparsed(rel, f"unsupported suffix", origin)]


def ingest(force: bool = False) -> list[Scar]:
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX
    scars: list[Scar] = []
    for rel in discover_sources():
        scars.extend(parse_source(rel))
    scars.sort(key=lambda s: (s.source_path, s.scar_id))
    _INDEX = scars
    return scars


def _model_hit(q_model: str | None, models: list[str]) -> str | None:
    if not q_model:
        return "wild"
    qm = canon_model(q_model)
    for m in models:
        if m == UNRECORDED:
            continue
        if canon_model(m) == qm:
            return "exact"
    return None


def _organ_hit(q_organ: str | None, scar: Scar) -> str | None:
    if not q_organ:
        return "wild"
    qo = canon_organ(q_organ)
    canons = [canon_organ(o) if o != UNRECORDED else UNRECORDED for o in scar.organs]
    if qo in canons:
        return "exact"
    if UNRECORDED in canons or "whole_model" in canons:
        return "weak"
    return None


def _field_hit(q: str | None, scar_val: str, canon) -> str | None:
    if not q:
        return "wild"
    qc = canon(q)
    sc = scar_val if scar_val != UNRECORDED else UNRECORDED
    if sc == UNRECORDED:
        return None
    if qc == sc:
        return "exact"
    if qc != UNRECORDED and (qc in sc or sc in qc):
        return "weak"
    return None


def _score(scar: Scar, q: dict[str, str | None]) -> int | None:
    """None = filtered out. Higher is more specific."""
    if scar.parse_status != PARSED:
        return None
    score = scar.keys_filled
    mh = _model_hit(q.get("model"), scar.models)
    # A GENERAL_PHYSICAL scar is a law about method and applies whatever parent
    # the query names. Filtering it out on a model miss made every process scar
    # unreachable from a model-specific proposal.
    if q.get("model") and mh is None and scar.level != "GENERAL_PHYSICAL":
        return None
    if mh == "exact":
        score += 8
    oh = _organ_hit(q.get("organ"), scar)
    if q.get("organ") and oh is None:
        return None
    if oh == "exact":
        score += 4
    elif oh == "weak":
        score += 1
    rh = _field_hit(q.get("representation"), scar.representation, canon_representation)
    if q.get("representation") and rh is None:
        return None
    if rh == "exact":
        score += 4
    fh = _field_hit(q.get("hypothesis_family"), scar.hypothesis_family, canon_family)
    if q.get("hypothesis_family") and fh is None:
        return None
    if fh == "exact":
        score += 16
    elif fh == "weak":
        score += 4
    xh = _field_hit(q.get("machine"), scar.machine, canon_machine)
    if q.get("machine") and xh is None:
        return None
    if xh == "exact":
        score += 2
    return score


def query(
    model: str | None = None,
    organ: str | None = None,
    representation: str | None = None,
    hypothesis_family: str | None = None,
    machine: str | None = None,
    scars: list[Scar] | None = None,
) -> list[dict[str, Any]]:
    """Matching scars, ranked by specificity. Each row has source_path."""
    pool = scars if scars is not None else ingest()
    q = {
        "model": model,
        "organ": organ,
        "representation": representation,
        "hypothesis_family": hypothesis_family,
        "machine": machine,
    }
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for s in pool:
        sc = _score(s, q)
        if sc is None:
            continue
        row = s.to_dict()
        row["match_score"] = sc
        ranked.append((-sc, s.scar_id, row))
    ranked.sort()
    return [r for _, _, r in ranked]


def _proposal_family(proposal: dict[str, Any]) -> str:
    for k in ("hypothesis_family", "technique", "mechanism", "lever", "seed", "family"):
        v = proposal.get(k)
        if v:
            return str(v)
    return ""


def refuse_if_dead(proposal: dict[str, Any] | str | None, scars: list[Scar] | None = None) -> dict[str, Any] | None:
    """Refusal citing a scar, or None. Targeted: family must match, and a
    MODEL_SPECIFIC scar does not prune a different named parent.
    """
    if proposal is None:
        return None
    if isinstance(proposal, str):
        try:
            loaded = json.loads(proposal)
            proposal = loaded if isinstance(loaded, dict) else {"hypothesis_family": proposal}
        except json.JSONDecodeError:
            proposal = {"hypothesis_family": proposal}
    if not isinstance(proposal, dict):
        return None
    family = _proposal_family(proposal)
    if not family:
        return None
    hits = query(
        model=proposal.get("model"),
        organ=proposal.get("organ"),
        representation=proposal.get("representation"),
        hypothesis_family=family,
        machine=proposal.get("machine"),
        scars=scars,
    )
    want_model = proposal.get("model")
    want_family = canon_family(family)
    for h in hits:
        if not h.get("refuse_eligible"):
            continue
        if h.get("parse_status") != PARSED:
            continue
        # Exact family gate: a weak substring hit from query() must not refuse.
        if want_family != h.get("hypothesis_family"):
            continue
        # A GENERAL_PHYSICAL scar is a law about method, not about a parent.
        # "a field named per_X is not per_X unless both sides count the same
        # events" does not stop being true because the query names a model, and
        # gating it on an exact model match made every process scar silently
        # unreachable from any model-specific proposal -- which is precisely the
        # narrow-probe-broad-label defect these scars record.
        if want_model and str(h.get("level") or "") != "GENERAL_PHYSICAL":
            mh = _model_hit(str(want_model), list(h.get("models") or [h.get("model") or UNRECORDED]))
            if mh != "exact":
                continue
        return {
            "refused": True,
            "reason": (
                "known-dead hypothesis; rediscovery is not free. "
                "Reopen only under the cited reopen_condition."
            ),
            "scar_id": h["scar_id"],
            "source_path": h["source_path"],
            "original_id": h.get("original_id"),
            "hypothesis_family": h.get("hypothesis_family"),
            "model": h.get("model"),
            "models": h.get("models"),
            "organ": h.get("organ"),
            "verdict": h.get("verdict"),
            "failure_mechanism": h.get("failure_mechanism"),
            "reopen_condition": h.get("reopen_condition"),
            "claim_refuted": h.get("claim_refuted"),
            "level": h.get("level"),
            "match_score": h.get("match_score"),
        }
    return None


def coverage(scars: list[Scar] | None = None) -> dict[str, Any]:
    pool = scars if scars is not None else ingest()
    by_source: dict[str, dict[str, Any]] = {}
    for s in pool:
        row = by_source.setdefault(
            s.source_path,
            {
                "path": s.source_path,
                "origin": s.source_origin,
                "n_scars": 0,
                "n_parsed": 0,
                "n_unparsed": 0,
            },
        )
        row["n_scars"] += 1
        if s.parse_status == PARSED:
            row["n_parsed"] += 1
        else:
            row["n_unparsed"] += 1
    families: dict[str, int] = {}
    models: dict[str, int] = {}
    for s in pool:
        if s.parse_status != PARSED:
            continue
        families[s.hypothesis_family] = families.get(s.hypothesis_family, 0) + 1
        for m in s.models:
            models[m] = models.get(m, 0) + 1
    n_unparsed = sum(1 for s in pool if s.parse_status != PARSED)
    n_parsed = sum(1 for s in pool if s.parse_status == PARSED)
    n_refuse = sum(1 for s in pool if s.refuse_eligible)
    return {
        "n_scars": len(pool),
        "n_parsed": n_parsed,
        "n_unparsed": n_unparsed,
        "n_refuse_eligible": n_refuse,
        "n_sources": len(by_source),
        "by_source": [by_source[k] for k in sorted(by_source)],
        # Lists, not maps: a family slug must never become a JSON key that
        # collides with HARDWARE_FIELDS (write_receipt walks every key).
        "by_hypothesis_family": [{"family": k, "n": families[k]} for k in sorted(families)],
        "by_model": [{"model": k, "n": models[k]} for k in sorted(models)],
        "does_not_cover": [
            "Protected GPU / FPGA / power-meter measurements (sidecar is STATIC_ONLY; bench state UNKNOWN).",
            "Promotion of MODEL_SPECIFIC scars to FAMILY or GENERAL_PHYSICAL — the index never auto-promotes.",
            "DIAGNOSTIC_RELATIVE vs PROTECTED_ABSOLUTE classification of source numbers (this lane produces neither).",
            ".hcli-legacy/bootstrap-director-v6/negative-science.jsonl (cited by noetic sweep; not in git HEAD).",
            "Live Codex receipts/headless beyond the named / name-scanned negative and *refuted* files.",
            "ramanujan.scaffold.core.stores Graveyard (in-process bury/revive, not a corpus index).",
            "Worktrees and grok/* branches the noetic sweeper walked; this index is HEAD + disk only.",
        ],
    }


def recovered_implementation() -> list[dict[str, str]]:
    return [
        {
            "path": "tools/headless/negative_science.py",
            "role": "nine-field failure store, prior_failures(organ, technique), three-level promotion gate",
            "adequate": "no",
            "gap": "query is organ/technique substring only; no representation/family keys; no refuse_if_dead; writes receipts/headless (Codex surface)",
        },
        {
            "path": "tools/headless/noetic_negative_science.py",
            "role": "32-entry archaeology catalog + sweep; writes receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            "adequate": "no",
            "gap": "hand catalog, not a keyed retrieval API a generator can call before proposing",
        },
        {
            "path": "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
            "role": "14 killed levers with reopen_condition (schema hawking.foundry.negative_transfer_atlas.v1)",
            "adequate": "no",
            "gap": "static atlas; no query / refuse path",
        },
        {
            "path": "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
            "role": "odyssey restatement of the foundry atlas plus the dense-lowrank Qwen3.8 kill",
            "adequate": "no",
            "gap": "document, not an index",
        },
        {
            "path": "research/lab/operators/ascension_graveyard.py",
            "role": "in-memory Graveyard with FailureClass taxonomy and bury/revive semantics",
            "adequate": "no",
            "gap": "empty of the campaign corpus; not wired to generators",
        },
        {
            "path": "workspace/docs/guides/dead_levers.md",
            "role": "canonical kill-ledger tables (throughput, silicon audit, Colab sub-Q4)",
            "adequate": "no",
            "gap": "markdown; nothing queried it",
        },
        {
            "path": "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
            "role": "L10 / 96-expert pairwise cosine measurement (near-orthogonal experts)",
            "adequate": "no",
            "gap": "raw measurement, no verdict field, not indexed",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "Heterogeneous ingest (JSON, JSONL, Markdown, a QWEN dict catalog) with UNPARSED retention.",
        "Alias-normalised keys: model, organ, representation, machine, hypothesis_family, failure_mechanism.",
        "query(...) ranked by specificity, every hit carrying source_path.",
        "refuse_if_dead(proposal) — targeted, not blanket; MODEL_SPECIFIC scars do not prune a different named parent.",
        "Sidecar receipt receipts/future/NEGATIVE_SCIENCE_INDEX.json (STATIC_ONLY / UNKNOWN) for sibling lanes.",
        "Coverage report that states how many scars, sources, unparsed, and what the index does not cover.",
        "Sparse-checkout ingest via git show HEAD:<path> so missing-on-disk is not treated as missing-in-git.",
    ]


def negative_findings() -> list[str]:
    return [
        "tools/headless/negative_science.py QWEN entries are not in the v1 NOETIC receipt; they are extracted from the Python catalog so they are not lost. Rebuilding the Codex receipt was out of scope.",
        ".hcli-legacy/bootstrap-director-v6/negative-science.jsonl is not in git HEAD; cited by noetic as HCLI tactic fingerprints, not representation science.",
        "The dual-gravity JSONL duplicates many qwen80/evolution JSONL records (same record_id). Both sources are kept; they are not merged.",
        "No FAMILY or GENERAL_PHYSICAL promotion was performed. Counts of distinct parents per family are visible in coverage.by_model but are not a promotion.",
        "Cannot certify whether any source number was DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE; this lane produces neither.",
    ]


def build(scars: list[Scar] | None = None) -> Any:
    pool = scars if scars is not None else ingest(force=True)
    cov = coverage(pool)
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Keyed index over Hawking's negative-science corpus, plus the "
            "refusal path every generator in this campaign calls before proposing."
        ),
        "how_to_use": {
            "query": (
                "from tools.future.negative_index import query; "
                "query(model=..., organ=..., representation=..., hypothesis_family=...)"
            ),
            "refuse_if_dead": (
                "from tools.future.negative_index import refuse_if_dead; "
                "refusal = refuse_if_dead({'model': ..., 'organ': ..., "
                "'representation': ..., 'hypothesis_family': ...})"
            ),
            "proposal_keys": [
                "model",
                "organ",
                "representation",
                "machine",
                "hypothesis_family",
            ],
            "refuse_semantics": (
                "Returns a dict citing source_path when a PARSED refuse-eligible "
                "scar matches the hypothesis family (and model, if the proposal "
                "names one). Returns None when the family is not dead on that "
                "parent. MODEL_SPECIFIC scars never prune a different named "
                "parent. A missing hypothesis_family is not a refuse. LIVE / "
                "untested / ARTIFACT_OF_METHOD entries are not refuse-eligible."
            ),
            "ranking": (
                "Hits are sorted by match_score descending: family +16, model +8, "
                "organ +4, representation +4, machine +2, plus keys_filled."
            ),
        },
        "key_space": {
            "model": "canonical parent (qwen3-80b, qwen3.8-27b, qwen3-235b-a22b, …)",
            "organ": "canonical organ (gate, up, down, attention, router, lm_head, …)",
            "representation": "canonical codec family (binary, q4, shared_basis, kronecker, …)",
            "machine": "canonical machine (m3_ultra, metal, apple_host_cpu, unrecorded)",
            "hypothesis_family": "canonical killed idea (cross_expert_structure, …)",
            "failure_mechanism": "source wording for why it failed",
        },
        "alias_tables": {
            "model_rules": [{"pattern": p, "canon": c} for p, c in MODEL_RULES],
            "organ_slugs": dict(sorted(ORGAN_SLUGS.items())),
            "family_slugs": dict(sorted(FAMILY_SLUGS.items())),
            "machine_slugs": dict(sorted(MACHINE_SLUGS.items())),
        },
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "coverage": cov,
        "scars": [s.to_dict() for s in pool],
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    return write_receipt(RECEIPT, doc, "tools/future/negative_index.py")


def selftest() -> Any:
    pool = ingest(force=True)
    if not pool:
        raise RuntimeError("negative index ingested zero records")
    dead = refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        },
        scars=pool,
    )
    if dead is None:
        # The Qwen80 measurement is the other known-dead parent.
        dead = refuse_if_dead(
            {
                "model": "qwen3-80b",
                "hypothesis_family": "trivial global expert sharing",
            },
            scars=pool,
        )
    if dead is None:
        raise RuntimeError("selftest: refuse_if_dead did not fire on a known-dead cross-expert hypothesis")
    live = refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "hwir_node_types",
        },
        scars=pool,
    )
    if live is not None:
        raise RuntimeError(f"selftest: refuse_if_dead blanket-refused a live family: {live}")
    return build(scars=pool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--query", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--organ")
    ap.add_argument("--representation")
    ap.add_argument("--hypothesis-family")
    ap.add_argument("--machine")
    ap.add_argument("--refuse", metavar="JSON")
    a = ap.parse_args()
    if a.refuse is not None:
        r = refuse_if_dead(a.refuse)
        print(json.dumps(r, indent=1, sort_keys=True) if r else "ALLOW")
        return 0 if r is None else 2
    if a.query:
        hits = query(
            model=a.model,
            organ=a.organ,
            representation=a.representation,
            hypothesis_family=a.hypothesis_family,
            machine=a.machine,
        )
        slim = [
            {
                "scar_id": h["scar_id"],
                "source_path": h["source_path"],
                "hypothesis_family": h["hypothesis_family"],
                "model": h["model"],
                "organ": h["organ"],
                "verdict": h["verdict"],
                "refuse_eligible": h["refuse_eligible"],
                "match_score": h["match_score"],
                "reopen_condition": h["reopen_condition"],
            }
            for h in hits[:40]
        ]
        print(json.dumps({"n": len(hits), "hits": slim}, indent=1, sort_keys=True))
        return 0
    if a.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
