#!/usr/bin/env python3
"""Odyssey-I Grok novelty engine (steer S004; bible §19/§54/§35/§36/§55).

When the deterministic engine converges to conventional / aggressive-quant
families with a large remaining target delta, emit a FRONTIER_NOVELTY_PACKET
and render independent Grok lane contracts for NONCONVENTIONAL mechanisms.

This module is deterministic and does not launch Grok. The driver/ctl
launches; Grok hypothesizes; scripts measure.

    python3 tools/odyssey_novelty.py --self-check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
ODYSSEY = REPO / "workspace" / "campaign" / "odyssey"
POLICY_PATH = ODYSSEY / "ODYSSEY_POLICY.json"
RULEBASE_PATH = ODYSSEY / "GRAVITY_RULEBASE.json"
NEGATIVE_PATH = ODYSSEY / "NEGATIVE_SCIENCE.json"
TRANSFER_PATH = ODYSSEY / "TRANSFER_MATRIX.json"
AUTO_DIR = ODYSSEY / "contracts" / "auto"
CANDIDATE_FAMILIES = ODYSSEY / "candidate_families.json"
GROK_TASKS = Path.home() / ".claude-grok" / "tasks"
LINT_JS = Path.home() / ".claude-grok" / "v2" / "lint.mjs"
NODE_BIN = Path("/opt/homebrew/bin/node")

SCHEMA_PACKET = "hawking.odyssey.frontier_novelty_packet.v1"
SCHEMA_PROPOSAL = "hawking.odyssey.novelty_proposal.v1"
SCHEMA_LANE_REPORT = "hawking.odyssey.novelty_lane_report.v1"
SCHEMA_FAMILY_ADD = "hawking.odyssey.candidate_family.v1"

LANES = (
    "representation",
    "numerical",
    "arch",
    "kernel",
    "adversarial-falsifier",
    "compression",
)

# Policy candidate_classes plus the candgen conventionality spellings.
CONVENTIONAL_CLASSES = frozenset({
    "BASELINE",
    "CONVENTIONAL",
    "CONVENTIONAL_ANCHOR",
})
AGGRESSIVE_CLASSES = frozenset({
    "AGGRESSIVE",
    "AGGRESSIVE_QUANT",
})
STRUCTURAL_OR_HIGHER = frozenset({
    "STRUCTURAL",
    "STRUCTURAL_GRAVITY",
    "ACTIVE_NX",
    "FRONTIER",
    "FINALIST",
})
SURVIVOR_STATUSES = frozenset({
    "PASS",
    "CANDIDATE_PASS",
    "SURVIVES",
    "SURVIVOR",
    "VERIFIED",
    "WIN",
    "WINS",
    "OK",
    "SUCCESS",
})
FAIL_STATUSES = frozenset({
    "FAIL",
    "FAILED",
    "REFUTED",
    "DEGRADED",
    "KILL",
    "KILLED",
    "DEAD",
    "REJECTED",
})

# Escalation_order_on_stall prefixes that must be done before grok_novelty_fanout.
PRIOR_STAGES = ("deterministic_search", "rule_transfer")

REQUIRED_PACKET_KEYS = (
    "schema",
    "oxx",
    "arch",
    "best_conventional_anchor",
    "aggressive_failures",
    "failure_localization",
    "stored_active_physical",
    "native_primitives",
    "negative_rules",
    "remaining_target_delta",
)

REQUIRED_PROPOSAL_FIELDS = (
    "mechanism",
    "complete_byte_accounting",
    "cheapest_falsifier",
    "execution_path",
    "kernel_implications",
    "applicability_class",
    "doctor_risk",
)

PROPOSAL_ALIASES = {
    "mechanism": ("mechanism", "mechanism_id", "family", "name", "hypothesis"),
    "complete_byte_accounting": (
        "complete_byte_accounting", "byte_accounting", "complete_bpw",
        "accounting", "bytes",
    ),
    "cheapest_falsifier": (
        "cheapest_falsifier", "falsifier", "cheapest_discriminator",
        "discriminator",
    ),
    "execution_path": (
        "execution_path", "native_execution", "nx_path", "execution",
    ),
    "kernel_implications": (
        "kernel_implications", "kernel", "kernel_gain", "kernel_story",
    ),
    "applicability_class": (
        "applicability_class", "applicability", "arch_class", "kind",
    ),
    "doctor_risk": ("doctor_risk", "doctor", "risk"),
}

JSON_OBJECT_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
JSON_ARRAY_FENCE_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)
OXX_RE = re.compile(r"\bO(\d{3})\b", re.I)
SLUG_OXX_RE = re.compile(r"odyssey-o(\d{3})", re.I)
# candgen / anti-complacency runner-spec grammar (q<bits>-g<group>[-experts],
# mixed-q2q3-experts, +correction, tiers). Harvest records the string; it
# does not require the runner to accept it yet.
RUNNER_SPEC_RE = re.compile(
    r"^(?:"
    r"q[1-8]-g\d+(?:-experts|-attn-mlp)?"
    r"|mixed-q[1-8]q[1-8](?:-experts)?"
    r"|matryoshka-T[0-3]"
    r"|tiers-T[0-3](?:-T[0-3])*"
    r")"
    r"(?:\+correction(?:-\d+(?:\.\d+)?)?)?"
    r"(?:\+tiers)?"
    r"$",
    re.I,
)
STAGE_ALIASES = {
    "deterministic_search": (
        "deterministic_search", "deterministic", "det_search",
        "deterministic-search",
    ),
    "rule_transfer": (
        "rule_transfer", "transfer", "rule-transfer", "transfer_matrix",
    ),
    "grok_novelty_fanout": (
        "grok_novelty_fanout", "grok_novelty", "novelty",
        "grok-novelty", "novelty_fanout",
    ),
}

# Presumptively conventional mechanism tokens (policy.conventionality_gate).
_CONV_NAME_RE = re.compile(
    r"(uniform-quant|affine-quant|symmetric-quant|per-group-scale|"
    r"ordinary-mixed-precision|gguf|mlx-like|q[34](?:-g\d+)?|"
    r"q4-g64|q3-g32)",
    re.I,
)
_AGG_NAME_RE = re.compile(
    r"(q2(?:-g\d+)?|q1(?:-g\d+)?|ternary|binary|~1\.x|sub-?2\b|"
    r"aggressive)",
    re.I,
)
_STRUCT_NAME_RE = re.compile(
    r"(mixed-q[1-8]q[1-8]|correction|matryoshka|tier|gather|"
    r"residual|procedural|organ|route-condition|nx-|active.nx|"
    r"structural|base\+correction|expert-family|prefetch)",
    re.I,
)

LANE_BRIEFS: dict[str, dict[str, str]] = {
    "representation": {
        "question": (
            "Which NONCONVENTIONAL representation (not uniform/affine/"
            "symmetric/per-group-scale/ordinary mixed-precision) can close "
            "the remaining stored/active target delta on this patient?"
        ),
        "search": (
            "base+correction (sparse/residual/selected-hi-prec-channels/"
            "expert- or route-conditioned/procedural), Matryoshka T0..T3, "
            "per-organ or per-expert codecs, layer-0-as-different-source, "
            "source-changing methods (not raw-weight PQ/VQ at ~1 bit)."
        ),
        "negatives": (
            "NS-raw-weight-pq-vq-at-one-bit, NS-uniform-subbit-allocation, "
            "NS-kronecker-factorisation (depth; layer-0 is the named exception), "
            "NS-post-hoc-coding-of-frozen-weights"
        ),
    },
    "numerical": {
        "question": (
            "Which numerical / metadata / scale×codec joint (not another "
            "integer bit-width) changes complete_bpw without a global "
            "precision retreat?"
        ),
        "search": (
            "alt group sizes, shared vs entropy-coded metadata, scale×codec "
            "joint, non-Lloyd / biased codebooks, non-conditional-mean "
            "gain, row-direction VQ + per-row scale. Query negatives first."
        ),
        "negatives": (
            "NS-entropy-coded-pq-indices (Lloyd-optimal indices are near-"
            "uniform), NS-posthoc-scalar-gain, NS-row-norm-stratification-premise, "
            "NS-ternary-factorization"
        ),
    },
    "arch": {
        "question": (
            "What architecture-conditioned lever (stored vs active vs state "
            "residency vs per-modality organs) is the real remaining delta, "
            "and which NONCONVENTIONAL mechanism attacks that axis?"
        ),
        "search": (
            "MoE: selected/full is a SELECTION opportunity, not 1/16 cost; "
            "native expert-gather; route-conditioned repr; no cold-expert "
            "assumption until routing is measured. Hybrid: SSM state vs KV. "
            "Multimodal: per-modality organs. MTP: tokens/expensive-traversal."
        ),
        "negatives": (
            "NS-inter-expert-redundancy, NS-expert-merging-omitted-from-survivors, "
            "NS-cross-expert-and-cross-layer-tying, NS-global-dense-lowrank-qwen38 "
            "(does NOT auto-kill MoE shared structure)"
        ),
    },
    "kernel": {
        "question": (
            "What native kernel / NX primitive must exist for a proposed "
            "repr to become a Hawking win (not a foreign-runtime SPECIMEN), "
            "and what relative kernel gain is the cheapest honest target?"
        ),
        "search": (
            "bible §54: 1.15x / 1.25x / 1.5x where headroom is obvious; "
            "do not invent a 5x. Wrapper TPS is not a Hawking win. Gather "
            "must move only active experts. Record kernel implications of "
            "the proposed codec (ALU, gather, prefetch, residency)."
        ),
        "negatives": (
            "NS-large-expert-cache (no cross-layer reuse on a lockstep pass); "
            "R-affine-grouped-q2-if-native-kernel (Doctor-valid q2 without a "
            "native kernel is not a pass)"
        ),
    },
    "adversarial-falsifier": {
        "question": (
            "For every live conventional/aggressive point and every novelty "
            "hypothesis on this packet: what is the cheapest falsifier, "
            "which negative-science entry already kills it, and where is "
            "the Goodhart hole?"
        ),
        "search": (
            "Kill first. Scope + evidence + arch_assumptions + reopen_if "
            "on every kill (policy.negative_science). Predicates over "
            "blacklists. Anti-Goodhart: complete_bpw, held-out Doctor "
            "pattern, info-count = novelty of evidence not receipt count."
        ),
        "negatives": (
            "entire NEGATIVE_SCIENCE.json; also policy.accounting_gates "
            "(no_fake_density / no_fake_active_density / no_fake_tps)"
        ),
    },
    "compression": {
        "question": (
            "Where do the remaining bytes actually live (payload vs scales "
            "vs biases vs tables vs offsets vs correction vs tier/router "
            "metadata vs alignment vs headers vs repr-attributable state), "
            "and which NONCONVENTIONAL compression attacks the dominant bin?"
        ),
        "search": (
            "complete_bpw accounting; stored vs active vs DRAM-traffic; "
            "sidecar bytes; correction-tier budget; do not claim sub-1 "
            "from nominal code bits. Physical numbers are MEASURED."
        ),
        "negatives": (
            "NS-post-hoc-coding-of-frozen-weights, NS-entropy-coded-pq-indices, "
            "policy.accounting_gates.no_fake_density"
        ),
    },
}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_policy(path: Path | None = None) -> dict[str, Any]:
    return _read_json(Path(path) if path else POLICY_PATH)


def _as_mapping(src: Any, default_path: Path | None = None) -> dict[str, Any]:
    if src is None:
        if default_path is not None and default_path.is_file():
            loaded = _read_json(default_path)
            return loaded if isinstance(loaded, dict) else {}
        return {}
    if isinstance(src, dict):
        return src
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.is_file():
            loaded = _read_json(p)
            return loaded if isinstance(loaded, dict) else {}
    return {}


def _as_list(src: Any) -> list[Any]:
    if src is None:
        return []
    if isinstance(src, list):
        return src
    if isinstance(src, dict):
        if isinstance(src.get("receipts"), list):
            return list(src["receipts"])
        if isinstance(src.get("entries"), list):
            return list(src["entries"])
        return [src]
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.is_file():
            loaded = json.loads(p.read_text())
            return _as_list(loaded)
        if p.is_dir():
            out: list[Any] = []
            for child in sorted(p.glob("*.json")):
                try:
                    out.extend(_as_list(json.loads(child.read_text())))
                except (OSError, json.JSONDecodeError):
                    continue
            return out
    return [src]


def _oxx_of(patient: Any, packet: dict[str, Any] | None = None) -> str:
    if isinstance(patient, str) and patient.strip():
        m = OXX_RE.search(patient)
        return f"O{m.group(1)}" if m else patient.strip()
    if isinstance(patient, dict):
        for key in ("oxx", "patient", "id", "patient_id"):
            v = patient.get(key)
            if v:
                return _oxx_of(str(v))
    if packet:
        v = packet.get("oxx") or (packet.get("identity") or {}).get("oxx")
        if v:
            return _oxx_of(str(v))
    return "UNKNOWN"


def _norm_token(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _norm_class(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        for key in ("candidate_class", "conventionality", "class", "best_class"):
            if raw.get(key):
                return _norm_class(raw[key])
        return ""
    s = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "CONVENTIONAL": "CONVENTIONAL_ANCHOR",
        "ANCHOR": "CONVENTIONAL_ANCHOR",
        "AGGRESSIVE": "AGGRESSIVE_QUANT",
        "QUANT": "AGGRESSIVE_QUANT",
        "STRUCTURAL": "STRUCTURAL_GRAVITY",
        "GRAVITY": "STRUCTURAL_GRAVITY",
        "NX": "ACTIVE_NX",
        "ACTIVE": "ACTIVE_NX",
    }
    return aliases.get(s, s)


def _norm_stage(raw: Any) -> str:
    tok = _norm_token(raw)
    for canon, aliases in STAGE_ALIASES.items():
        if tok in {_norm_token(a) for a in aliases}:
            return canon
    return str(raw or "").strip().lower()


def classify_family(item: Any) -> str:
    """Return CONVENTIONAL_ANCHOR / AGGRESSIVE_QUANT / STRUCTURAL_GRAVITY / ''."""
    if item is None:
        return ""
    if isinstance(item, dict):
        for key in ("conventionality", "candidate_class", "class", "family_class"):
            c = _norm_class(item.get(key))
            if c in CONVENTIONAL_CLASSES:
                return "CONVENTIONAL_ANCHOR"
            if c in AGGRESSIVE_CLASSES:
                return "AGGRESSIVE_QUANT"
            if c in STRUCTURAL_OR_HIGHER:
                return c if c in STRUCTURAL_OR_HIGHER else "STRUCTURAL_GRAVITY"
        name = str(
            item.get("id") or item.get("family") or item.get("spec")
            or item.get("mechanism") or item.get("name") or ""
        )
        return classify_family(name)
    name = str(item)
    if _STRUCT_NAME_RE.search(name):
        return "STRUCTURAL_GRAVITY"
    if _AGG_NAME_RE.search(name):
        return "AGGRESSIVE_QUANT"
    if _CONV_NAME_RE.search(name):
        return "CONVENTIONAL_ANCHOR"
    c = _norm_class(name)
    if c in CONVENTIONAL_CLASSES:
        return "CONVENTIONAL_ANCHOR"
    if c in AGGRESSIVE_CLASSES:
        return "AGGRESSIVE_QUANT"
    if c in STRUCTURAL_OR_HIGHER:
        return "STRUCTURAL_GRAVITY"
    return ""


def _family_status(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = item.get("status") or item.get("verdict") or item.get("result") or ""
    return str(raw).strip().upper().replace("-", "_").replace(" ", "_")


def _is_survivor(item: Any) -> bool:
    st = _family_status(item)
    if st in SURVIVOR_STATUSES:
        return True
    if st in FAIL_STATUSES:
        return False
    if isinstance(item, dict) and item.get("survived") is True:
        return True
    return False


def large_delta_threshold(policy: dict[str, Any] | None = None) -> float:
    """Remaining-bpw gap that counts as 'large' (policy pressure zones)."""
    pol = policy if policy is not None else load_policy()
    zones = pol.get("target_pressure_zones_bpw") or {}
    try:
        reachable = float(zones.get("reachable_or_explained", 3.0))
        pressure = float(zones.get("pressure", 2.5))
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, reachable - pressure)


def coerce_target_delta(
    target_delta: Any, policy: dict[str, Any] | None = None,
) -> float | None:
    if target_delta is None:
        return None
    if isinstance(target_delta, bool):
        return None
    if isinstance(target_delta, (int, float)):
        return float(target_delta)
    if isinstance(target_delta, str):
        try:
            return float(target_delta.strip())
        except ValueError:
            return None
    if isinstance(target_delta, dict):
        for key in ("target_delta", "remaining", "remaining_target_delta", "delta"):
            if target_delta.get(key) is not None:
                return coerce_target_delta(target_delta[key], policy)
        best = None
        for key in ("best_bpw", "best_complete_bpw", "complete_bpw", "stored_bpw"):
            if target_delta.get(key) is not None:
                try:
                    best = float(target_delta[key])
                    break
                except (TypeError, ValueError):
                    pass
        tgt = None
        for key in ("target_bpw", "target", "pressure_bpw"):
            if target_delta.get(key) is not None:
                try:
                    tgt = float(target_delta[key])
                    break
                except (TypeError, ValueError):
                    pass
        if tgt is None:
            zones = (policy or {}).get("target_pressure_zones_bpw") or {}
            try:
                tgt = float(zones.get("pressure", 2.5))
            except (TypeError, ValueError):
                tgt = 2.5
        if best is not None:
            return best - tgt
    return None


def _iter_families(families_tried: Any) -> list[Any]:
    if families_tried is None:
        return []
    if isinstance(families_tried, dict):
        for key in ("families", "families_tried", "tried", "tried_mechanisms"):
            if isinstance(families_tried.get(key), list):
                return list(families_tried[key])
        return [families_tried]
    if isinstance(families_tried, (list, tuple, set)):
        return list(families_tried)
    return [families_tried]


def _stages_from(patient: Any, families_tried: Any) -> set[str]:
    stages: set[str] = set()

    def absorb(raw: Any) -> None:
        if raw is None:
            return
        if isinstance(raw, (list, tuple, set)):
            for x in raw:
                stages.add(_norm_stage(x))
        else:
            stages.add(_norm_stage(raw))

    if isinstance(patient, dict):
        for key in (
            "stages_completed", "completed_stages", "escalation_done",
            "stages",
        ):
            absorb(patient.get(key))
        cur = patient.get("escalation_stage") or patient.get("search_stage")
        if cur:
            cur_n = _norm_stage(cur)
            stages.add(cur_n)
            # Standing at grok_novelty (or later) implies the prior two ran.
            order = [
                "deterministic_search", "rule_transfer",
                "grok_novelty_fanout", "grok_adversarial_review", "opus",
            ]
            if cur_n in order:
                idx = order.index(cur_n)
                stages.update(order[:idx])
        if patient.get("deterministic_exhausted") or patient.get(
            "deterministic_search_exhausted"
        ) or patient.get("deterministic_search"):
            stages.add("deterministic_search")
        if patient.get("rule_transfer_done") or patient.get(
            "rule_transfer_exhausted"
        ) or patient.get("rule_transfer"):
            stages.add("rule_transfer")

    blob = families_tried
    if isinstance(blob, dict):
        if blob.get("deterministic_search") or blob.get("deterministic_exhausted"):
            stages.add("deterministic_search")
        if blob.get("rule_transfer") or blob.get("rule_transfer_done"):
            stages.add("rule_transfer")
        absorb(blob.get("stages_completed"))
        for item in _iter_families(blob):
            if isinstance(item, dict) and item.get("stage"):
                if item.get("done") is False:
                    continue
                stages.add(_norm_stage(item.get("stage")))
            elif isinstance(item, str) and _norm_stage(item) in STAGE_ALIASES:
                stages.add(_norm_stage(item))
    elif isinstance(blob, (list, tuple)):
        for item in blob:
            if isinstance(item, dict) and item.get("stage"):
                if item.get("done") is False:
                    continue
                stages.add(_norm_stage(item.get("stage")))
            elif isinstance(item, str) and _norm_stage(item) in STAGE_ALIASES:
                stages.add(_norm_stage(item))
    return {s for s in stages if s}


def deterministic_search_exhausted(
    patient: Any, families_tried: Any,
    policy: dict[str, Any] | None = None,
) -> bool:
    """True iff deterministic_search and rule_transfer are done.

    escalation_order_on_stall: deterministic_search → rule_transfer →
    grok_novelty_fanout. Grok novelty is third; do not skip the first two.
    """
    stages = _stages_from(patient, families_tried)
    return set(PRIOR_STAGES) <= stages


def has_structural_survivor(best_class: Any, families_tried: Any) -> bool:
    bc = _norm_class(best_class)
    if bc in STRUCTURAL_OR_HIGHER:
        return True
    for item in _iter_families(families_tried):
        klass = classify_family(item)
        if klass in STRUCTURAL_OR_HIGHER and _is_survivor(item):
            return True
    return False


def should_escalate(
    patient: Any,
    best_class: Any,
    target_delta: Any,
    families_tried: Any,
    policy: dict[str, Any] | None = None,
) -> bool:
    """Escalate to grok-novelty when conventional search has stalled.

    True only when:
      - at least one family was tried
      - no STRUCTURAL (or higher) survivor (best_class and families)
      - target_delta is large vs ODYSSEY_POLICY pressure zones
      - deterministic_search and rule_transfer are exhausted
    """
    pol = policy if policy is not None else load_policy()
    tried = _iter_families(families_tried)
    if not tried:
        return False
    if has_structural_survivor(best_class, families_tried):
        return False
    delta = coerce_target_delta(target_delta, pol)
    if delta is None or delta < large_delta_threshold(pol):
        return False
    if not deterministic_search_exhausted(patient, families_tried, pol):
        return False
    return True


# ---------------------------------------------------------------------------
# FRONTIER_NOVELTY_PACKET
# ---------------------------------------------------------------------------

def _arch_of(patient: Any, packet: dict[str, Any]) -> dict[str, Any]:
    ident = packet.get("identity") if isinstance(packet.get("identity"), dict) else {}
    arch = packet.get("architecture") if isinstance(packet.get("architecture"), dict) else {}
    if not arch and isinstance(patient, dict):
        arch = patient.get("architecture") if isinstance(patient.get("architecture"), dict) else {}
        if not arch and patient.get("kind"):
            arch = {"kind": patient.get("kind")}
    kind = (
        arch.get("kind")
        or (patient.get("kind") if isinstance(patient, dict) else None)
        or (patient.get("class") if isinstance(patient, dict) else None)
        or packet.get("class")
        or "unknown"
    )
    return {
        "oxx": _oxx_of(patient, packet),
        "kind": kind,
        "arch": arch.get("arch") or arch.get("name"),
        "total_params": arch.get("total_params"),
        "active_params": arch.get("active_params") or arch.get("active_params_per_token"),
        "active_pct": arch.get("active_pct"),
        "layers": arch.get("layers"),
        "experts": arch.get("experts"),
        "experts_per_tok": arch.get("experts_per_tok"),
        "modality": arch.get("modality"),
        "source_repo": ident.get("source_repo") or (
            patient.get("source") if isinstance(patient, dict) else None
        ),
        "model_family": ident.get("model_family"),
        "objective": None,
        "_evidence": arch.get("_evidence") or "INFERRED",
    }


def _policy_objective(kind: Any, policy: dict[str, Any]) -> str | None:
    obj = policy.get("arch_objective") or {}
    k = str(kind or "").lower()
    if "moe" in k:
        return obj.get("moe")
    if "hybrid" in k or "ssm" in k or "mamba" in k:
        return obj.get("hybrid_ssm")
    if "multi" in k or "vl" in k:
        return obj.get("multimodal")
    if "mtp" in k:
        return obj.get("mtp")
    if "stream" in k:
        return obj.get("streamed")
    if "dense" in k:
        return obj.get("dense")
    return obj.get("dense")


def _receipt_class(rec: dict[str, Any]) -> str:
    return _norm_class(
        rec.get("candidate_class") or rec.get("conventionality")
        or classify_family(rec.get("spec") or rec.get("quant") or rec)
    )


def _is_fail_receipt(rec: dict[str, Any]) -> bool:
    v = str(rec.get("verdict") or rec.get("status") or "").upper()
    if v in FAIL_STATUSES or "DEGRADED" in v or "FAIL" in v:
        return True
    try:
        if rec.get("delta_hits") is not None and float(rec["delta_hits"]) < -1:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _best_conventional_anchor(
    packet: dict[str, Any], receipts: list[Any],
) -> dict[str, Any]:
    grav = packet.get("gravity") if isinstance(packet.get("gravity"), dict) else {}
    candidates: list[dict[str, Any]] = []
    for rec in receipts:
        if not isinstance(rec, dict):
            continue
        klass = _receipt_class(rec)
        if klass not in CONVENTIONAL_CLASSES and classify_family(
            rec.get("spec") or rec
        ) not in CONVENTIONAL_CLASSES:
            # q3/q4 affine is the conventional anchor even if untagged
            spec = str(rec.get("spec") or rec.get("quant") or "")
            if not _CONV_NAME_RE.search(spec):
                continue
            klass = "CONVENTIONAL_ANCHOR"
        if _is_fail_receipt(rec):
            continue
        candidates.append({
            "spec": rec.get("spec") or rec.get("quant"),
            "candidate_class": klass or "CONVENTIONAL_ANCHOR",
            "stored_bpw": rec.get("stored_bpw") or rec.get("complete_bpw"),
            "complete_bpw": rec.get("complete_bpw") or rec.get("stored_bpw"),
            "active_bpw": rec.get("active_bpw"),
            "nominal_bits": rec.get("nominal_bits"),
            "battery": rec.get("battery"),
            "delta_hits": rec.get("delta_hits"),
            "verdict": rec.get("verdict"),
            "receipt": rec.get("out") or rec.get("receipt"),
            "label": rec.get("label"),
            "_evidence": rec.get("_evidence") or "MEASURED",
        })
    for win in grav.get("wins") or []:
        if not isinstance(win, dict):
            continue
        spec = str(win.get("spec") or "")
        klass = _norm_class(win.get("candidate_class") or win.get("conventionality"))
        if klass in STRUCTURAL_OR_HIGHER:
            continue
        if klass not in CONVENTIONAL_CLASSES and not _CONV_NAME_RE.search(spec):
            continue
        candidates.append({
            "spec": win.get("spec"),
            "candidate_class": klass or "CONVENTIONAL_ANCHOR",
            "stored_bpw": win.get("stored_bpw"),
            "complete_bpw": win.get("complete_bpw") or win.get("stored_bpw"),
            "active_bpw": win.get("active_bpw"),
            "battery": win.get("battery"),
            "delta_hits": win.get("delta_hits"),
            "verdict": win.get("verdict"),
            "receipt": win.get("receipt"),
            "label": win.get("label"),
            "_evidence": win.get("_evidence") or "MEASURED",
        })
    if not candidates:
        last = grav.get("last") if isinstance(grav.get("last"), dict) else {}
        if last and _CONV_NAME_RE.search(str(last.get("spec") or "")):
            candidates.append({
                "spec": last.get("spec"),
                "candidate_class": "CONVENTIONAL_ANCHOR",
                "stored_bpw": last.get("stored_bpw"),
                "complete_bpw": last.get("complete_bpw") or last.get("stored_bpw"),
                "active_bpw": last.get("active_bpw"),
                "verdict": last.get("verdict"),
                "receipt": last.get("receipt"),
                "_evidence": last.get("_evidence") or "INFERRED",
            })
    if not candidates:
        return {
            "spec": None,
            "candidate_class": None,
            "stored_bpw": None,
            "complete_bpw": None,
            "active_bpw": None,
            "note": "no conventional anchor in packet/receipts",
            "_evidence": "UNKNOWN",
        }

    def _bpw(c: dict[str, Any]) -> float:
        for k in ("complete_bpw", "stored_bpw"):
            try:
                if c.get(k) is not None:
                    return float(c[k])
            except (TypeError, ValueError):
                pass
        return 1e9

    candidates.sort(key=_bpw)
    best = dict(candidates[0])
    best["n_anchors"] = len(candidates)
    return best


def _aggressive_failures(
    packet: dict[str, Any], receipts: list[Any],
) -> tuple[list[dict[str, Any]], list[Any]]:
    grav = packet.get("gravity") if isinstance(packet.get("gravity"), dict) else {}
    fails: list[dict[str, Any]] = []
    locs: list[Any] = []
    seen: set[str] = set()

    def absorb(rec: dict[str, Any], source: str) -> None:
        klass = _receipt_class(rec) or classify_family(rec.get("spec") or rec)
        is_aggr = klass in AGGRESSIVE_CLASSES or bool(
            _AGG_NAME_RE.search(str(rec.get("spec") or rec.get("quant") or ""))
        )
        is_fail = _is_fail_receipt(rec) or source == "kill"
        if not (is_aggr or is_fail):
            return
        if not is_fail and klass not in AGGRESSIVE_CLASSES:
            return
        loc = (
            rec.get("failure_localization")
            or rec.get("localization")
            or rec.get("targeted_repair")
        )
        spec = rec.get("spec") or rec.get("quant") or rec.get("mechanism")
        key = _norm_token(spec)
        if key and key in seen:
            if loc:
                for existing in fails:
                    if _norm_token(existing.get("spec")) == key and not existing.get(
                        "failure_localization"
                    ):
                        existing["failure_localization"] = loc
                if loc not in locs:
                    locs.append(loc)
            return
        if key:
            seen.add(key)
        if loc:
            locs.append(loc)
        fails.append({
            "spec": spec,
            "candidate_class": klass or "AGGRESSIVE_QUANT",
            "verdict": rec.get("verdict") or rec.get("status"),
            "stored_bpw": rec.get("stored_bpw") or rec.get("complete_bpw"),
            "active_bpw": rec.get("active_bpw"),
            "delta_hits": rec.get("delta_hits"),
            "failure_localization": loc,
            "receipt": rec.get("out") or rec.get("receipt"),
            "source": source,
            "_evidence": rec.get("_evidence") or "MEASURED",
        })

    for rec in receipts:
        if isinstance(rec, dict):
            absorb(rec, "receipt")
    for kill in grav.get("kills") or []:
        if isinstance(kill, dict):
            absorb(dict(kill, verdict=kill.get("verdict") or "KILLED"), "kill")
        elif kill:
            fails.append({
                "spec": str(kill),
                "candidate_class": classify_family(kill) or "AGGRESSIVE_QUANT",
                "verdict": "KILLED",
                "failure_localization": None,
                "source": "kill",
                "_evidence": "INFERRED",
            })
    return fails, locs


def _stored_active_physical(
    packet: dict[str, Any], receipts: list[Any], policy: dict[str, Any],
) -> dict[str, Any]:
    rep = packet.get("representation") if isinstance(packet.get("representation"), dict) else {}
    exe = packet.get("execution") if isinstance(packet.get("execution"), dict) else {}
    nx = packet.get("nx") if isinstance(packet.get("nx"), dict) else {}
    grav = packet.get("gravity") if isinstance(packet.get("gravity"), dict) else {}
    last = grav.get("last") if isinstance(grav.get("last"), dict) else {}

    stored_bpw = (
        last.get("stored_bpw") or rep.get("best_stored_bpw_eq")
        or rep.get("stored_bpw")
    )
    active_bpw = last.get("active_bpw") or rep.get("active_bpw_eq")
    stored_bytes = last.get("stored_bytes") or rep.get("source_bytes")
    active_bytes = (
        last.get("active_bytes_per_token")
        or exe.get("active_learned_bytes_per_token")
        or nx.get("selected_bytes_per_token")
        or rep.get("active_bytes_per_token_bf16")
    )
    for rec in receipts:
        if not isinstance(rec, dict) or _is_fail_receipt(rec):
            continue
        if rec.get("stored_bpw") is not None:
            try:
                val = float(rec["stored_bpw"])
                if stored_bpw is None or val < float(stored_bpw):
                    stored_bpw = val
                    stored_bytes = rec.get("stored_bytes") or stored_bytes
            except (TypeError, ValueError):
                pass
        if rec.get("active_bpw") is not None:
            try:
                aval = float(rec["active_bpw"])
                if active_bpw is None or aval < float(active_bpw):
                    active_bpw = aval
                    active_bytes = rec.get("active_bytes_per_token") or active_bytes
            except (TypeError, ValueError):
                pass

    def _r4(v: Any) -> Any:
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return v

    return {
        "stored": {
            "bpw": _r4(stored_bpw) if stored_bpw is not None else None,
            "bytes": stored_bytes,
            "organs_bytes_GB": rep.get("organs_bytes_GB"),
            "metadata_bytes": rep.get("metadata_bytes"),
            "corrections": rep.get("corrections"),
        },
        "active": {
            "bpw": _r4(active_bpw) if active_bpw is not None else None,
            "bytes_per_token": active_bytes,
            "selected_bytes_per_token": nx.get("selected_bytes_per_token"),
            "full_expert_body_bytes": nx.get("full_expert_body_bytes"),
            "ratio_selected_over_full": nx.get("ratio_selected_over_full"),
        },
        "physical": {
            "tps": exe.get("tps") or exe.get("baseline_tps") or exe.get("tps_specimen"),
            "ttft": exe.get("ttft"),
            "token_ns": exe.get("token_ns"),
            "dram_per_token": exe.get("dram_per_token"),
            "label": exe.get("label") or exe.get("not_base_true_tps"),
            "not_base_true_tps": exe.get("not_base_true_tps", True),
            "runtime": exe.get("baseline_runtime"),
        },
        "accounting_gates": (policy.get("accounting_gates") or {}),
        "_evidence": rep.get("_evidence") or exe.get("_evidence") or "INFERRED",
    }


def _native_primitives(packet: dict[str, Any], arch: dict[str, Any]) -> dict[str, Any]:
    nx = packet.get("nx") if isinstance(packet.get("nx"), dict) else {}
    prim = nx.get("primitive_set")
    derived: list[str] = []
    kind = str(arch.get("kind") or "").lower()
    if "moe" in kind:
        derived.extend([
            "expert-gather",
            "route-prefetch",
            "per-expert-codec",
            "router-protect",
        ])
    if "hybrid" in kind or "ssm" in kind or "mamba" in kind:
        derived.extend(["ssm-state-residency", "attn-mlp-split"])
    if "dense" in kind or not derived:
        derived.extend(["per-organ-codec", "layer-codec-select"])
    return {
        "declared": prim,
        "machine_lowering": nx.get("machine_lowering"),
        "kernel_bindings": nx.get("kernel_bindings"),
        "fallback_count": nx.get("fallback_count"),
        "best_preliminary_nx": nx.get("best_preliminary_nx"),
        "mlx_gathers": nx.get("mlx_gathers"),
        "derived_hypotheses": derived,
        "not_hawking_nx_win": nx.get("not_hawking_nx_win", True),
        "_evidence": nx.get("_evidence") or (
            "MEASURED" if prim else "HYPOTHESIS (derived from arch.kind)"
        ),
    }


def _negative_rules(
    negatives: dict[str, Any],
    transfer: dict[str, Any],
    rulebase: dict[str, Any],
    oxx: str,
    arch: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in negatives.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        out.append({
            "id": entry.get("id"),
            "mechanism": entry.get("mechanism"),
            "verdict": entry.get("verdict"),
            "reopen_if": entry.get("reopen_if"),
            "does_not_automatically_kill": entry.get("does_not_automatically_kill"),
            "killed_on": entry.get("killed_on"),
            "_evidence": entry.get("_evidence") or "INFERRED",
        })
    # Transfer cells for this patient that are PATIENT_SPECIFIC / FAILED /
    # HARMFUL are live negative constraints.
    for row in transfer.get("rows") or []:
        if not isinstance(row, dict):
            continue
        cell = (row.get("cells") or {}).get(oxx)
        if cell in {"FAILED", "HARMFUL", "PATIENT_SPECIFIC"}:
            out.append({
                "id": row.get("rule"),
                "mechanism": row.get("rule"),
                "verdict": cell,
                "reopen_if": None,
                "note": row.get("note"),
                "source": "TRANSFER_MATRIX",
                "_evidence": row.get("_evidence") or "INFERRED",
            })
    kind = str(arch.get("kind") or "").lower()
    for rule in rulebase.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        conds = " ".join(str(c) for c in (rule.get("conditions") or []))
        if "moe" in conds and "moe" not in kind:
            continue
        out.append({
            "id": rule.get("id"),
            "mechanism": rule.get("then"),
            "verdict": "RULE",
            "reopen_if": rule.get("reopen_if"),
            "confidence": rule.get("confidence"),
            "source": "GRAVITY_RULEBASE",
            "_evidence": rule.get("_evidence") or rule.get("confidence") or "HYPOTHESIS",
        })
    return out


def _remaining_target_delta(
    decomp: dict[str, Any],
    policy: dict[str, Any],
    arch: dict[str, Any],
    explicit: Any = None,
) -> dict[str, Any]:
    zones = policy.get("target_pressure_zones_bpw") or {}
    try:
        pressure = float(zones.get("pressure", 2.5))
    except (TypeError, ValueError):
        pressure = 2.5
    stored = (decomp.get("stored") or {}).get("bpw")
    active = (decomp.get("active") or {}).get("bpw")
    def _r4(v: Any) -> float | None:
        try:
            return None if v is None else round(float(v), 4)
        except (TypeError, ValueError):
            return None

    stored_rem = None if stored is None else _r4(float(stored) - pressure)
    active_rem = None if active is None else _r4(float(active) - pressure)
    objective = arch.get("objective")
    primary = stored_rem
    if objective == "active-bytes/token" and active_rem is not None:
        primary = active_rem
    if explicit is not None:
        coerced = coerce_target_delta(explicit, policy)
        if coerced is not None:
            primary = _r4(coerced)
    large = (
        primary is not None
        and primary >= large_delta_threshold(policy)
    )
    return {
        "primary": primary,
        "stored_minus_pressure": stored_rem,
        "active_minus_pressure": active_rem,
        "pressure_bpw": round(pressure, 4),
        "zones": {
            k: zones[k] for k in (
                "reachable_or_explained", "pressure", "aggressive",
                "structural_correction_tier", "frontier_full_accounting",
            ) if k in zones
        },
        "objective": objective,
        "large": large,
        "threshold": large_delta_threshold(policy),
        "_evidence": "DERIVED (best measured bpw − policy.pressure)",
    }


def build_packet(
    patient: Any,
    packet: Any,
    receipts: Any,
    rulebase: Any,
    transfer: Any,
    negatives: Any,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a FRONTIER_NOVELTY_PACKET from campaign artifacts."""
    pol = policy if policy is not None else load_policy()
    pkt = _as_mapping(packet)
    recs = [r for r in _as_list(receipts) if isinstance(r, dict)]
    rb = _as_mapping(rulebase, RULEBASE_PATH)
    tr = _as_mapping(transfer, TRANSFER_PATH)
    neg = _as_mapping(negatives, NEGATIVE_PATH)

    oxx = _oxx_of(patient, pkt)
    arch = _arch_of(patient, pkt)
    arch["objective"] = _policy_objective(arch.get("kind"), pol)
    anchor = _best_conventional_anchor(pkt, recs)
    fails, locs = _aggressive_failures(pkt, recs)
    decomp = _stored_active_physical(pkt, recs, pol)
    prims = _native_primitives(pkt, arch)
    negs = _negative_rules(neg, tr, rb, oxx, arch)
    remaining = _remaining_target_delta(decomp, pol, arch)

    out = {
        "schema": SCHEMA_PACKET,
        "oxx": oxx,
        "arch": arch,
        "best_conventional_anchor": anchor,
        "aggressive_failures": fails,
        "failure_localization": locs,
        "stored_active_physical": decomp,
        "native_primitives": prims,
        "negative_rules": negs,
        "remaining_target_delta": remaining,
        "lanes": list(LANES),
        "escalation_order_on_stall": list(pol.get("escalation_order_on_stall") or []),
        "conventionality_gate": pol.get("conventionality_gate") or {},
        "proposal_required_fields": list(REQUIRED_PROPOSAL_FIELDS),
        "_evidence": "DERIVED (packet + receipts + rulebase + transfer + negatives)",
        "_doc": (
            "Grok novelty input. Hypotheses only; scripts measure. "
            "Do not launch from this module."
        ),
    }
    if isinstance(patient, dict):
        out["patient_stages"] = sorted(_stages_from(patient, None))
    return out


# ---------------------------------------------------------------------------
# lane contracts
# ---------------------------------------------------------------------------

def _unfenced_lines(text: str) -> list[str]:
    lines: list[str] = []
    fenced = False
    for line in text.splitlines():
        if re.match(r"^\s*```", line):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def contract_has_write_and_verify(text: str) -> bool:
    lines = _unfenced_lines(text)
    has_write = any(re.match(r"^\s*WRITE\b", ln) for ln in lines)
    has_verify = any(re.match(r"^\s*VERIFY\b", ln) for ln in lines)
    return has_write and has_verify


def _scope_block(writes: list[str], reads: list[str],
                 verify_path: str, verify_cmd: str) -> str:
    lines = ["## SCOPE"]
    for w in writes:
        lines.append(f"WRITE {w}")
    lines.append("READ " + ", ".join(reads))
    lines.append(
        f"VERIFY {verify_path} by running the unfenced command below; "
        "must pass, exit 0."
    )
    lines.append(verify_cmd)
    lines.append("Do not modify tools/odyssey_ctl.py")
    lines.append("Do not modify tools/odyssey_patient_runner.py")
    lines.append("Do not touch Genesis state or tools/odyssey/.")
    return "\n".join(lines)


def _lane_contract_text(packet: dict[str, Any], lane: str) -> str:
    oxx = packet.get("oxx") or "UNKNOWN"
    arch = packet.get("arch") if isinstance(packet.get("arch"), dict) else {}
    anchor = packet.get("best_conventional_anchor") or {}
    remaining = packet.get("remaining_target_delta") or {}
    decomp = packet.get("stored_active_physical") or {}
    prims = packet.get("native_primitives") or {}
    fails = packet.get("aggressive_failures") or []
    brief = LANE_BRIEFS.get(lane) or {
        "question": f"NONCONVENTIONAL {lane} hypotheses for {oxx}.",
        "search": "nonconventional mechanisms only",
        "negatives": "NEGATIVE_SCIENCE.json",
    }
    kind = arch.get("kind") or "unknown"
    report_rel = f"receipts/odyssey-i/{oxx}_NOVELTY_{lane}.json"
    fields = ", ".join(REQUIRED_PROPOSAL_FIELDS)
    fail_specs = ", ".join(
        str(f.get("spec")) for f in fails if isinstance(f, dict) and f.get("spec")
    ) or "(none recorded)"
    loc = packet.get("failure_localization") or []
    stored_bpw = (decomp.get("stored") or {}).get("bpw")
    active_bpw = (decomp.get("active") or {}).get("bpw")
    primary = remaining.get("primary")
    pressure = remaining.get("pressure_bpw")
    nx = prims.get("best_preliminary_nx") or prims.get("declared") or "unspecified"
    writes = [
        "workspace/campaign/odyssey/contracts/auto/",
        "receipts/odyssey-i/",
        "workspace/campaign/odyssey/candidate_families.json",
    ]
    reads = [
        "workspace/campaign/odyssey/ODYSSEY_POLICY.json",
        "workspace/campaign/odyssey/GRAVITY_RULEBASE.json",
        "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
        "workspace/campaign/odyssey/TRANSFER_MATRIX.json",
    ]
    req_py = (
        "python3 -c \"import json,pathlib,sys; "
        f"p=pathlib.Path('{report_rel}'); "
        "d=json.loads(p.read_text()); "
        "ps=d.get('proposals') if isinstance(d.get('proposals'), list) else [d]; "
        "req=(" + ",".join(repr(f) for f in REQUIRED_PROPOSAL_FIELDS) + "); "
        "missing=[k for pr in ps if isinstance(pr, dict) for k in req if k not in pr]; "
        "sys.exit(0 if p.is_file() and ps and not missing else 1)\""
    )
    scope = _scope_block(
        writes, reads, "receipts/odyssey-i/", req_py,
    )
    loc_s = json.dumps(loc, ensure_ascii=False)[:400]
    return f"""# DELEGATION — {oxx} FRONTIER NOVELTY / {lane}

Patient {oxx} ({kind}; {arch.get('arch') or arch.get('source_repo') or 'unknown'}).
Repo: `/Users/scammermike/Downloads/hawking`. Branch odyssey-i.
Grok novelty lane. Hypotheses only; scripts measure (bible §55 / §13).
Opus is not involved. Do not launch further Grok from this lane.

This patient stalled on conventional / aggressive-quant families with a
LARGE remaining target delta (primary={primary}, pressure={pressure} bpw,
stored_bpw={stored_bpw}, active_bpw={active_bpw}). Best conventional
anchor: spec={anchor.get('spec')} class={anchor.get('candidate_class')}
complete_bpw={anchor.get('complete_bpw') or anchor.get('stored_bpw')}.
Aggressive failures: {fail_specs}. Localization: {loc_s}.
Native / NX so far: {nx}.

## Read first
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
READ workspace/campaign/odyssey/GRAVITY_RULEBASE.json
READ workspace/campaign/odyssey/NEGATIVE_SCIENCE.json
READ workspace/campaign/odyssey/TRANSFER_MATRIX.json

Query NEGATIVE_SCIENCE before every expensive hypothesis (bible §19).
A kill stays dead unless this architecture invalidates the old premise.

## QUESTION
{brief['question']}

## SEARCH SPACE (nonconventional only)
{brief['search']}

Live negatives that already constrain this lane:
{brief['negatives']}

Presumptively CONVENTIONAL (do not re-propose as a frontier):
uniform-quant, affine-quant, symmetric-quant, per-group-scale-bias,
ordinary-mixed-precision, gguf-mlx-like-per-weight, global integer
bit-width retreat. CONVENTIONAL_ANCHOR != REPRESENTATION_FRONTIER.

## BUILD
Write `{report_rel}` (schema {SCHEMA_LANE_REPORT}) with one or more
proposals. Each proposal REQUIRES these keys: {fields}.

Also required on each proposal:
- family_addition: a candidate_families row (mechanism, conventionality=STRUCTURAL,
  cheapest_falsifier, expected_win, doctor_risk, applicability)
- runner_spec if the hypothesis can be expressed in the runner grammar
  (q<bits>-g<group>-experts, mixed-q2q3-experts, +correction, tiers)
- info_gain (0-10) and cost (wall/gpu relative 1-10)
- reopen_if, evidence class (HYPOTHESIS), NEXT_BOTTLENECK

conventionality of a novelty proposal is STRUCTURAL (or ACTIVE_NX), never
CONVENTIONAL_ANCHOR. Complete byte accounting must name payload+scales+
biases+tables+offsets+correction+tier/router-metadata+alignment+headers+
repr-attributable-state (policy.accounting_gates.no_fake_density).

Do not edit tools/odyssey_ctl.py or tools/odyssey_patient_runner.py.
Do not claim a Hawking NX win. Do not measure TPS (physics is programmatic).

## ACCEPTANCE
- `{report_rel}` exists, schema {SCHEMA_LANE_REPORT}, each proposal has
  {fields}. Must pass, exit 0.
- Zero conventional-family re-proposals tagged as FRONTIER.

{scope}
"""


def render_lane_contracts(
    packet: dict[str, Any],
    auto_dir: Path | None = None,
) -> list[str]:
    """Write SG-valid Grok contracts for each novelty lane. Does not launch."""
    dest_dir = Path(auto_dir) if auto_dir is not None else AUTO_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    oxx = (packet.get("oxx") or "unknown").lower()
    lanes = packet.get("lanes") or list(LANES)
    paths: list[str] = []
    for lane in lanes:
        if str(lane) not in LANES:
            continue
        text = _lane_contract_text(packet, str(lane))
        dest = dest_dir / f"{oxx}_novelty-{lane}.md"
        dest.write_text(text)
        paths.append(str(dest))
    return paths


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------

def _first_present(blob: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if blob.get(k) is not None:
            return blob[k]
    return None


def _normalize_proposal(raw: Any, *, oxx: str | None, lane: str | None,
                        task: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"mechanism": raw}
    if not isinstance(raw, dict):
        return None
    # Nested
    if "proposals" in raw and isinstance(raw["proposals"], list):
        return None  # caller unwraps
    out: dict[str, Any] = {
        "schema": SCHEMA_PROPOSAL,
        "oxx": raw.get("oxx") or oxx,
        "lane": raw.get("lane") or lane,
        "task": task,
        "_evidence": raw.get("_evidence") or raw.get("evidence") or "HYPOTHESIS",
    }
    for field, aliases in PROPOSAL_ALIASES.items():
        out[field] = _first_present(raw, aliases)
    if not out.get("mechanism"):
        return None
    try:
        out["info_gain"] = float(
            raw.get("info_gain", raw.get("info", raw.get("value", 5.0)))
        )
    except (TypeError, ValueError):
        out["info_gain"] = 5.0
    try:
        out["cost"] = float(raw.get("cost", raw.get("wall_cost", 3.0)))
    except (TypeError, ValueError):
        out["cost"] = 3.0
    if out["cost"] <= 0:
        out["cost"] = 0.01
    out["rank_score"] = out["info_gain"] / out["cost"]
    spec = raw.get("runner_spec") or raw.get("spec") or raw.get("gravity_spec")
    if spec and RUNNER_SPEC_RE.match(str(spec).strip()):
        out["runner_spec"] = str(spec).strip()
    elif spec:
        out["runner_spec"] = str(spec).strip()
        out["runner_spec_note"] = "not matching published grammar; keep as family_addition"
    else:
        out["runner_spec"] = None
    fam = raw.get("family_addition") if isinstance(raw.get("family_addition"), dict) else {}
    out["family_addition"] = {
        "schema": SCHEMA_FAMILY_ADD,
        "id": fam.get("id") or _norm_token(out["mechanism"]),
        "mechanism": fam.get("mechanism") or out["mechanism"],
        "conventionality": fam.get("conventionality") or "STRUCTURAL",
        "cheapest_falsifier": (
            fam.get("cheapest_falsifier") or out.get("cheapest_falsifier")
        ),
        "expected_win": fam.get("expected_win") or raw.get("expected_win"),
        "doctor_risk": fam.get("doctor_risk") or out.get("doctor_risk"),
        "applicability": fam.get("applicability") or out.get("applicability_class"),
        "runner_spec": fam.get("runner_spec") or out.get("runner_spec"),
    }
    out["reopen_if"] = raw.get("reopen_if")
    out["complete_byte_accounting"] = (
        out.get("complete_byte_accounting") or raw.get("accounting")
    )
    return out


def _extract_json_blobs(text: str) -> list[Any]:
    blobs: list[Any] = []
    for rx in (JSON_OBJECT_FENCE_RE, JSON_ARRAY_FENCE_RE):
        for m in rx.finditer(text):
            try:
                blobs.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                continue
    if not blobs:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                blobs.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return blobs


def _proposals_from_blob(blob: Any, *, oxx: str | None, lane: str | None,
                         task: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(blob, list):
        for item in blob:
            out.extend(_proposals_from_blob(item, oxx=oxx, lane=lane, task=task))
        return out
    if not isinstance(blob, dict):
        return out
    if isinstance(blob.get("proposals"), list):
        lane2 = blob.get("lane") or lane
        oxx2 = blob.get("oxx") or oxx
        for item in blob["proposals"]:
            p = _normalize_proposal(item, oxx=oxx2, lane=lane2, task=task)
            if p:
                out.append(p)
        return out
    p = _normalize_proposal(blob, oxx=oxx, lane=lane, task=task)
    if p:
        out.append(p)
    return out


def _resolve_task(task_id: Any) -> tuple[str | None, str, str | None]:
    """Return (task_name, report_text, oxx)."""
    if isinstance(task_id, dict):
        oxx = None
        if task_id.get("oxx"):
            oxx = _oxx_of(str(task_id["oxx"]))
        text = ""
        if task_id.get("report_text"):
            text = str(task_id["report_text"])
        elif task_id.get("mechanism") or task_id.get("proposals"):
            text = json.dumps(task_id)
        return task_id.get("task") or task_id.get("id"), text, oxx
    raw = str(task_id)
    p = Path(raw)
    if p.is_file():
        return p.stem, p.read_text(errors="replace"), None
    if p.is_dir():
        report = p / "grok-report.md"
        alt = p / "report.md"
        src = report if report.is_file() else alt
        text = src.read_text(errors="replace") if src.is_file() else ""
        return p.name, text, None
    grok = GROK_TASKS / raw
    if grok.is_dir():
        report = grok / "grok-report.md"
        text = report.read_text(errors="replace") if report.is_file() else ""
        return raw, text, None
    # bare name under AUTO? last resort empty
    return raw, "", None


def _lane_from_task(name: str | None) -> str | None:
    if not name:
        return None
    for lane in LANES:
        if lane in name:
            return lane
    return None


def _experiment_contract_text(prop: dict[str, Any]) -> str:
    oxx = prop.get("oxx") or "UNKNOWN"
    mech = prop.get("mechanism") or "unknown"
    spec = prop.get("runner_spec")
    fam = prop.get("family_addition") or {}
    slug = _norm_token(mech)[:40] or "mech"
    receipt = f"receipts/odyssey-i/{oxx}_NOVELTY_EXP_{slug}.json"
    if spec:
        extra = f"--gravity {spec}"
        verify_cmd = (
            "python3 -c \"import json,pathlib,sys; "
            f"p=pathlib.Path('{receipt}'); "
            "d=json.loads(p.read_text()); "
            "sys.exit(0 if d.get('spec') or d.get('family_addition') else 1)\""
        )
        build = (
            f"The runner may not yet accept `{spec}`. If the spec is in "
            f"the published grammar, run tools/odyssey_patient_runner.py "
            f"{extra} and write {receipt}. Otherwise merge family_addition "
            f"into workspace/campaign/odyssey/candidate_families.json and "
            f"write {receipt} recording the merge (do not edit the runner)."
        )
    else:
        verify_cmd = (
            "python3 -c \"import json,pathlib,sys; "
            f"p=pathlib.Path('workspace/campaign/odyssey/candidate_families.json'); "
            "d=json.loads(p.read_text()); "
            "sys.exit(0 if d else 1)\""
        )
        build = (
            f"Merge this family_addition into "
            f"workspace/campaign/odyssey/candidate_families.json: "
            f"{json.dumps(fam, ensure_ascii=False)}. "
            f"Write {receipt} with the merged row. Do not edit the runner."
        )
    writes = [
        "workspace/campaign/odyssey/candidate_families.json",
        "receipts/odyssey-i/",
        "workspace/campaign/odyssey/contracts/auto/",
    ]
    reads = [
        "workspace/campaign/odyssey/ODYSSEY_POLICY.json",
        "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
    ]
    scope = _scope_block(writes, reads, "receipts/odyssey-i/", verify_cmd)
    return f"""# DELEGATION — {oxx} NOVELTY EXPERIMENT / {mech}

Deterministic follow-up from a harvested Grok novelty proposal.
Scripts measure; do not re-ask a model for arithmetic (bible §55).
Repo: `/Users/scammermike/Downloads/hawking`. Branch odyssey-i.

Mechanism: {mech}
Complete byte accounting: {prop.get('complete_byte_accounting')}
Cheapest falsifier: {prop.get('cheapest_falsifier')}
Execution path: {prop.get('execution_path')}
Kernel implications: {prop.get('kernel_implications')}
Applicability: {prop.get('applicability_class')}
Doctor risk: {prop.get('doctor_risk')}
Runner spec: {spec}
info_gain={prop.get('info_gain')} cost={prop.get('cost')} rank={prop.get('rank_score')}

## Read first
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
READ workspace/campaign/odyssey/NEGATIVE_SCIENCE.json

## BUILD
{build}

## ACCEPTANCE
- Experiment receipt or candidate_families row exists. Must pass, exit 0.

{scope}
"""


def harvest_proposals(
    task_ids: Any,
    *,
    auto_dir: Path | None = None,
    write_contracts: bool = True,
) -> list[dict[str, Any]]:
    """Parse Grok novelty reports; dedup; rank by info-gain/cost.

    Turns each survivor into a family_addition and/or runner_spec plus an
    optional deterministic experiment contract. Does not launch Grok.
    """
    if task_ids is None:
        items: list[Any] = []
    elif isinstance(task_ids, (str, dict, Path)):
        items = [task_ids]
    else:
        items = list(task_ids)

    collected: list[dict[str, Any]] = []
    for tid in items:
        name, text, oxx_hint = _resolve_task(tid)
        oxx = oxx_hint
        if oxx is None and name:
            m = SLUG_OXX_RE.search(name) or OXX_RE.search(name)
            if m:
                oxx = f"O{m.group(1)}"
        lane = _lane_from_task(name)
        if isinstance(tid, dict) and not text:
            collected.extend(
                _proposals_from_blob(tid, oxx=oxx, lane=lane, task=name)
            )
            continue
        if not text:
            continue
        blobs = _extract_json_blobs(text)
        if not blobs:
            # last chance: treat whole file as one mechanism paragraph
            continue
        for blob in blobs:
            collected.extend(
                _proposals_from_blob(blob, oxx=oxx, lane=lane, task=name)
            )

    # Dedup by normalized mechanism; keep highest rank_score.
    best: dict[str, dict[str, Any]] = {}
    for prop in collected:
        key = _norm_token(prop.get("mechanism"))
        if not key:
            continue
        prev = best.get(key)
        if prev is None or float(prop.get("rank_score") or 0) > float(
            prev.get("rank_score") or 0
        ):
            best[key] = prop
    ranked = sorted(
        best.values(),
        key=lambda p: (-float(p.get("rank_score") or 0), str(p.get("mechanism"))),
    )
    dest_dir = Path(auto_dir) if auto_dir is not None else AUTO_DIR
    if write_contracts:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for prop in ranked:
            oxx = (prop.get("oxx") or "unk").lower()
            slug = _norm_token(prop.get("mechanism"))[:40] or "mech"
            path = dest_dir / f"{oxx}_novelty-exp-{slug}.md"
            path.write_text(_experiment_contract_text(prop))
            prop["experiment_contract"] = str(path)
    return ranked


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _sg_lint(contract: Path, repo: Path | None = None) -> tuple[bool, str]:
    if not LINT_JS.is_file():
        return True, "LINT_MISSING"
    node = str(NODE_BIN) if NODE_BIN.is_file() else "node"
    import subprocess
    r = subprocess.run(
        [node, str(LINT_JS), str(contract), str(repo or REPO)],
        capture_output=True, text=True,
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if out.startswith("ERROR"):
        return False, out
    return True, out


def _self_check() -> int:
    pol = load_policy()
    assert pol.get("escalation_order_on_stall"), "policy missing escalation_order_on_stall"
    order = [str(x).split()[0] for x in pol["escalation_order_on_stall"]]
    assert "deterministic_search" in order[0] or order[0].startswith("deterministic")
    assert any("rule_transfer" in x or x.startswith("rule") for x in order[:2])
    assert any("grok_novelty" in x or "novelty" in x for x in order)

    stalled_patient = {
        "oxx": "O005",
        "kind": "moe",
        "stages_completed": ["deterministic_search", "rule_transfer"],
    }
    conv_families = [
        {"id": "q3-g32-experts", "conventionality": "CONVENTIONAL"},
        {
            "id": "q2-g32-experts",
            "conventionality": "AGGRESSIVE_QUANT",
            "status": "FAIL",
        },
    ]
    assert should_escalate(
        stalled_patient, "CONVENTIONAL_ANCHOR", 1.5, conv_families, pol,
    ) is True, "expected escalate on conventional stall + large delta"
    assert should_escalate(
        stalled_patient, "STRUCTURAL_GRAVITY", 1.5, conv_families, pol,
    ) is False, "STRUCTURAL survivor must not escalate"
    assert should_escalate(
        stalled_patient, "CONVENTIONAL_ANCHOR", 0.1, conv_families, pol,
    ) is False, "small target_delta must not escalate"
    assert should_escalate(
        {"oxx": "O005"}, "CONVENTIONAL_ANCHOR", 1.5, conv_families, pol,
    ) is False, "deterministic search not exhausted"
    assert should_escalate(
        stalled_patient, "CONVENTIONAL_ANCHOR", 1.5, [], pol,
    ) is False, "no families tried"
    assert should_escalate(
        stalled_patient,
        "CONVENTIONAL_ANCHOR",
        1.5,
        [
            {
                "id": "mixed-q2q3-experts",
                "conventionality": "STRUCTURAL",
                "status": "CANDIDATE_PASS",
            }
        ],
        pol,
    ) is False, "STRUCTURAL family survivor"
    # exhausted via families_tried markers + numeric delta dict
    assert should_escalate(
        {"oxx": "O001"},
        "AGGRESSIVE_QUANT",
        {"best_bpw": 4.0, "target_bpw": 2.5},
        {
            "families": ["q4-g64-attn-mlp", "q2-g64-attn-mlp"],
            "deterministic_search": True,
            "rule_transfer": True,
        },
        pol,
    ) is True

    packet_in = {
        "oxx": "O005",
        "class": "moe",
        "identity": {"source_repo": "Qwen/Qwen3-30B-A3B", "model_family": "Qwen3-MoE"},
        "architecture": {
            "kind": "moe",
            "arch": "Qwen3MoeForCausalLM",
            "total_params": 30532122624,
            "active_params_per_token": 3353032704,
            "active_pct": 11.0,
            "layers": 48,
            "experts": 128,
            "experts_per_tok": 8,
            "modality": "text",
        },
        "representation": {
            "stored_bpw": 4.0253,
            "best_stored_bpw_eq": 4.0253,
            "active_bpw_eq": 4.2305,
            "organs_bytes_GB": {"expert": 57.98, "attn": 1.81, "router": 0.03},
            "source_bytes": 61064245248,
        },
        "execution": {
            "baseline_tps": 35.391,
            "ttft": 0.1642,
            "not_base_true_tps": True,
            "label": "SPECIMEN",
        },
        "gravity": {
            "tried_mechanisms": ["q3-g32-experts"],
            "wins": [{
                "spec": "q3-g32-experts",
                "stored_bpw": 4.0253,
                "active_bpw": 4.2305,
                "verdict": "CANDIDATE_PASS",
                "candidate_class": "CONVENTIONAL_ANCHOR",
                "receipt": "receipts/odyssey-i/O005_GRAVITY_q3-g32-experts.json",
            }],
            "kills": [{
                "spec": "q2-g32-experts",
                "verdict": "DEGRADED",
                "candidate_class": "AGGRESSIVE_QUANT",
                "failure_localization": {
                    "organ": "router",
                    "repair": "protect router precision",
                },
            }],
            "last": {
                "spec": "q3-g32-experts",
                "stored_bpw": 4.0253,
                "active_bpw": 4.2305,
            },
        },
        "nx": {
            "primitive_set": None,
            "best_preliminary_nx": "active-expert-gather",
            "not_hawking_nx_win": True,
            "ratio_selected_over_full": 0.0625,
        },
    }
    receipts = [
        {
            "schema": "odyssey.patient.gravity.v1",
            "oxx": "O005",
            "spec": "q3-g32-experts",
            "stored_bpw": 4.0253,
            "active_bpw": 4.2305,
            "verdict": "CANDIDATE_PASS",
            "candidate_class": "CONVENTIONAL_ANCHOR",
            "delta_hits": 0,
        },
        {
            "schema": "odyssey.patient.gravity.v1",
            "oxx": "O005",
            "spec": "q2-g32-experts",
            "stored_bpw": 3.1,
            "verdict": "DEGRADED",
            "candidate_class": "AGGRESSIVE_QUANT",
            "delta_hits": -4,
            "failure_localization": {"organ": "gate", "repair": "protect gate/up"},
        },
    ]
    built = build_packet(
        stalled_patient, packet_in, receipts,
        rulebase=None, transfer=None, negatives=None, policy=pol,
    )
    missing = [k for k in REQUIRED_PACKET_KEYS if k not in built]
    assert not missing, f"build_packet missing keys: {missing}"
    for k in REQUIRED_PACKET_KEYS:
        assert built.get(k) is not None, k
    assert built["schema"] == SCHEMA_PACKET
    assert built["oxx"] == "O005"
    assert built["arch"].get("kind") == "moe"
    assert built["best_conventional_anchor"].get("spec") == "q3-g32-experts"
    assert built["aggressive_failures"]
    assert built["remaining_target_delta"].get("large") is True
    assert built["negative_rules"], "negatives should load from disk"

    AUTO_DIR.mkdir(parents=True, exist_ok=True)
    paths = render_lane_contracts(built, auto_dir=AUTO_DIR)
    assert len(paths) >= 3, paths
    assert len(paths) == len(LANES), paths
    for p in paths:
        text = Path(p).read_text()
        assert contract_has_write_and_verify(text), f"missing unfenced WRITE/VERIFY: {p}"
        assert "must pass" in text.lower() or "exit 0" in text
        assert "Do not modify tools/odyssey_ctl.py" in text
        for field in REQUIRED_PROPOSAL_FIELDS:
            assert field in text, (p, field)
        ok, msg = _sg_lint(Path(p))
        assert ok, (p, msg)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        t1 = td_p / "odyssey-o005-novelty-representation"
        t2 = td_p / "odyssey-o005-novelty-kernel"
        t1.mkdir()
        t2.mkdir()
        (t1 / "grok-report.md").write_text(
            "**Completion report**\n\n"
            "```json\n"
            + json.dumps({
                "schema": SCHEMA_LANE_REPORT,
                "oxx": "O005",
                "lane": "representation",
                "proposals": [
                    {
                        "mechanism": "base+correction-route-conditioned",
                        "complete_byte_accounting": {
                            "payload": 2.1, "scales": 0.1, "correction": 0.05,
                        },
                        "cheapest_falsifier": "ablate correction tier; Doctor delta",
                        "execution_path": "native gather + residual add",
                        "kernel_implications": "needs residual-add epilogue",
                        "applicability_class": "moe",
                        "doctor_risk": "medium",
                        "info_gain": 8,
                        "cost": 2,
                        "runner_spec": "q2-g32-experts+correction",
                    },
                    {
                        "mechanism": "matryoshka-T0-T2",
                        "complete_byte_accounting": {"payload": 1.8, "tiers": 0.3},
                        "cheapest_falsifier": "drop T2; measure Doctor",
                        "execution_path": "tiered fetch",
                        "kernel_implications": "tier pointer table",
                        "applicability_class": "moe",
                        "doctor_risk": "high",
                        "info_gain": 6,
                        "cost": 4,
                        "runner_spec": "matryoshka-T0",
                    },
                ],
            })
            + "\n```\n"
        )
        (t2 / "grok-report.md").write_text(
            "```json\n"
            + json.dumps({
                "oxx": "O005",
                "lane": "kernel",
                "proposals": [
                    {
                        "mechanism": "base+correction-route-conditioned",
                        "complete_byte_accounting": "dup should lose to higher score",
                        "cheapest_falsifier": "weaker copy",
                        "execution_path": "gather",
                        "kernel_implications": "same",
                        "applicability_class": "moe",
                        "doctor_risk": "medium",
                        "info_gain": 3,
                        "cost": 5,
                    },
                    {
                        "mechanism": "native-expert-gather-kernel",
                        "complete_byte_accounting": {
                            "active_bytes_per_token": 3.6e9,
                        },
                        "cheapest_falsifier": "compare mlx gather vs dense wall",
                        "execution_path": "Hawking native gather",
                        "kernel_implications": "1.15x relative if bandwidth-bound",
                        "applicability_class": "moe",
                        "doctor_risk": "low",
                        "info_gain": 9,
                        "cost": 3,
                    },
                ],
            })
            + "\n```\n"
        )
        harvested = harvest_proposals(
            [str(t1), str(t2)], auto_dir=td_p / "auto", write_contracts=True,
        )
        mechs = [h["mechanism"] for h in harvested]
        assert len(mechs) == 3, mechs
        assert mechs.count("base+correction-route-conditioned") == 1
        assert harvested[0]["rank_score"] >= harvested[-1]["rank_score"]
        for h in harvested:
            for field in REQUIRED_PROPOSAL_FIELDS:
                assert h.get(field) is not None, (h.get("mechanism"), field)
            assert h.get("family_addition")
            assert Path(h["experiment_contract"]).is_file()
            exp = Path(h["experiment_contract"]).read_text()
            assert contract_has_write_and_verify(exp)
        # inline dict path
        inline = harvest_proposals(
            [{
                "oxx": "O001",
                "mechanism": "ssm-state-codec",
                "complete_byte_accounting": {"state": 1.0},
                "cheapest_falsifier": "swap SSM state to bf16",
                "execution_path": "state residency",
                "kernel_implications": "scan kernel unchanged",
                "applicability_class": "hybrid",
                "doctor_risk": "low",
                "info_gain": 4,
                "cost": 1,
            }],
            write_contracts=False,
        )
        assert len(inline) == 1 and inline[0]["mechanism"] == "ssm-state-codec"

    print("self-check ok")
    print(f"  should_escalate true/false: ok")
    print(f"  build_packet keys: {list(REQUIRED_PACKET_KEYS)}")
    print(f"  rendered {len(paths)} contracts under {AUTO_DIR}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-check", action="store_true",
                    help="synthetic escalate/packet/render/harvest checks")
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
