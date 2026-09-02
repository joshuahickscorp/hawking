"""H-ROADMAP §9.2 three zeros, as checks that can FAIL.

ZERO STORAGE:              can this object cease to be stored independently?
ZERO INDEPENDENT INFORMATION: can this object be derived/shared/generated?
ZERO EXECUTION:            can this operation cease to execute on the critical path?

A low-BPW result that fails all three is ordinary quantization. PASS requires
positive architectural or receipt evidence; FAIL requires positive evidence
the zero is unavailable; UNKNOWN is for a missing field, never a silent healthy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ZERO_STORAGE = "ZERO_STORAGE"
ZERO_INDEPENDENT_INFORMATION = "ZERO_INDEPENDENT_INFORMATION"
ZERO_EXECUTION = "ZERO_EXECUTION"
THREE_ZEROS: tuple[str, ...] = (
    ZERO_STORAGE,
    ZERO_INDEPENDENT_INFORMATION,
    ZERO_EXECUTION,
)

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
VERDICTS = frozenset({PASS, FAIL, UNKNOWN})

EVIDENCE_TIER = "STATIC"

# Cross-expert cosine below this is evidence the experts are independent.
INDEPENDENT_COSINE = 0.05
# Rank-1 energy below this is evidence the spectrum does not collapse (not a factor).
FACTOR_RANK1_MIN = 0.5


@dataclass(frozen=True)
class ZeroResult:
    zero: str
    verdict: str
    confidence: float
    evidence: tuple[str, ...]
    uncertainty: str
    absent: tuple[str, ...] = ()
    evidence_tier: str = EVIDENCE_TIER

    def as_dict(self) -> dict[str, Any]:
        return {
            "zero": self.zero,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "uncertainty": self.uncertainty,
            "absent": list(self.absent),
            "evidence_tier": self.evidence_tier,
        }


def _cls(organ: Mapping[str, Any]) -> str:
    raw = organ.get("organ_class") or organ.get("organ") or organ.get("name") or ""
    return str(raw).lower()


def _bool(organ: Mapping[str, Any], key: str) -> bool | None:
    v = organ.get(key)
    if isinstance(v, bool):
        return v
    return None


def _int(organ: Mapping[str, Any], *keys: str) -> int | None:
    for k in keys:
        v = organ.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v == int(v):
            return int(v)
    return None


def _is_shared_expert(organ: Mapping[str, Any]) -> bool:
    return "shared_expert" in _cls(organ)


def _is_routed(organ: Mapping[str, Any]) -> bool:
    cls = _cls(organ)
    if _is_shared_expert(organ) or "router" in cls:
        return False
    n = _int(organ, "n_experts", "num_experts", "experts", "n_routed_experts")
    k = _int(organ, "experts_per_tok", "num_experts_per_tok", "top_k")
    if "routed" in cls or cls in {"expert_bank", "moe_expert", "experts"}:
        return True
    if n is not None and k is not None and n > k > 0 and "expert" in cls:
        return True
    return False


def _is_tied_pair(organ: Mapping[str, Any]) -> bool:
    cls = _cls(organ)
    tied = _bool(organ, "tie_word_embeddings")
    if tied is True and (
        "embed" in cls or "lm_head" in cls or cls in {"lmhead", "vocabulary"}
    ):
        return True
    if organ.get("derived_from") or organ.get("generated_from"):
        if "embed" in cls or "lm_head" in cls:
            return True
    return False


def _is_vision(organ: Mapping[str, Any]) -> bool:
    cls = _cls(organ)
    return "vision" in cls or cls in {"visual", "mm_projector"}


def _is_mtp_or_ngram(organ: Mapping[str, Any]) -> bool:
    cls = _cls(organ)
    return cls in {"mtp", "ngram_engine", "ngram"} or "mtp" in cls or "ngram" in cls


def check_zero_storage(organ: Mapping[str, Any]) -> ZeroResult:
    """Can this object cease to be stored independently?"""
    evidence: list[str] = []
    absent: list[str] = ["weight-body inspection (forbidden in this lane)"]
    tied = _bool(organ, "tie_word_embeddings")
    stored = _bool(organ, "stored_independently")
    generated = organ.get("generated_from")
    derived = organ.get("derived_from")

    if generated:
        evidence.append(f"generated_from={generated}")
        return ZeroResult(
            ZERO_STORAGE,
            PASS,
            0.85,
            tuple(evidence),
            "generator quality is unmeasured here; PASS is architectural",
            tuple(absent),
        )
    if _is_tied_pair(organ):
        evidence.append("tie_word_embeddings=true so embed/lm_head can share storage")
        return ZeroResult(
            ZERO_STORAGE,
            PASS,
            0.95,
            tuple(evidence),
            "a runtime that materializes both copies anyway would still store twice",
            tuple(absent),
        )
    if stored is False:
        evidence.append("stored_independently=false")
        return ZeroResult(
            ZERO_STORAGE,
            PASS,
            0.8,
            tuple(evidence),
            "the flag is architectural; a loader could still duplicate",
            tuple(absent),
        )
    if stored is True:
        evidence.append("stored_independently=true")
        if tied is False:
            evidence.append("tie_word_embeddings=false")
        return ZeroResult(
            ZERO_STORAGE,
            FAIL,
            0.9,
            tuple(evidence),
            "sharing/codebook/generation were not demonstrated; a later SHARE/GENERATE ask may reopen",
            tuple(absent),
        )
    if tied is False and (
        "embed" in _cls(organ) or "lm_head" in _cls(organ)
    ):
        evidence.append("untied embed/lm_head: two independently stored vocab matrices")
        return ZeroResult(
            ZERO_STORAGE,
            FAIL,
            0.85,
            tuple(evidence),
            "tying is a possible mutation; the current architecture does not have it",
            tuple(absent),
        )
    if _is_routed(organ) or _is_shared_expert(organ) or "mlp" in _cls(organ) or "attention" in _cls(organ):
        evidence.append(
            f"organ_class={_cls(organ)} is a stored weight body with no tying/generation declared"
        )
        return ZeroResult(
            ZERO_STORAGE,
            FAIL,
            0.7,
            tuple(evidence),
            "a shared basis or generator could still exist; SHARE/GENERATE have not been shown",
            tuple(absent),
        )
    if derived:
        evidence.append(f"derived_from={derived}")
        return ZeroResult(
            ZERO_STORAGE,
            PASS,
            0.75,
            tuple(evidence),
            "derivation may still serialize a copy",
            tuple(absent),
        )
    return ZeroResult(
        ZERO_STORAGE,
        UNKNOWN,
        0.2,
        ("no tying, generation, or stored_independently flag",),
        "architecture did not declare enough to decide",
        tuple(absent),
    )


def check_zero_independent_information(organ: Mapping[str, Any]) -> ZeroResult:
    """Can this object be derived, shared, or generated from another structure?"""
    evidence: list[str] = []
    absent: list[str] = ["parent-tensor SVD (forbidden in this lane)"]
    independent = _bool(organ, "independent_information")
    cosine = organ.get("cross_expert_cosine")
    rank1 = organ.get("rank_1_energy")
    generated = organ.get("generated_from")
    derived = organ.get("derived_from")

    if generated or derived or _is_tied_pair(organ):
        if generated:
            evidence.append(f"generated_from={generated}")
        if derived:
            evidence.append(f"derived_from={derived}")
        if _is_tied_pair(organ):
            evidence.append("tied embed/lm_head: one vocab, two views")
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            PASS,
            0.9,
            tuple(evidence),
            "the derived view may still carry a residual that this lane did not measure",
            tuple(absent),
        )
    if isinstance(cosine, (int, float)) and float(cosine) < INDEPENDENT_COSINE:
        evidence.append(
            f"cross_expert_cosine={float(cosine):.6f} < {INDEPENDENT_COSINE} "
            "(experts do not share a direction)"
        )
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            FAIL,
            0.9,
            tuple(evidence),
            "cosine is from a sampled screen, not the full bank",
            tuple(absent) + ("full-bank cosine",),
        )
    if isinstance(rank1, (int, float)) and float(rank1) < FACTOR_RANK1_MIN:
        evidence.append(
            f"rank_1_energy={float(rank1):.4f} < {FACTOR_RANK1_MIN} "
            "(spectrum does not collapse; not a free factor)"
        )
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            FAIL,
            0.8,
            tuple(evidence),
            "energy is from a sampled screen, not a full SVD",
            tuple(absent) + ("full singular spectrum",),
        )
    if independent is True:
        evidence.append("independent_information=true")
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            FAIL,
            0.85,
            tuple(evidence),
            "a rotation or codebook could still couple it; not demonstrated",
            tuple(absent),
        )
    if independent is False:
        evidence.append("independent_information=false")
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            PASS,
            0.75,
            tuple(evidence),
            "the flag is a prior, not a new measurement",
            tuple(absent),
        )
    if _is_routed(organ):
        evidence.append(
            "routed expert bank with no sharing/cosine evidence: treated as independent "
            "until a screen says otherwise"
        )
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            FAIL,
            0.55,
            tuple(evidence),
            "cross-expert structure was not measured in this diagnosis; a bank screen can reopen",
            tuple(absent) + ("cross-expert cosine",),
        )
    if _is_shared_expert(organ) or "mlp" in _cls(organ) or "attention" in _cls(organ):
        evidence.append(
            f"organ_class={_cls(organ)} is a unique weight body; no derivation declared"
        )
        return ZeroResult(
            ZERO_INDEPENDENT_INFORMATION,
            FAIL,
            0.6,
            tuple(evidence),
            "shared-basis / low-rank / generator remain legal hypotheses; they are not evidence",
            tuple(absent),
        )
    return ZeroResult(
        ZERO_INDEPENDENT_INFORMATION,
        UNKNOWN,
        0.2,
        ("no derivation, cosine, or independence flag",),
        "architecture did not declare enough to decide",
        tuple(absent),
    )


def check_zero_execution(organ: Mapping[str, Any]) -> ZeroResult:
    """Can this operation cease to execute on the critical path?"""
    evidence: list[str] = []
    absent: list[str] = ["activation trace (forbidden in this lane)"]
    every = _bool(organ, "executes_every_token")
    n = _int(organ, "n_experts", "num_experts", "experts", "n_routed_experts")
    k = _int(organ, "experts_per_tok", "num_experts_per_tok", "top_k")
    on_path = _bool(organ, "on_decode_critical_path")

    if _is_routed(organ) and n is not None and k is not None and n > k > 0:
        evidence.append(f"routed {k}/{n} experts per token; the other {n - k} do not fire")
        return ZeroResult(
            ZERO_EXECUTION,
            PASS,
            0.95,
            tuple(evidence),
            "a router that degenerates to all-experts would collapse this zero",
            tuple(absent),
        )
    if _is_routed(organ):
        evidence.append("organ is routed; expert-count vs top-k not both present")
        if n is None or k is None:
            return ZeroResult(
                ZERO_EXECUTION,
                PASS,
                0.6,
                tuple(evidence),
                "routing is declared but the skip fraction is unknown",
                tuple(absent) + ("num_experts", "experts_per_tok"),
            )
    if _is_mtp_or_ngram(organ):
        evidence.append(
            f"organ_class={_cls(organ)} is a decode-reduction organ; it can replace a full forward"
        )
        return ZeroResult(
            ZERO_EXECUTION,
            PASS,
            0.7,
            tuple(evidence),
            "whether the main forward actually skips is a runtime fact, ABSENT here",
            tuple(absent) + ("accepted-token skip rate",),
        )
    if _is_vision(organ):
        evidence.append("vision organ can cease to execute on a text-only decode")
        return ZeroResult(
            ZERO_EXECUTION,
            PASS,
            0.7,
            tuple(evidence),
            "a VL prompt still needs it; PASS is for the text-decode critical path",
            tuple(absent),
        )
    if every is False or on_path is False:
        evidence.append(
            "executes_every_token=false"
            if every is False
            else "on_decode_critical_path=false"
        )
        return ZeroResult(
            ZERO_EXECUTION,
            PASS,
            0.85,
            tuple(evidence),
            "a different serving path could still force it on",
            tuple(absent),
        )
    if _is_shared_expert(organ):
        evidence.append("shared_expert fires on every token (never averaged away)")
        return ZeroResult(
            ZERO_EXECUTION,
            FAIL,
            0.9,
            tuple(evidence),
            "a later router over the shared expert would reopen this; the architecture has none",
            tuple(absent),
        )
    if every is True or on_path is True:
        evidence.append(
            "executes_every_token=true"
            if every is True
            else "on_decode_critical_path=true"
        )
        return ZeroResult(
            ZERO_EXECUTION,
            FAIL,
            0.9,
            tuple(evidence),
            "conditional compute / MoD is untested on this organ",
            tuple(absent) + ("activation sparsity",),
        )
    cls = _cls(organ)
    if any(tok in cls for tok in ("mlp", "attention", "gqa", "embed", "lm_head", "norm", "deltanet")):
        evidence.append(f"organ_class={cls} is on the dense decode path by architecture")
        return ZeroResult(
            ZERO_EXECUTION,
            FAIL,
            0.75,
            tuple(evidence),
            "mixture-of-depths / sparsity could skip it; not declared",
            tuple(absent),
        )
    return ZeroResult(
        ZERO_EXECUTION,
        UNKNOWN,
        0.2,
        ("no routing, skip, or always-on flag",),
        "architecture did not declare enough to decide",
        tuple(absent),
    )


_CHECKERS = {
    ZERO_STORAGE: check_zero_storage,
    ZERO_INDEPENDENT_INFORMATION: check_zero_independent_information,
    ZERO_EXECUTION: check_zero_execution,
}


def check_three_zeros(organ: Mapping[str, Any]) -> dict[str, ZeroResult]:
    """Run all three zeros. Each check can FAIL. This is the gate symbol."""
    return {name: checker(organ) for name, checker in _CHECKERS.items()}


def zeros_as_dict(results: Mapping[str, ZeroResult]) -> dict[str, Any]:
    return {name: r.as_dict() for name, r in results.items()}


def ordinary_quantization(
    results: Mapping[str, ZeroResult] | Mapping[str, Mapping[str, Any]],
) -> bool:
    """True iff every zero FAILED. UNKNOWN does not count as a failure of the search.

    Roadmap 9.2: a low-BPW result that fails all three questions may merely
    be ordinary quantization.
    """
    if not results:
        return False
    verdicts: list[str] = []
    for name in THREE_ZEROS:
        row = results.get(name)
        if row is None:
            return False
        if isinstance(row, ZeroResult):
            verdicts.append(row.verdict)
        elif isinstance(row, Mapping):
            verdicts.append(str(row.get("verdict") or ""))
        else:
            return False
    return verdicts == [FAIL, FAIL, FAIL]


def organ_from_mapping(
    raw: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a receipt organ or config organ into the zeros input shape."""
    geom = dict(geometry or {})
    cls = str(raw.get("organ_class") or raw.get("organ") or raw.get("name") or "unknown")
    cls_l = cls.lower()
    inherit_bank = "routed" in cls_l or cls_l in {
        "expert_bank",
        "moe_expert",
        "experts",
        "routed_expert",
        "routed_experts",
    }
    n_experts = raw.get("n_experts") or raw.get("num_experts")
    top_k = (
        raw.get("experts_per_tok")
        or raw.get("num_experts_per_tok")
        or raw.get("top_k")
    )
    if inherit_bank:
        n_experts = (
            n_experts
            or geom.get("experts")
            or geom.get("num_experts")
            or geom.get("n_routed_experts")
        )
        top_k = (
            top_k
            or geom.get("top_k")
            or geom.get("num_experts_per_tok")
            or geom.get("experts_per_tok")
        )
    tied = raw.get("tie_word_embeddings")
    if tied is None:
        tied = geom.get("tie_word_embeddings")
    organ: dict[str, Any] = {
        "name": str(raw.get("name") or raw.get("organ") or cls),
        "organ_class": cls,
        "n_experts": n_experts if isinstance(n_experts, int) else None,
        "experts_per_tok": top_k if isinstance(top_k, int) else None,
        "tie_word_embeddings": tied if isinstance(tied, bool) else None,
        "stored_independently": raw.get("stored_independently"),
        "independent_information": raw.get("independent_information"),
        "executes_every_token": raw.get("executes_every_token"),
        "on_decode_critical_path": raw.get("on_decode_critical_path"),
        "generated_from": raw.get("generated_from"),
        "derived_from": raw.get("derived_from"),
        "cross_expert_cosine": raw.get("cross_expert_cosine"),
        "rank_1_energy": raw.get("rank_1_energy"),
        "params": raw.get("params"),
        "source_bytes": raw.get("source_bytes") or raw.get("source_representation_bytes"),
        "capability_sensitivity": raw.get("capability_sensitivity"),
        "rationale": raw.get("rationale"),
        "evidence_tier": EVIDENCE_TIER,
    }
    if organ["stored_independently"] is None:
        if organ["tie_word_embeddings"] is True and (
            "embed" in cls_l or "lm_head" in cls_l
        ):
            organ["stored_independently"] = False
            organ["derived_from"] = organ["derived_from"] or (
                "lm_head" if "embed" in cls_l else "embed_tokens"
            )
        elif organ["generated_from"]:
            organ["stored_independently"] = False
    if organ["executes_every_token"] is None:
        if "shared_expert" in cls_l:
            organ["executes_every_token"] = True
        elif "routed" in cls_l or cls_l in {"expert_bank", "moe_expert", "experts"}:
            organ["executes_every_token"] = False
        elif cls_l in {"mtp", "ngram_engine", "ngram", "vision", "vision_backbone"}:
            organ["executes_every_token"] = False
    return organ


def organs_from_doc(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Lift organs from a receipt. Empty means the caller should use a whole-artifact organ."""
    geometry = doc.get("geometry") if isinstance(doc.get("geometry"), Mapping) else {}
    text = doc.get("text_config") if isinstance(doc.get("text_config"), Mapping) else {}
    if text:
        geometry = {
            "experts": text.get("num_experts") or geometry.get("experts"),
            "num_experts": text.get("num_experts"),
            "top_k": text.get("num_experts_per_tok") or geometry.get("top_k"),
            "num_experts_per_tok": text.get("num_experts_per_tok"),
            "tie_word_embeddings": text.get("tie_word_embeddings", geometry.get("tie_word_embeddings")),
            **geometry,
        }
    raw = doc.get("organs")
    if isinstance(raw, list):
        return [organ_from_mapping(o, geometry=geometry) for o in raw if isinstance(o, Mapping)]
    if isinstance(raw, Mapping):
        out = []
        for name, body in raw.items():
            if isinstance(body, Mapping):
                row = dict(body)
                row.setdefault("name", name)
                row.setdefault("organ", name)
                out.append(organ_from_mapping(row, geometry=geometry))
        return out
    return []


def whole_artifact_organ(doc: Mapping[str, Any]) -> dict[str, Any]:
    """When a ledger names no organs, the ledger itself is still an object."""
    packed = doc.get("claim_boundary") if isinstance(doc.get("claim_boundary"), Mapping) else {}
    artifact_packed = packed.get("artifact_packed")
    return organ_from_mapping(
        {
            "name": "whole_artifact_ledger",
            "organ_class": "whole_artifact_ledger",
            "stored_independently": None,
            "independent_information": None,
            "executes_every_token": None,
            "rationale": (
                "ledger names no organs; zeros stay UNKNOWN rather than invented. "
                f"artifact_packed={artifact_packed}"
            ),
        }
    )


# --- fixtures the tests and ebpw selftest watch fail / pass -----------------

BROKEN_ORGAN: dict[str, Any] = {
    "name": "deliberately_broken_dense_mlp",
    "organ_class": "dense_mlp",
    "stored_independently": True,
    "independent_information": True,
    "executes_every_token": True,
    "on_decode_critical_path": True,
    "tie_word_embeddings": False,
    "n_experts": None,
    "experts_per_tok": None,
    "generated_from": None,
    "derived_from": None,
}

TIED_EMBED_ORGAN: dict[str, Any] = {
    "name": "tied_embed_tokens",
    "organ_class": "embed_tokens",
    "tie_word_embeddings": True,
    "stored_independently": False,
    "independent_information": False,
    "derived_from": "lm_head",
    "executes_every_token": True,
}

ROUTED_EXPERT_ORGAN: dict[str, Any] = {
    "name": "routed_expert_bank",
    "organ_class": "routed_expert",
    "n_experts": 512,
    "experts_per_tok": 10,
    "stored_independently": True,
    "independent_information": True,
    "executes_every_token": False,
    "tie_word_embeddings": False,
}
