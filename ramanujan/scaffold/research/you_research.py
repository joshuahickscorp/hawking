"""HIDE YOU cited-web / deep-research controller (Y5).

Offline, fixture-only research controller. No network. Web results never become
durable memory unless an explicit, recorded promotion occurs.

Four-way epistemic separation is enforced by type (not convention):

  RetrievedFact       -- extracted from a captured source document
  ModelInference      -- derived by the model; no source of its own
  UserProvided        -- the user said it
  UncertainHypothesis -- flagged low-confidence guess

A claim cannot change category without an explicit CategoryTransition record.
Every factual claim either links to evidence or is marked UNSUPPORTED — there is
no third state.

Claim–evidence graph reuses ramanujan.cognition.ResearchObjectGraph for
dependency + transitive refutation. This module does not invent a second graph
shape; it stores claim metadata beside the same nodes/depends_on/refute lattice.

Authority: NON_PRODUCTION_AUTHORITY for fixture runs.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from ramanujan.cognition import ResearchObjectGraph
from ramanujan.evidence import Tier
from ramanujan.layout import FIXTURES_ROOT

AUTHORITY = "NON_PRODUCTION_AUTHORITY"
FIXTURE_CORPUS_PATH = FIXTURES_ROOT / "you_research_corpus.json"

# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


class ResearchMode(Enum):
    QUICK_SEARCH = "quick_search"
    CITED_ANSWER = "cited_answer"
    DEEP_RESEARCH = "deep_research"
    WATCH_MONITOR = "watch_monitor"
    LITERATURE_REVIEW = "literature_review"
    COMPARISON = "comparison"
    FACT_AUDIT = "fact_audit"


# --------------------------------------------------------------------------
# Four-way epistemic categories (type-enforced)
# --------------------------------------------------------------------------


class ClaimCategory(Enum):
    """The four categories a claim may inhabit. Transitions are recorded."""

    RETRIEVED_FACT = "retrieved_fact"
    MODEL_INFERENCE = "model_inference"
    USER_PROVIDED = "user_provided"
    UNCERTAIN_HYPOTHESIS = "uncertain_hypothesis"


class EvidenceBinding(Enum):
    """Binary: linked to evidence, or explicitly unsupported. No third state."""

    LINKED = "linked"
    UNSUPPORTED = "unsupported"


class TransitionRefused(RuntimeError):
    """Raised when a claim tries to change category without a recorded transition."""


class EvidenceRequired(RuntimeError):
    """Raised when a factual claim would exist without evidence or unsupported mark."""


class PromotionRefused(RuntimeError):
    """Raised when web/research material tries to enter durable memory silently."""


# --------------------------------------------------------------------------
# Core records
# --------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class SourceQuality:
    authority: float
    recency: float
    independence: float
    reproducibility: float

    def grade(self) -> float:
        return (self.authority + self.recency + self.independence + self.reproducibility) / 4.0

    def band(self) -> str:
        g = self.grade()
        if g >= 0.75:
            return "high"
        if g >= 0.5:
            return "medium"
        return "low"


@dataclass(frozen=True)
class CapturedDocument:
    """A document captured into the research run (ephemeral unless promoted)."""

    id: str
    source_id: str
    title: str
    uri: str
    body: str
    published_at: str
    captured_at: str
    source_type: str
    quality: SourceQuality
    content_hash: str

    @staticmethod
    def from_fixture(raw: dict) -> "CapturedDocument":
        body = raw["body"]
        q = raw["quality"]
        h = hashlib.sha256(body.encode()).hexdigest()[:16]
        return CapturedDocument(
            id=raw["id"],
            source_id=raw["source_id"],
            title=raw["title"],
            uri=raw["uri"],
            body=body,
            published_at=raw["published_at"],
            captured_at=raw["captured_at"],
            source_type=raw["source_type"],
            quality=SourceQuality(
                authority=float(q["authority"]),
                recency=float(q["recency"]),
                independence=float(q["independence"]),
                reproducibility=float(q["reproducibility"]),
            ),
            content_hash=h,
        )


@dataclass(frozen=True)
class Citation:
    """A span of a captured document used as evidence for a claim."""

    id: str
    doc_id: str
    source_id: str
    excerpt: str
    char_range: tuple[int, int]
    content_hash: str


@dataclass(frozen=True)
class Claim:
    """A claim with fixed category. Mutating category requires transition_category()."""

    id: str
    text: str
    category: ClaimCategory
    evidence: EvidenceBinding
    citation_ids: tuple[str, ...]
    subject: str | None
    value: str | None
    confidence: float
    created_at: str
    # For RetrievedFact: the source doc that produced it (metadata; citations are evidence).
    origin_doc_id: str | None = None
    origin_source_id: str | None = None
    # Staleness is NOT baked in permanently; computed at answer time from sources.
    source_captured_at: str | None = None

    def is_factual(self) -> bool:
        """Facts and user statements are factual claims; inferences/hypotheses are not."""
        return self.category in (ClaimCategory.RETRIEVED_FACT, ClaimCategory.USER_PROVIDED)

    def assert_evidence_law(self) -> None:
        """Every factual claim links to evidence or is explicitly unsupported."""
        if not self.is_factual():
            return
        if self.evidence is EvidenceBinding.LINKED and not self.citation_ids:
            raise EvidenceRequired(
                f"claim {self.id!r} is LINKED but has no citation_ids"
            )
        if self.evidence is EvidenceBinding.UNSUPPORTED and self.citation_ids:
            raise EvidenceRequired(
                f"claim {self.id!r} is UNSUPPORTED but still carries citations"
            )
        # The only two states; enum already forbids a third.


@dataclass(frozen=True)
class CategoryTransition:
    claim_id: str
    from_category: ClaimCategory
    to_category: ClaimCategory
    reason: str
    at: str
    actor: str
    transition_id: str


@dataclass(frozen=True)
class Contradiction:
    """Two sources disagree. The controller surfaces both; it does not auto-resolve."""

    id: str
    subject: str
    claim_a_id: str
    claim_b_id: str
    source_a_id: str
    source_b_id: str
    value_a: str | None
    value_b: str | None
    # Explicit non-resolution: neither preferred by grade nor by recency.
    resolution: str = "unresolved_both_surfaced"
    preferred_claim_id: str | None = None  # always None by construction


@dataclass(frozen=True)
class FreshnessReport:
    claim_id: str
    source_captured_at: str | None
    answer_at: str
    age_seconds: float | None
    stale: bool
    threshold_seconds: float


@dataclass(frozen=True)
class MemoryPromotion:
    """Explicit, recorded promotion of research material into durable memory."""

    id: str
    claim_id: str
    run_id: str
    at: str
    actor: str
    reason: str


@dataclass
class ResearchCheckpoint:
    run_id: str
    seq: int
    at: str
    kind: str
    detail: dict


# --------------------------------------------------------------------------
# Durable memory (web results do NOT enter automatically)
# --------------------------------------------------------------------------


@dataclass
class DurableMemory:
    """Standing durable store. Research runs never write here without promote()."""

    entries: dict[str, dict] = field(default_factory=dict)
    promotions: list[MemoryPromotion] = field(default_factory=list)
    write_count: int = 0

    def snapshot(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self.entries.items()}

    def fingerprint(self) -> str:
        body = json.dumps(self.snapshot(), sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def promote(
        self,
        claim: Claim,
        run_id: str,
        actor: str,
        reason: str,
        *,
        at: str | None = None,
    ) -> MemoryPromotion:
        if not reason or not reason.strip():
            raise PromotionRefused("promotion requires an explicit non-empty reason")
        if not actor or not actor.strip():
            raise PromotionRefused("promotion requires an explicit actor")
        at = at or _iso()
        pid = hashlib.sha256(f"{claim.id}|{run_id}|{at}|{reason}".encode()).hexdigest()[:12]
        promo = MemoryPromotion(
            id=pid,
            claim_id=claim.id,
            run_id=run_id,
            at=at,
            actor=actor,
            reason=reason,
        )
        self.entries[claim.id] = {
            "claim_id": claim.id,
            "text": claim.text,
            "category": claim.category.value,
            "promoted_by": actor,
            "promotion_id": pid,
            "run_id": run_id,
            "reason": reason,
            "at": at,
        }
        self.promotions.append(promo)
        self.write_count += 1
        return promo


# --------------------------------------------------------------------------
# Claim–evidence graph (same shape as ResearchObjectGraph)
# --------------------------------------------------------------------------


class ClaimEvidenceGraph:
    """Claim–evidence graph that *is* a ResearchObjectGraph plus claim metadata.

    Relation to ramanujan.cognition.ResearchObjectGraph
    --------------------------------------------------
    This class owns a ResearchObjectGraph instance and uses it for every
    dependency edge and every refutation. Refuting a source (or claim) node
    propagates transitively through ResearchObjectGraph.refute — the same
    mechanism cognition tests already lock. Claim records live in `claims`;
    citations live in `citations`. There is no parallel depends_on map and no
    second propagation algorithm.

    Node kinds used in the underlying graph:
      claim, citation, source, document
    Dependency convention:
      claim  depends_on  citation(s)  depends_on  document/source
    so refuting a source undermines citations and the claims that rest on them.
    """

    def __init__(self) -> None:
        self.rog = ResearchObjectGraph()
        self.claims: dict[str, Claim] = {}
        self.citations: dict[str, Citation] = {}
        self.transitions: list[CategoryTransition] = []
        self.contradictions: list[Contradiction] = []

    # -- node registration -------------------------------------------------
    def add_source_node(self, source_id: str, **meta: Any) -> None:
        if source_id not in self.rog.nodes:
            self.rog.add(source_id, kind="source", tier=Tier.ASSERTED, **meta)

    def add_document_node(self, doc: CapturedDocument) -> None:
        if doc.id not in self.rog.nodes:
            self.rog.add(
                doc.id,
                kind="document",
                tier=Tier.ASSERTED,
                source_id=doc.source_id,
                content_hash=doc.content_hash,
            )
        self.add_source_node(doc.source_id, uri=doc.uri)
        # document depends on its source: refuting the source undermines the doc
        self.rog.depend(doc.id, doc.source_id)

    def add_citation(self, citation: Citation) -> None:
        if citation.id in self.citations:
            return
        self.citations[citation.id] = citation
        self.rog.add(
            citation.id,
            kind="citation",
            tier=Tier.EMPIRICALLY_SUPPORTED,
            doc_id=citation.doc_id,
        )
        if citation.doc_id in self.rog.nodes:
            self.rog.depend(citation.id, citation.doc_id)

    def add_claim(self, claim: Claim) -> None:
        claim.assert_evidence_law()
        if claim.id in self.claims:
            raise ValueError(f"claim {claim.id!r} already exists")
        self.claims[claim.id] = claim
        self.rog.add(
            claim.id,
            kind="claim",
            tier=Tier.ASSERTED,
            category=claim.category.value,
            evidence=claim.evidence.value,
        )
        for cid in claim.citation_ids:
            if cid not in self.citations:
                raise KeyError(f"citation {cid!r} must be registered before claim {claim.id!r}")
            self.rog.depend(claim.id, cid)

    def transition_category(
        self,
        claim_id: str,
        to: ClaimCategory,
        reason: str,
        actor: str,
        *,
        at: str | None = None,
    ) -> Claim:
        """Only legal way to change a claim's category. Records the transition."""
        if not reason or not reason.strip():
            raise TransitionRefused("category transition requires an explicit reason")
        old = self.claims[claim_id]
        if old.category is to:
            raise TransitionRefused(f"claim {claim_id!r} is already {to.value}")
        at = at or _iso()
        tid = hashlib.sha256(f"{claim_id}|{old.category.value}|{to.value}|{at}".encode()).hexdigest()[:12]
        rec = CategoryTransition(
            claim_id=claim_id,
            from_category=old.category,
            to_category=to,
            reason=reason,
            at=at,
            actor=actor,
            transition_id=tid,
        )
        self.transitions.append(rec)
        # frozen Claim → replace
        new = Claim(
            id=old.id,
            text=old.text,
            category=to,
            evidence=old.evidence,
            citation_ids=old.citation_ids,
            subject=old.subject,
            value=old.value,
            confidence=old.confidence,
            created_at=old.created_at,
            origin_doc_id=old.origin_doc_id,
            origin_source_id=old.origin_source_id,
            source_captured_at=old.source_captured_at,
        )
        # Inferences/hypotheses need not obey the factual evidence law the same way,
        # but if still factual after transition, re-check.
        new.assert_evidence_law()
        self.claims[claim_id] = new
        self.rog.nodes[claim_id]["category"] = to.value
        self.rog.nodes[claim_id]["last_transition"] = tid
        return new

    def mark_unsupported(self, claim_id: str, reason: str) -> Claim:
        """Explicit unsupported mark — the only alternative to linked evidence."""
        old = self.claims[claim_id]
        new = Claim(
            id=old.id,
            text=old.text,
            category=old.category,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=(),
            subject=old.subject,
            value=old.value,
            confidence=old.confidence,
            created_at=old.created_at,
            origin_doc_id=old.origin_doc_id,
            origin_source_id=old.origin_source_id,
            source_captured_at=old.source_captured_at,
        )
        new.assert_evidence_law()
        # drop citation dependencies in the ROG for this claim
        self.rog.depends_on[claim_id] = set()
        self.claims[claim_id] = new
        self.rog.nodes[claim_id]["evidence"] = EvidenceBinding.UNSUPPORTED.value
        self.rog.nodes[claim_id]["unsupported_reason"] = reason
        return new

    def refute_source(self, source_id: str, why: str) -> set[str]:
        """Refute a source; undermines documents, citations, and claims that rest on it."""
        return self.rog.refute(source_id, why)

    def refute_claim(self, claim_id: str, why: str) -> set[str]:
        return self.rog.refute(claim_id, why)

    def standing(self) -> dict[str, list[str]]:
        return self.rog.standing()

    def record_contradiction(self, c: Contradiction) -> None:
        if c.preferred_claim_id is not None:
            raise ValueError(
                "Contradiction must not prefer a side; preferred_claim_id must be None"
            )
        if c.resolution != "unresolved_both_surfaced":
            raise ValueError(
                "Contradiction must surface both sides unresolved; "
                f"got resolution={c.resolution!r}"
            )
        self.contradictions.append(c)


# --------------------------------------------------------------------------
# Retrieval interface + honest fake (no network, no embeddings)
# --------------------------------------------------------------------------


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[CapturedDocument]:
        ...


class FakeRetriever:
    """Honestly-named fixture retriever. Token-overlap over a committed corpus.

    Not an embedding model. Not a reranker. Not a live search API.
    """

    def __init__(self, documents: list[CapturedDocument] | None = None) -> None:
        if documents is None:
            documents = load_fixture_documents()
        self.documents = list(documents)

    def retrieve(self, query: str, k: int = 5) -> list[CapturedDocument]:
        qt = set(query.lower().split())
        scored: list[tuple[float, CapturedDocument]] = []
        for doc in self.documents:
            tokens = set((doc.title + " " + doc.body).lower().split())
            score = len(qt & tokens) / max(1, len(qt))
            scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [d for s, d in scored[:k] if s > 0]


def load_fixture_corpus() -> dict:
    return json.loads(FIXTURE_CORPUS_PATH.read_text())


def load_fixture_documents() -> list[CapturedDocument]:
    raw = load_fixture_corpus()
    return [CapturedDocument.from_fixture(d) for d in raw["documents"]]


def load_seed_claims() -> list[dict]:
    return list(load_fixture_corpus()["seed_claims"])


# --------------------------------------------------------------------------
# Decomposition, capture, citation, quality, contradiction, freshness
# --------------------------------------------------------------------------


def decompose_query(query: str, mode: ResearchMode) -> list[str]:
    """Query decomposition. Deterministic, fixture-honest (no LLM call)."""
    q = query.strip()
    if not q:
        return []
    base = [q]
    # Split on "vs" / "versus" / "compare" for comparison mode
    if mode is ResearchMode.COMPARISON or re.search(r"\bvs\.?\b|versus|compare", q, re.I):
        parts = re.split(r"\bvs\.?\b|versus|compare\s+", q, flags=re.I)
        parts = [p.strip(" :?") for p in parts if p and p.strip(" :?")]
        if len(parts) >= 2:
            return parts
    if mode is ResearchMode.DEEP_RESEARCH or mode is ResearchMode.LITERATURE_REVIEW:
        # Expand into aspect sub-queries without inventing external knowledge.
        aspects = [
            f"{q} measurements",
            f"{q} methods",
            f"{q} contradictions",
        ]
        return base + aspects
    if mode is ResearchMode.FACT_AUDIT:
        return [f"audit: {q}", f"counter-evidence: {q}"]
    return base


def grade_source(doc: CapturedDocument) -> dict:
    return {
        "source_id": doc.source_id,
        "doc_id": doc.id,
        "grade": doc.quality.grade(),
        "band": doc.quality.band(),
        "components": {
            "authority": doc.quality.authority,
            "recency": doc.quality.recency,
            "independence": doc.quality.independence,
            "reproducibility": doc.quality.reproducibility,
        },
    }


def extract_citations_for_seed(
    doc: CapturedDocument,
    seed: dict,
) -> Citation | None:
    """Extract a citation span from a fixture seed claim against its document body."""
    text = seed["text"]
    body_l = doc.body.lower()
    # Find a distinctive substring from the seed text in the body
    needle = None
    for token in sorted(text.split(), key=len, reverse=True):
        if len(token) >= 4 and token.lower() in body_l:
            needle = token
            break
    if needle is None:
        # fall back to value if present
        if seed.get("value") and str(seed["value"]).lower() in body_l:
            needle = str(seed["value"])
        else:
            return None
    start = body_l.find(needle.lower())
    if start < 0:
        return None
    end = start + len(needle)
    # widen excerpt a little for human readability
    lo = max(0, start - 40)
    hi = min(len(doc.body), end + 40)
    excerpt = doc.body[lo:hi]
    cid = hashlib.sha256(f"{doc.id}|{seed['claim_key']}|{start}".encode()).hexdigest()[:12]
    return Citation(
        id=cid,
        doc_id=doc.id,
        source_id=doc.source_id,
        excerpt=excerpt,
        char_range=(start, end),
        content_hash=doc.content_hash,
    )


def detect_contradictions(claims: Iterable[Claim]) -> list[Contradiction]:
    """Same subject, different values → recorded contradiction. No auto-pick."""
    by_subject: dict[str, list[Claim]] = {}
    for c in claims:
        if c.subject is None or c.value is None:
            continue
        if c.category is not ClaimCategory.RETRIEVED_FACT:
            continue
        by_subject.setdefault(c.subject, []).append(c)

    out: list[Contradiction] = []
    for subject, group in sorted(by_subject.items()):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.value == b.value:
                    continue
                if a.origin_source_id == b.origin_source_id:
                    continue  # same source restating itself is not a cross-source contradiction
                cid = hashlib.sha256(
                    f"{subject}|{a.id}|{b.id}".encode()
                ).hexdigest()[:12]
                out.append(
                    Contradiction(
                        id=cid,
                        subject=subject,
                        claim_a_id=a.id,
                        claim_b_id=b.id,
                        source_a_id=a.origin_source_id or "unknown",
                        source_b_id=b.origin_source_id or "unknown",
                        value_a=a.value,
                        value_b=b.value,
                        resolution="unresolved_both_surfaced",
                        preferred_claim_id=None,
                    )
                )
    return out


def freshness_for_claim(
    claim: Claim,
    answer_at: str,
    *,
    threshold_seconds: float = 90 * 24 * 3600,
) -> FreshnessReport:
    """Every source carries captured-at; claim carries staleness at answer time."""
    answer_dt = _parse_iso(answer_at)
    if claim.source_captured_at is None:
        return FreshnessReport(
            claim_id=claim.id,
            source_captured_at=None,
            answer_at=answer_at,
            age_seconds=None,
            stale=False,
            threshold_seconds=threshold_seconds,
        )
    cap = _parse_iso(claim.source_captured_at)
    age = (answer_dt - cap).total_seconds()
    return FreshnessReport(
        claim_id=claim.id,
        source_captured_at=claim.source_captured_at,
        answer_at=answer_at,
        age_seconds=age,
        stale=age > threshold_seconds,
        threshold_seconds=threshold_seconds,
    )


# --------------------------------------------------------------------------
# Research run + controller
# --------------------------------------------------------------------------


@dataclass
class ResearchRunResult:
    run_id: str
    mode: ResearchMode
    query: str
    sub_queries: list[str]
    documents: list[CapturedDocument]
    source_grades: list[dict]
    claims: list[Claim]
    citations: list[Citation]
    contradictions: list[Contradiction]
    freshness: list[FreshnessReport]
    checkpoints: list[ResearchCheckpoint]
    graph_standing: dict[str, list[str]]
    answer_at: str
    durable_memory_fingerprint_before: str
    durable_memory_fingerprint_after: str
    durable_memory_untouched: bool
    promotions: list[MemoryPromotion]
    authority: str = AUTHORITY

    def cited_answer(self) -> dict:
        """Answer surface: claims + citations + unresolved contradictions + freshness."""
        return {
            "query": self.query,
            "mode": self.mode.value,
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "category": c.category.value,
                    "evidence": c.evidence.value,
                    "citation_ids": list(c.citation_ids),
                    "subject": c.subject,
                    "value": c.value,
                    "confidence": c.confidence,
                    "source_captured_at": c.source_captured_at,
                }
                for c in self.claims
            ],
            "citations": [
                {
                    "id": ct.id,
                    "doc_id": ct.doc_id,
                    "source_id": ct.source_id,
                    "excerpt": ct.excerpt,
                }
                for ct in self.citations
            ],
            "contradictions": [
                {
                    "id": x.id,
                    "subject": x.subject,
                    "claim_a_id": x.claim_a_id,
                    "claim_b_id": x.claim_b_id,
                    "source_a_id": x.source_a_id,
                    "source_b_id": x.source_b_id,
                    "value_a": x.value_a,
                    "value_b": x.value_b,
                    "resolution": x.resolution,
                    "preferred_claim_id": x.preferred_claim_id,
                }
                for x in self.contradictions
            ],
            "freshness": [
                {
                    "claim_id": f.claim_id,
                    "source_captured_at": f.source_captured_at,
                    "answer_at": f.answer_at,
                    "age_seconds": f.age_seconds,
                    "stale": f.stale,
                }
                for f in self.freshness
            ],
            "durable_memory_untouched": self.durable_memory_untouched,
        }


class ResearchController:
    """Cited-web / deep-research controller for HIDE YOU (Y5).

    Pipeline per run:
      decompose → multi-source retrieve → capture → cite → grade →
      extract claims → detect contradictions → freshness → checkpoint

    Durable memory is injected and never written unless promote_to_memory is called.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        durable_memory: DurableMemory | None = None,
        *,
        clock: Callable[[], str] | None = None,
        seed_claims: list[dict] | None = None,
    ) -> None:
        self.retriever = retriever or FakeRetriever()
        self.memory = durable_memory if durable_memory is not None else DurableMemory()
        self.clock = clock or _iso
        self.seed_claims = seed_claims if seed_claims is not None else load_seed_claims()
        self.graph = ClaimEvidenceGraph()
        self._run_seq = 0
        self.checkpoints: list[ResearchCheckpoint] = []
        self.last_result: ResearchRunResult | None = None

    def _checkpoint(self, run_id: str, kind: str, detail: dict | None = None) -> ResearchCheckpoint:
        self._run_seq += 1
        cp = ResearchCheckpoint(
            run_id=run_id,
            seq=self._run_seq,
            at=self.clock(),
            kind=kind,
            detail=detail or {},
        )
        self.checkpoints.append(cp)
        return cp

    def _make_claim_from_seed(
        self,
        doc: CapturedDocument,
        seed: dict,
        citation: Citation,
        run_id: str,
    ) -> Claim:
        claim_id = hashlib.sha256(
            f"{run_id}|{doc.id}|{seed['claim_key']}|{seed['text']}".encode()
        ).hexdigest()[:12]
        return Claim(
            id=claim_id,
            text=seed["text"],
            category=ClaimCategory.RETRIEVED_FACT,
            evidence=EvidenceBinding.LINKED,
            citation_ids=(citation.id,),
            subject=seed.get("subject"),
            value=str(seed["value"]) if seed.get("value") is not None else None,
            confidence=min(0.95, 0.5 + doc.quality.grade() / 2.0),
            created_at=self.clock(),
            origin_doc_id=doc.id,
            origin_source_id=doc.source_id,
            source_captured_at=doc.captured_at,
        )

    def run(
        self,
        query: str,
        mode: ResearchMode = ResearchMode.CITED_ANSWER,
        *,
        k: int = 5,
        answer_at: str | None = None,
        freshness_threshold_seconds: float = 90 * 24 * 3600,
        add_unsupported_inference: bool = True,
    ) -> ResearchRunResult:
        """Execute a research run. Does not touch durable memory."""
        answer_at = answer_at or self.clock()
        run_id = hashlib.sha256(f"{query}|{mode.value}|{answer_at}".encode()).hexdigest()[:16]
        mem_before = self.memory.fingerprint()
        write_count_before = self.memory.write_count

        self._checkpoint(run_id, "opened", {"query": query, "mode": mode.value})

        sub_queries = decompose_query(query, mode)
        self._checkpoint(run_id, "decomposed", {"sub_queries": sub_queries})

        # Multi-source retrieval (fixture; no network)
        seen: dict[str, CapturedDocument] = {}
        for sq in sub_queries:
            for doc in self.retriever.retrieve(sq, k=k):
                seen[doc.id] = doc
        documents = list(seen.values())
        self._checkpoint(run_id, "retrieved", {"doc_ids": [d.id for d in documents]})

        # Capture into graph
        for doc in documents:
            self.graph.add_document_node(doc)
        self._checkpoint(run_id, "captured", {"n": len(documents)})

        source_grades = [grade_source(d) for d in documents]
        self._checkpoint(run_id, "graded", {"grades": source_grades})

        # Citation extraction + claim minting from fixture seeds matched to retrieved docs
        doc_ids = {d.id for d in documents}
        citations: list[Citation] = []
        claims: list[Claim] = []
        for seed in self.seed_claims:
            if seed["doc_id"] not in doc_ids:
                continue
            doc = seen[seed["doc_id"]]
            cit = extract_citations_for_seed(doc, seed)
            if cit is None:
                continue
            self.graph.add_citation(cit)
            citations.append(cit)
            claim = self._make_claim_from_seed(doc, seed, cit, run_id)
            self.graph.add_claim(claim)
            claims.append(claim)
        self._checkpoint(
            run_id,
            "claims_extracted",
            {"n_claims": len(claims), "n_citations": len(citations)},
        )

        # Optional model inference / unsupported hypothesis for modes that synthesize
        if add_unsupported_inference and mode in (
            ResearchMode.CITED_ANSWER,
            ResearchMode.DEEP_RESEARCH,
            ResearchMode.COMPARISON,
            ResearchMode.FACT_AUDIT,
        ):
            inf_id = hashlib.sha256(f"{run_id}|inference".encode()).hexdigest()[:12]
            inference = Claim(
                id=inf_id,
                text=f"model synthesis regarding: {query}",
                category=ClaimCategory.MODEL_INFERENCE,
                evidence=EvidenceBinding.UNSUPPORTED,
                citation_ids=(),
                subject=None,
                value=None,
                confidence=0.35,
                created_at=self.clock(),
            )
            # ModelInference is not a factual claim; evidence law allows unsupported.
            inference.assert_evidence_law()
            self.graph.add_claim(inference)
            claims.append(inference)

            hyp_id = hashlib.sha256(f"{run_id}|hypothesis".encode()).hexdigest()[:12]
            hypothesis = Claim(
                id=hyp_id,
                text=f"uncertain hypothesis about: {query}",
                category=ClaimCategory.UNCERTAIN_HYPOTHESIS,
                evidence=EvidenceBinding.UNSUPPORTED,
                citation_ids=(),
                subject=None,
                value=None,
                confidence=0.15,
                created_at=self.clock(),
            )
            hypothesis.assert_evidence_law()
            self.graph.add_claim(hypothesis)
            claims.append(hypothesis)

        # Contradiction detection — surface both, never auto-resolve
        contradictions = detect_contradictions(claims)
        for c in contradictions:
            self.graph.record_contradiction(c)
        self._checkpoint(run_id, "contradictions", {"n": len(contradictions)})

        # Freshness at answer time
        freshness = [
            freshness_for_claim(
                c, answer_at, threshold_seconds=freshness_threshold_seconds
            )
            for c in claims
            if c.category is ClaimCategory.RETRIEVED_FACT
        ]
        self._checkpoint(run_id, "freshness", {"n": len(freshness)})

        # Run complete — durable memory must be untouched
        mem_after = self.memory.fingerprint()
        untouched = (
            mem_before == mem_after and self.memory.write_count == write_count_before
        )
        self._checkpoint(
            run_id,
            "complete",
            {
                "durable_memory_untouched": untouched,
                "mem_fp_before": mem_before,
                "mem_fp_after": mem_after,
            },
        )

        result = ResearchRunResult(
            run_id=run_id,
            mode=mode,
            query=query,
            sub_queries=sub_queries,
            documents=documents,
            source_grades=source_grades,
            claims=claims,
            citations=citations,
            contradictions=contradictions,
            freshness=freshness,
            checkpoints=[c for c in self.checkpoints if c.run_id == run_id],
            graph_standing=self.graph.standing(),
            answer_at=answer_at,
            durable_memory_fingerprint_before=mem_before,
            durable_memory_fingerprint_after=mem_after,
            durable_memory_untouched=untouched,
            promotions=[],
        )
        self.last_result = result
        return result

    def promote_to_memory(
        self,
        claim_id: str,
        actor: str,
        reason: str,
        *,
        run_id: str | None = None,
    ) -> MemoryPromotion:
        """Explicit promotion path. The only way research material enters durable memory."""
        claim = self.graph.claims[claim_id]
        rid = run_id or (self.last_result.run_id if self.last_result else "manual")
        promo = self.memory.promote(claim, rid, actor, reason, at=self.clock())
        self._checkpoint(
            rid,
            "memory_promotion",
            {"claim_id": claim_id, "promotion_id": promo.id, "actor": actor},
        )
        if self.last_result and self.last_result.run_id == rid:
            self.last_result.promotions.append(promo)
            self.last_result.durable_memory_fingerprint_after = self.memory.fingerprint()
            self.last_result.durable_memory_untouched = False
        return promo

    def add_user_claim(self, text: str, *, subject: str | None = None, value: str | None = None) -> Claim:
        """Record a UserProvided claim. Factual ⇒ must be LINKED or UNSUPPORTED."""
        cid = hashlib.sha256(f"user|{text}|{self.clock()}".encode()).hexdigest()[:12]
        claim = Claim(
            id=cid,
            text=text,
            category=ClaimCategory.USER_PROVIDED,
            evidence=EvidenceBinding.UNSUPPORTED,  # user said it; no external evidence yet
            citation_ids=(),
            subject=subject,
            value=value,
            confidence=1.0,
            created_at=self.clock(),
        )
        claim.assert_evidence_law()
        self.graph.add_claim(claim)
        return claim


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def offline_controller(memory: DurableMemory | None = None) -> ResearchController:
    """Construct a fully offline controller over the committed fixture corpus."""
    return ResearchController(
        retriever=FakeRetriever(),
        durable_memory=memory if memory is not None else DurableMemory(),
        seed_claims=load_seed_claims(),
    )
