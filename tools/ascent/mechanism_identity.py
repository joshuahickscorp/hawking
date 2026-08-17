"""Semantic identity for mechanisms — not bottlenecks.

A bottleneck (weight_addressing) may be attacked again. A mechanism
(fuse tiny kernels into the following GEMV) may not, unless a listed
delta applies. Identity is semantic: paraphrases of a settled attempt
are the same mechanism.

No network. Matching is deterministic: canonical aliases, then token /
bigram / shingle overlap, then verb+axis agreement.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

SCHEMA = "hawking.ascent.mechanism_identity.v1"

# Combined score at or above HIGH is a duplicate on wording alone.
# MID needs a shared roof-axis and a shared verb family as well.
HIGH_SIMILARITY = 0.62
MID_SIMILARITY = 0.45
SHINGLE_SIZE = 5

STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "at",
        "of",
        "on",
        "in",
        "to",
        "ns",
        "ms",
        "us",
        "token",
        "per",
        "dirty",
        "engineering",
        "median",
        "class",
        "vs",
        "and",
        "or",
        "for",
        "with",
        "from",
        "by",
        "as",
        "into",
        "that",
        "this",
        "these",
        "those",
        "be",
        "been",
        "being",
        "was",
        "were",
        "are",
        "it",
        "its",
        "not",
        "no",
        "do",
        "does",
        "did",
        "than",
        "then",
        "via",
        "using",
        "use",
        "used",
        "one",
        "two",
        "only",
        "still",
        "also",
        "same",
        "every",
        "any",
        "all",
        "next",
        "following",
        "after",
        "before",
        "between",
        "across",
        "over",
        "under",
        "again",
        "already",
        "never",
        "must",
        "may",
        "can",
        "cannot",
        "will",
        "would",
        "should",
        "their",
        "them",
        "they",
        "we",
        "our",
        "my",
        "your",
        "so",
        "if",
        "but",
        "because",
        "when",
        "where",
        "which",
        "who",
        "how",
        "what",
        "why",
        "about",
        "against",
        "without",
        "within",
        "each",
        "both",
        "more",
        "less",
        "very",
        "just",
        "such",
        "other",
        "own",
        "new",
        "old",
        "true",
        "false",
        "yes",
        "measured",
        "complete",
        "wall",
        "cost",
        "time",
        "lane",
        "try",
        "retry",
        "attempt",
        "model",
        "qwen",
        "qwen38",
        "q80",
        "dsv4f",
    }
)

# Irregular stems so "fusing" and "fuse" collide.
IRREGULAR = {
    "fusing": "fuse",
    "fused": "fuse",
    "sharing": "share",
    "shared": "share",
    "caching": "cache",
    "cached": "cache",
    "amortizing": "amortize",
    "amortized": "amortize",
    "amortisation": "amortize",
    "amortization": "amortize",
    "reconstructing": "reconstruct",
    "reconstruction": "reconstruct",
    "assigning": "assign",
    "assignment": "assign",
    "collapsing": "collapse",
    "collapsed": "collapse",
    "dropping": "drop",
    "addressing": "address",
    "sessions": "session",
    "kernels": "kernel",
    "weights": "weight",
    "activations": "activation",
    "processes": "process",
    "pages": "page",
    "codecs": "codec",
    "layers": "layer",
    "heads": "head",
    "dispatches": "dispatch",
    "encoders": "encoder",
    "variants": "variant",
    "islands": "island",
}

VERB_FAMILIES = {
    "fuse": frozenset({"fuse", "merge", "fold", "collapse", "combine"}),
    "cache": frozenset({"cache", "reuse", "retain", "hot", "resident"}),
    "share": frozenset({"share", "amortize", "common"}),
    "drop": frozenset({"drop", "lsb", "trim", "truncate"}),
    "reconstruct": frozenset({"reconstruct", "decode", "unpack"}),
    "assign": frozenset({"assign", "mixed", "perlayer", "perhead"}),
    "layout": frozenset({"layout", "permute", "interleave", "reorder", "blocked"}),
    "skip": frozenset({"skip", "gate", "sparse", "omit"}),
    "prefetch": frozenset({"prefetch", "preload", "prewarm"}),
    "fit": frozenset({"fit", "calibrate", "capture"}),
}

AXIS_CUES = {
    "representation": frozenset(
        {
            "codec",
            "bpw",
            "q3",
            "q4",
            "q8",
            "lsb",
            "absmax",
            "outlier",
            "sparsity",
            "pack",
            "quant",
            "assignment",
            "perlayer",
            "perhead",
        }
    ),
    "bytes": frozenset({"byte", "bytes", "traffic", "dram", "bandwidth", "unique"}),
    "kernel": frozenset({"kernel", "gemv", "matvec", "shader", "metal", "simd"}),
    "launch_geometry": frozenset({"tpr", "tg", "threadgroup", "occupancy", "geometry"}),
    "fusion": frozenset({"fuse", "fusion", "fold", "merge", "collapse"}),
    "command_topology": frozenset(
        {"dispatch", "encoder", "command", "cb", "topology", "encoder"}
    ),
    "synchronization": frozenset({"sync", "synchronization", "wait", "barrier"}),
    "residency": frozenset(
        {"resident", "residency", "cache", "arena", "session", "process", "page"}
    ),
    "addressing": frozenset({"address", "addressing", "gather", "layout", "stride"}),
    "host_gpu_partition": frozenset({"host", "gpu", "partition", "bind", "encode"}),
    "kv_strategy": frozenset({"kv", "cachekv", "state"}),
    "genome": frozenset({"genome", "runtime", "family"}),
    "evidence": frozenset(
        {
            "gaussian",
            "synthetic",
            "degraded",
            "undersampled",
            "underdetermined",
            "cosine",
            "proxy",
        }
    ),
}

# (canonical_id, axis, phrases). Phrases are matched after normalize().
# These are the campaign-settled mechanisms plus the open next ones.
ALIASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "fuse_tiny_kernels_into_gemv",
        "fusion",
        (
            "fuse tiny kernels into the following gemv",
            "fusing tiny kernels into the following gemv",
            "fuse small kernels into next gemv",
            "fuse small metal kernels into the next gemv",
            "merge microkernels into the trailing gemv",
            "fold tiny kernels into the gemv",
            "fuse micro dispatches into gemv",
            "folding tiny kernels into the following gemv",
        ),
    ),
    (
        "cross_token_weight_cache",
        "residency",
        (
            "cross-token cache reuse",
            "cross token cache reuse",
            "retain weights between tokens",
            "hot cache weights across tokens",
            "weight cache between decode steps",
            "keep a tier that retains weights between tokens",
        ),
    ),
    (
        "shared_resident_weight_amortize",
        "residency",
        (
            "n sessions sharing one resident weight body",
            "sessions sharing one resident weight body",
            "multi session dram amortization",
            "concurrent sessions share weights to cut dram",
            "share one weight body across sessions",
            "n sessions sharing one resident weight body to amortize dram",
        ),
    ),
    (
        "process_page_share",
        "residency",
        (
            "separate processes share artifact pages",
            "separate processes sharing artifact pages",
            "multi process page cache sharing",
            "two processes map the same artifact",
            "processes share artifact pages",
        ),
    ),
    (
        "out_proj_denser_codec",
        "representation",
        (
            "out_proj denser codec",
            "o_proj denser codec",
            "absmax-per-64 on out_proj",
            "absmax per 64 on out_proj",
            "median scale on out_proj",
            "p90 scale on out_proj",
            "outlier islands on out_proj",
            "structured sparsity on out_proj",
            "per-head assignment on out_proj",
            "out_proj q3",
            "o_proj q3",
            "every codec tried on out_proj",
        ),
    ),
    (
        "lm_head_drop_lsb",
        "representation",
        (
            "lm_head drop-lsb",
            "lm_head drop lsb",
            "drop lsb on lm_head",
            "drop-lsb lm_head",
            "lm head drop least significant bits",
        ),
    ),
    (
        "reconstruction_codec_cost",
        "kernel",
        (
            "reconstruction cost of the codec",
            "codec reconstruction is the wall",
            "decode reconstruction penalty",
            "reconstruction is free at tpr64",
            "codec variants on real activations",
        ),
    ),
    (
        "gaussian_synthetic_activations",
        "evidence",
        (
            "gaussian activations",
            "synthetic activations",
            "gaussian synthetic proxy activations",
            "evaluate or fit on gaussian activations",
            "fit compression on synthetic activations",
            "evaluate or fit compression on gaussian synthetic proxy activations",
            "evaluate or fit compression on gaussian / synthetic proxy activations",
        ),
    ),
    (
        "degraded_model_fits",
        "evidence",
        (
            "fits from a degraded model",
            "capture x from a candidate under test",
            "calibrate from a quantized gibberish baseline",
            "fits taken from a degraded model",
        ),
    ),
    (
        "undersampled_fits",
        "evidence",
        (
            "undersampled fits",
            "underdetermined unit",
            "fit on fewer rows than the fitted dimension",
            "median 92 rows against 2048 dims",
        ),
    ),
    (
        "per_layer_per_head_assignment",
        "representation",
        (
            "per-layer and per-head assignment",
            "per layer per head assignment",
            "assign codecs per layer and per head",
            "per-layer codec assignment",
            "per-head codec assignment",
            "mixed per-layer representation on the coherent vehicle",
        ),
    ),
    (
        "activation_gated_weight_skip",
        "bytes",
        (
            "activation gated weight skip",
            "skip unread weight rows this token",
            "do not read every weight every token",
            "activation-sparse weight fetch",
            "gate weight traffic on this token's activations",
        ),
    ),
    (
        "sub1_bpw_attention_embed_norms",
        "representation",
        (
            "sub-1 bpw on attention embed norms",
            "sub 1 bpw attention embed norms",
            "cut attention embed norms below one bpw",
            "sub-bit attention and embeddings",
        ),
    ),
    (
        "addressing_layout_not_codec",
        "addressing",
        (
            "change addressing layout without changing the codec",
            "blocked or morton weight layout",
            "organ-contiguous addressing",
            "address generation layout independent of representation",
        ),
    ),
    (
        "host_gpu_partition_of_addressing",
        "host_gpu_partition",
        (
            "move address generation between host and gpu",
            "host gpu partition of addressing",
            "bind and address on the other side of the host gpu cut",
        ),
    ),
    (
        "fewer_weight_bytes_bpw",
        "bytes",
        (
            "fewer weight bytes",
            "cut unique-once bytes",
            "lower active bpw of the bytes actually moved",
            "attack the 13.618 gb unique-once quantity",
        ),
    ),
    (
        "storage_bpw_as_active",
        "evidence",
        (
            "treat storage bpw as active bpw",
            "complete_physical bpw as the bpw decode moves",
        ),
    ),
    (
        "reuse_band_as_decode_ceiling",
        "evidence",
        (
            "use the reuse band as the decode ceiling",
            "560-647 as the decode ceiling",
            "535-637 as the decode ceiling",
        ),
    ),
)


_WORD = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")
_HYPHEN_KEEP = re.compile(r"[^a-z0-9._]+")


def normalize(text: str) -> str:
    """Lowercase, keep letters/digits, collapse punctuation to space."""
    if text is None:
        return ""
    s = str(text).lower().replace("per-layer", "perlayer").replace("per-head", "perhead")
    s = s.replace("per layer", "perlayer").replace("per head", "perhead")
    s = s.replace("out_proj", "outproj").replace("o_proj", "outproj")
    s = s.replace("lm_head", "lmhead").replace("drop-lsb", "droplsb")
    s = s.replace("_", " ")
    s = _HYPHEN_KEEP.sub(" ", s)
    return " ".join(s.split())


def _stem(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "ies", "ied", "ed", "es"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            if suf == "ies":
                return word[:-3] + "y"
            return word[: -len(suf)]
    if len(word) > 5 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Content tokens, stemmed, stopwords and large pure integers dropped."""
    out: list[str] = []
    for raw in _WORD.findall(normalize(text)):
        if raw in STOP:
            continue
        if raw.isdigit() and len(raw) > 4:
            continue
        stem = _stem(raw)
        if stem in STOP or len(stem) < 2:
            continue
        out.append(stem)
    return out


def _verbs(tokens: Iterable[str]) -> frozenset[str]:
    found: set[str] = set()
    bag = set(tokens)
    for family, members in VERB_FAMILIES.items():
        if bag & members or family in bag:
            found.add(family)
    return frozenset(found)


def _axes(tokens: Iterable[str]) -> frozenset[str]:
    found: set[str] = set()
    bag = set(tokens)
    for axis, cues in AXIS_CUES.items():
        if bag & cues:
            found.add(axis)
    return frozenset(found)


@dataclass(frozen=True)
class Fingerprint:
    raw: str
    canonical_id: str | None
    axis: str | None
    tokens: frozenset[str]
    bigrams: frozenset[str]
    shingles: frozenset[str]
    verbs: frozenset[str]
    axes: frozenset[str]
    digest: str

    def empty(self) -> bool:
        return not self.tokens and not self.canonical_id


@dataclass(frozen=True)
class Match:
    same: bool
    score: float
    reason: str
    canonical_id: str | None = None
    left: str = ""
    right: str = ""
    extras: dict = field(default_factory=dict)


def _alias_table() -> list[tuple[str, str, frozenset[str], str]]:
    """canonical_id, axis, token-set, phrase."""
    rows = []
    for cid, axis, phrases in ALIASES:
        for phrase in phrases:
            rows.append((cid, axis, frozenset(tokenize(phrase)), phrase))
    return rows


_ALIAS_ROWS = _alias_table()


def _alias_hit_score(candidate: frozenset[str], phrase: frozenset[str]) -> float:
    """Jaccard, plus a subset bonus so a short alias names a longer wording."""
    score = jaccard(candidate, phrase)
    if score >= 0.72:
        return score
    # Listed alias sits inside a longer wording. The other direction
    # (a short fragment of a long alias) is how "per-head assignment"
    # would collapse onto "per-head assignment on out_proj".
    if len(phrase) >= 2 and phrase <= candidate:
        return max(score, 0.80)
    return score


def _best_alias(tokens: frozenset[str]) -> tuple[str | None, str | None, float]:
    if not tokens:
        return None, None, 0.0
    best_id, best_axis, best = None, None, 0.0
    for cid, axis, phrase_tokens, _phrase in _ALIAS_ROWS:
        score = _alias_hit_score(tokens, phrase_tokens)
        # Prefer the tighter (higher) score; ties keep the first, which is
        # the more specific phrase listed earlier in that id's group.
        if score > best:
            best_id, best_axis, best = cid, axis, score
    if best >= 0.72:
        return best_id, best_axis, best
    return None, None, best


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    s = normalize(text).replace(" ", "")
    if not s:
        return frozenset()
    if len(s) < size:
        return frozenset([s])
    return frozenset(s[i : i + size] for i in range(len(s) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0  # empty-empty is not identity; the gate must see a hollow input
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _digest(canonical_id: str | None, tokens: frozenset[str]) -> str:
    if canonical_id:
        body = f"id:{canonical_id}"
    else:
        body = "tok:" + ",".join(sorted(tokens))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def fingerprint(text: str) -> Fingerprint:
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    toks = frozenset(tokenize(raw))
    cid, axis, _ = _best_alias(toks)
    grams = frozenset()
    seq = tokenize(raw)
    if len(seq) >= 2:
        grams = frozenset(f"{seq[i]}|{seq[i+1]}" for i in range(len(seq) - 1))
    axes = _axes(toks)
    if axis:
        axes = frozenset(set(axes) | {axis})
    return Fingerprint(
        raw=raw,
        canonical_id=cid,
        axis=axis,
        tokens=toks,
        bigrams=grams,
        shingles=shingles(raw),
        verbs=_verbs(toks),
        axes=axes,
        digest=_digest(cid, toks),
    )


def combined_score(a: Fingerprint, b: Fingerprint) -> float:
    token_j = jaccard(a.tokens, b.tokens)
    gram_j = jaccard(a.bigrams, b.bigrams)
    shin_j = jaccard(a.shingles, b.shingles)
    return 0.50 * token_j + 0.30 * gram_j + 0.20 * shin_j


def same_mechanism(left: str, right: str) -> Match:
    """True when *left* and *right* name the same mechanism."""
    fa, fb = fingerprint(left), fingerprint(right)
    if fa.empty() or fb.empty():
        return Match(
            False,
            0.0,
            "empty_mechanism",
            left=fa.raw,
            right=fb.raw,
        )
    if fa.canonical_id and fa.canonical_id == fb.canonical_id:
        return Match(
            True,
            1.0,
            "canonical_id",
            canonical_id=fa.canonical_id,
            left=fa.raw,
            right=fb.raw,
        )
    score = combined_score(fa, fb)
    if score >= HIGH_SIMILARITY:
        return Match(
            True,
            score,
            "high_similarity",
            canonical_id=fa.canonical_id or fb.canonical_id,
            left=fa.raw,
            right=fb.raw,
        )
    shared_axes = fa.axes & fb.axes
    shared_verbs = fa.verbs & fb.verbs
    if score >= MID_SIMILARITY and shared_axes and shared_verbs:
        return Match(
            True,
            score,
            "mid_similarity_same_axis_verb",
            canonical_id=fa.canonical_id or fb.canonical_id,
            left=fa.raw,
            right=fb.raw,
            extras={"axes": sorted(shared_axes), "verbs": sorted(shared_verbs)},
        )
    return Match(
        False,
        score,
        "distinct",
        canonical_id=None,
        left=fa.raw,
        right=fb.raw,
        extras={"axes": sorted(shared_axes), "verbs": sorted(shared_verbs)},
    )


def is_bottleneck_name(text: str) -> bool:
    """True when the string is only a component name (+ optional numbers)."""
    toks = list(tokenize(text))
    if not toks:
        return False
    bag = set(toks)
    if bag <= {"weight", "address"}:
        return True
    if bag <= {"weight", "decode", "reconstruct"}:
        return True
    if bag <= {"dense", "swiglu"}:
        return True
    if bag <= {"host", "preparation"}:
        return True
    if bag <= {"kv", "state"}:
        return True
    if bag <= {"terminal", "head"}:
        return True
    if bag <= {"command", "submission"}:
        return True
    if bag <= {"unattributed", "residual"}:
        return True
    if bag <= {"weight", "address", "complete", "wall"}:
        return True
    if len(bag) == 1 and next(iter(bag)) in {
        "deltanet",
        "gqa",
        "normalization",
        "residual",
        "synchronization",
    }:
        return True
    compact = {
        "weightaddress",
        "weightdecode",
        "weightdecodereconstruct",
        "denseswiglu",
        "hostpreparation",
        "kvstate",
        "terminalhead",
        "commandsubmission",
        "unattributedresidual",
        "weightaddresscompletewall",
    }
    return "".join(toks) in compact
