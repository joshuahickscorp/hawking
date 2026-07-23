#!/usr/bin/env python3.12
"""Build and verify the offline GLM-5.2 quality-corpus integrity contract.

This module deliberately contains *no* model-loading or network code.  It loads
the official tokenizer JSON from the immutable GLM-5.2 snapshot, constructs a
small deterministic prompt corpus plus real tokenized long-context windows, and
hard-fails every leakage/inflation condition in Part IX of the campaign plan.

The emitted JSON is a reproducibility manifest, not a bundle of prompt text.
Every record is reconstructable from this source file and is bound by hashes;
``verify`` reconstructs it and requires byte-for-byte canonical equality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from glm52_common import (
    Glm52Error,
    atomic_json,
    atomic_text,
    canonical,
    read_sealed_json,
    seal,
    sha256_file,
    verify_sealed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPOSITORY = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_BYTES = 20_217_442
TOKENIZER_VOCAB_SIZE = 154_856
DEFAULT_TOKENIZER_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--zai-org--GLM-5.2"
    / "snapshots"
    / REVISION
    / "tokenizer.json"
)
OUTPUT_JSON = REPO_ROOT / "GLM52_CORPUS_INTEGRITY.json"
OUTPUT_MARKDOWN = REPO_ROOT / "GLM52_CORPUS_INTEGRITY.md"
SCHEMA = "hawking.glm52.corpus_integrity.v2"
GENERATOR_ID = "hawking-glm52-part-ix-v2"
GENERATOR_SEED = "glm52-bf16-xet-gravity/corpus/v2/2026-07-21"

PARTITIONS = (
    "representation fit",
    "router/indexer calibration",
    "Doctor training",
    "cross-validation",
    "score",
    "held-out",
    "replication",
    "protected-domain holdout",
    "long-context holdout",
)
TRAIN_PARTITIONS = frozenset(
    {"representation fit", "router/indexer calibration", "Doctor training"}
)
EVALUATION_PARTITIONS = frozenset(
    {
        "score",
        "held-out",
        "replication",
        "protected-domain holdout",
        "long-context holdout",
    }
)
DOMAINS = (
    "general prose",
    "factual completion",
    "science",
    "mathematics",
    "reasoning",
    "coding",
    "instruction following",
    "tool formatting",
    "agentic coding",
    "rare tokens",
    "Chinese",
    "English",
    "mixed-language",
    "long-context retrieval",
    "long-context synthesis",
)
ADMITTED_CONTEXT_RUNGS = (2_048, 8_192, 32_768, 131_072)
CONTEXT_RUNG_LABELS = {
    2_048: "2K",
    8_192: "8K",
    32_768: "32K",
    131_072: "128K",
    262_144: "256K",
    1_048_576: "1M",
}
QUALITY_METRICS = (
    "NLL/perplexity",
    "symmetric and directional KL",
    "logit cosine",
    "top-1 agreement",
    "top-k overlap",
    "deterministic generation",
    "route/index agreement",
    "hidden-state cosine/relative error",
    "weighted MoE output",
    "attention output",
    "coding",
    "math",
    "reasoning",
    "instruction following",
    "tool formatting",
    "long-context retrieval",
    "long-context synthesis",
)
POSITION_FRACTIONS = (0.07, 0.17, 0.28, 0.39, 0.50, 0.61, 0.72, 0.83, 0.93)
POSITION_BUCKETS = ("opening", "early", "middle", "late", "closing")
NORMALIZE_RE = re.compile(r"\s+")
SEMANTIC_VARIABLE_RE = re.compile(
    r"\b(?:[a-z]+-)?[0-9a-f]{8,}\b|\b\d+(?:\.\d+)?\b", re.IGNORECASE
)
CHAR_SHINGLE_WIDTH = 9
TOKEN_SHINGLE_WIDTH = 3
CHAR_SHINGLE_MAX_ALL_CROSS_SPLIT = 0.35
CHAR_SHINGLE_MAX_TRAIN_VS_EVAL = 0.25
TOKEN_SHINGLE_MAX_ALL_CROSS_SPLIT = 0.35
TOKEN_SHINGLE_MAX_TRAIN_VS_EVAL = 0.28


class CorpusIntegrityError(Glm52Error):
    """Raised with a stable Part-IX gate code when corpus admission fails."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class TokenizerBundle:
    tokenizer: Any
    path: Path
    resolved_path: Path
    sha256: str
    byte_count: int
    vocab_size: int


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    partition: str
    domain: str
    kind: str
    document_family_id: str
    source_document_id: str
    source_document: str
    atomic_segments: tuple[str, ...]
    context_window: str
    prompt: str
    expected_answer: str
    provenance: dict[str, Any]
    token_count: int
    token_ids_sha256: str
    embedding_claim_token_ids: tuple[int, ...] = ()
    context_rung_tokens: int | None = None
    evidence_markers: tuple[str, ...] = ()
    evidence_segment_indices: tuple[int, ...] = ()
    position_fraction: float | None = None
    position_bucket: str | None = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _normalized(value: str) -> str:
    return NORMALIZE_RE.sub(" ", value.strip()).casefold()


def _normalized_hash(value: str) -> str:
    return _sha256_text(_normalized(value))


def _semantic_skeleton(value: str) -> str:
    """Redact superficial numeric/identifier variation before family comparison."""
    return _normalized(SEMANTIC_VARIABLE_RE.sub("<variable>", value))


def _shingles(values: Sequence[Any], width: int) -> frozenset[tuple[Any, ...]]:
    if not values:
        return frozenset()
    if len(values) < width:
        return frozenset({tuple(values)})
    return frozenset(
        tuple(values[index : index + width])
        for index in range(len(values) - width + 1)
    )


def _jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _token_id_hash(ids: Sequence[int]) -> str:
    # Decimal JSON is architecture-independent and unambiguous.
    return _sha256_bytes(canonical(list(ids)))


def _merkleish_hash(hashes: Sequence[str]) -> str:
    # The ordered digest is sufficient here; the name avoids claiming a tree proof.
    return _sha256_bytes(canonical(list(hashes)))


def _provenance(record_id: str, recipe: str) -> dict[str, Any]:
    return {
        "origin_type": "deterministically_generated",
        "source_locator": f"builtin://{GENERATOR_ID}/{record_id}",
        "generator_id": GENERATOR_ID,
        "generator_seed_sha256": _sha256_text(f"{GENERATOR_SEED}/{record_id}"),
        "recipe": recipe,
        "license": "CC0-1.0",
        "network_access": False,
    }


def load_pinned_tokenizer(path: Path = DEFAULT_TOKENIZER_PATH) -> TokenizerBundle:
    """Load only the official local tokenizer after checking immutable identity."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment failure is explicit
        raise CorpusIntegrityError(
            "TOKENIZER_RUNTIME_MISSING", "the pinned tokenizers runtime is unavailable"
        ) from exc

    path = path.expanduser()
    if not path.exists():
        raise CorpusIntegrityError(
            "PINNED_TOKENIZER_MISSING", f"offline tokenizer does not exist: {path}"
        )
    if path.name != "tokenizer.json" or REVISION not in str(path):
        raise CorpusIntegrityError(
            "TOKENIZER_REVISION_MISMATCH",
            f"tokenizer path is not under immutable revision {REVISION}: {path}",
        )
    resolved = path.resolve(strict=True)
    digest = sha256_file(path)
    byte_count = path.stat().st_size
    if digest != TOKENIZER_SHA256 or byte_count != TOKENIZER_BYTES:
        raise CorpusIntegrityError(
            "TOKENIZER_IDENTITY_MISMATCH",
            f"sha256/bytes={digest}/{byte_count}, expected "
            f"{TOKENIZER_SHA256}/{TOKENIZER_BYTES}",
        )
    tokenizer = Tokenizer.from_file(str(path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size != TOKENIZER_VOCAB_SIZE:
        raise CorpusIntegrityError(
            "TOKENIZER_VOCAB_MISMATCH",
            f"vocabulary={vocab_size}, expected {TOKENIZER_VOCAB_SIZE}",
        )
    return TokenizerBundle(
        tokenizer=tokenizer,
        path=path,
        resolved_path=resolved,
        sha256=digest,
        byte_count=byte_count,
        vocab_size=vocab_size,
    )


def _encode(bundle: TokenizerBundle, text: str) -> tuple[int, ...]:
    return tuple(bundle.tokenizer.encode(text, add_special_tokens=False).ids)


def _make_record(
    bundle: TokenizerBundle,
    *,
    record_id: str,
    partition: str,
    domain: str,
    kind: str,
    document_family_id: str,
    source_document_id: str,
    source_document: str,
    atomic_segments: Sequence[str],
    context_window: str,
    prompt: str,
    expected_answer: str,
    provenance: dict[str, Any],
    embedding_claim_token_ids: Sequence[int] = (),
    context_rung_tokens: int | None = None,
    evidence_markers: Sequence[str] = (),
    evidence_segment_indices: Sequence[int] = (),
    position_fraction: float | None = None,
    position_bucket: str | None = None,
) -> CorpusRecord:
    ids = _encode(bundle, context_window)
    return CorpusRecord(
        record_id=record_id,
        partition=partition,
        domain=domain,
        kind=kind,
        document_family_id=document_family_id,
        source_document_id=source_document_id,
        source_document=source_document,
        atomic_segments=tuple(atomic_segments),
        context_window=context_window,
        prompt=prompt,
        expected_answer=expected_answer,
        provenance=copy.deepcopy(provenance),
        token_count=len(ids),
        token_ids_sha256=_token_id_hash(ids),
        embedding_claim_token_ids=tuple(embedding_claim_token_ids),
        context_rung_tokens=context_rung_tokens,
        evidence_markers=tuple(evidence_markers),
        evidence_segment_indices=tuple(evidence_segment_indices),
        position_fraction=position_fraction,
        position_bucket=position_bucket,
    )


def _domain_task(domain: str, partition_index: int) -> tuple[str, str, str]:
    """Return a distinct semantic family, prompt, and answer for one split cell.

    The nine rows within a domain intentionally exercise different operations and
    discourse forms.  Partition identity never appears in the prompt and changing only
    a number cannot turn one family into another.
    """
    n = partition_index + 1
    a = 17 + 13 * n
    b = 29 + 11 * n
    c = 7 + 5 * n
    def compact_json(value: Any) -> str:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    tasks: dict[str, tuple[tuple[str, str, str], ...]] = {
        "general prose": (
            (
                "fact-preserving-compression",
                f"Fuse these notes into one smooth sentence: parcel Cedar {a} arrived; "
                f"its blue seal remained intact; shelf {b} was empty.",
                f"Parcel Cedar {a} arrived with its blue seal intact and shelf {b} empty.",
            ),
            (
                "active-voice-revision",
                f"Put this passive sentence in active voice: The copper sample was placed "
                f"in tray {b} by technician Iona.",
                f"Technician Iona placed the copper sample in tray {b}.",
            ),
            (
                "chronological-narration",
                f"Narrate these events in time order: at 09:40 the gate closed; at 09:{c:02d} "
                f"the cart entered; at 09:{n:02d} the guard logged it.",
                f"The guard logged the cart at 09:{n:02d}, it entered at 09:{c:02d}, and "
                "the gate closed at 09:40.",
            ),
            (
                "neutral-register",
                f"Rewrite without emotional language: It was outrageous that crate {a} "
                f"waited {c} minutes beside the lift.",
                f"Crate {a} waited {c} minutes beside the lift.",
            ),
            (
                "pronoun-resolution",
                f"Replace the ambiguous pronoun with 'Mara': Mara handed Lio folder {b} "
                "after she indexed the final page.",
                f"Mara handed Lio folder {b} after Mara indexed the final page.",
            ),
            (
                "plain-language-edit",
                f"Express this in plain language: Utilization of corridor {n} is prohibited "
                f"subsequent to hour {a}.",
                f"Do not use corridor {n} after hour {a}.",
            ),
            (
                "parallel-list-repair",
                f"Repair the list so its verbs are parallel: Team Birch will inspect rack "
                f"{a}, labeling bin {b}, and it will seal door {c}.",
                f"Team Birch will inspect rack {a}, label bin {b}, and seal door {c}.",
            ),
            (
                "formal-register",
                f"Make this suitable for a formal log: We kinda moved unit {b} 'cause bay "
                f"{n} got too warm.",
                f"Unit {b} was relocated because bay {n} became too warm.",
            ),
            (
                "headline-abstraction",
                f"Write a concise headline for this event: an overnight storm delayed ferry "
                f"{a}, but all {c} passengers reached port safely.",
                f"Storm Delays Ferry; Passengers Arrive Safely",
            ),
        ),
        "factual completion": (
            (
                "inventory-total",
                f"A generated ledger lists {a} basalt samples and {b} ice samples. Complete "
                "the total with one integer.",
                str(a + b),
            ),
            (
                "symbolic-map-lookup",
                f"In the synthetic atlas, glyph Lumen maps to sector Q-{b}. The entry for "
                "Lumen is sector",
                f"Q-{b}",
            ),
            (
                "arithmetic-sequence-completion",
                f"Continue the generated sequence by one term: {a}, {a + c}, {a + 2*c}, "
                f"{a + 3*c},",
                str(a + 4 * c),
            ),
            (
                "attribute-recall",
                f"Synthetic moon Orin-{n} is defined as having color violet-{c} and mass "
                f"class M-{a}. Its defined color is",
                f"violet-{c}",
            ),
            (
                "stock-remainder",
                f"Depot K began with {a + b} coils and issued {c}. The recorded remainder is",
                str(a + b - c),
            ),
            (
                "taxonomy-membership",
                f"The invented taxonomy declares zef-{a} a glider, mip-{b} a burrower, and "
                f"tor-{c} a swimmer. The class of mip-{b} is",
                "burrower",
            ),
            (
                "coordinate-recall",
                f"A local map assigns beacon Rowan coordinates ({a}, {b}). Complete: Rowan's "
                "vertical coordinate equals",
                str(b),
            ),
            (
                "ordered-event-recall",
                f"Generated chronology: Vale rang first, Nera rang second, Sol rang third. "
                "The bell immediately before Sol was",
                "Nera",
            ),
            (
                "boolean-fact-completion",
                f"Register rule: flag amber-{a} is active and flag teal-{b} is inactive. "
                f"Complete the truth value of 'amber-{a} is active':",
                "true",
            ),
        ),
        "science": (
            (
                "kinematic-speed",
                f"A test cart covers {a} metres in {c} seconds. Report mean speed in metres "
                "per second to six decimal places.",
                f"{a / c:.6f}",
            ),
            (
                "density-ratio",
                f"A manufactured sample has mass {a * c} grams and volume {c} cubic "
                "centimetres. What is its density in g/cm³?",
                str(a),
            ),
            (
                "newtonian-force",
                f"An idealized body of mass {c} kg accelerates at {n + 2} m/s². Give the net "
                "force in newtons.",
                str(c * (n + 2)),
            ),
            (
                "electrical-energy",
                f"A laboratory lamp draws {a} watts for {c} seconds. Compute energy used in "
                "joules.",
                str(a * c),
            ),
            (
                "solution-concentration",
                f"A solution contains {c} grams of solute in {a} litres. State grams per "
                "litre to six decimals.",
                f"{c / a:.6f}",
            ),
            (
                "mechanical-pressure",
                f"A piston applies {a * c} newtons uniformly over {c} square metres. Find "
                "pressure in pascals.",
                str(a),
            ),
            (
                "temperature-change",
                f"A chamber warms from {c} °C to {a} °C. Give the signed temperature change "
                "in Celsius degrees.",
                str(a - c),
            ),
            (
                "series-resistance",
                f"Ideal resistors of {a}, {b}, and {c} ohms are connected in series. What is "
                "their equivalent resistance?",
                str(a + b + c),
            ),
            (
                "wave-frequency",
                f"A synthetic wave travels at {a * c} m/s with wavelength {c} m. Find its "
                "frequency in hertz.",
                str(a),
            ),
        ),
        "mathematics": (
            ("integer-expression", f"Evaluate ({a} × {b}) + {c}.", str(a * b + c)),
            (
                "greatest-common-divisor",
                f"Find gcd({a * c}, {b * c}).",
                str(math.gcd(a * c, b * c)),
            ),
            (
                "linear-equation",
                f"Solve for x: {c}x + {a} = {c * b + a}.",
                str(b),
            ),
            (
                "arithmetic-series",
                f"Sum the integers from {n} through {n + c}, inclusive.",
                str(sum(range(n, n + c + 1))),
            ),
            (
                "rectangle-perimeter",
                f"A rectangle has side lengths {a} and {c}. Give its perimeter.",
                str(2 * (a + c)),
            ),
            (
                "modular-remainder",
                f"What remainder results when {a * b + c} is divided by {b}?",
                str((a * b + c) % b),
            ),
            (
                "finite-mean",
                f"Compute the arithmetic mean of {a}, {b}, and {c} to six decimal places.",
                f"{(a + b + c) / 3:.6f}",
            ),
            (
                "binomial-choice",
                f"How many unordered pairs can be chosen from {n + 5} distinct objects?",
                str(math.comb(n + 5, 2)),
            ),
            (
                "difference-of-squares",
                f"Simplify numerically: {b}² − {a}².",
                str(b * b - a * a),
            ),
        ),
        "reasoning": (
            (
                "set-exclusion-syllogism",
                f"Every amber key opens chest {a}. Nothing that opens chest {a} opens chest "
                f"{b}. Can an amber key open chest {b}? Answer yes or no.",
                "no",
            ),
            (
                "transitive-order",
                f"Rin is taller than Sol, and Sol is taller than Tev. Who is shortest?",
                "Tev",
            ),
            (
                "contrapositive",
                f"Rule: if lamp K is on, sensor {a} is awake. Sensor {a} is not awake. Is "
                "lamp K on?",
                "no",
            ),
            (
                "category-inheritance",
                f"All nembles are quiet. Object P-{b} is a nemble. Must P-{b} be quiet?",
                "yes",
            ),
            (
                "exclusive-choice",
                "Exactly one door is unlocked. The west door is locked and the east door is "
                "locked. Which door is unlocked: north, east, or west?",
                "north",
            ),
            (
                "schedule-elimination",
                f"Ari cannot meet on Monday; Bo cannot meet on Tuesday; the only options are "
                "Monday and Tuesday. If Ari must attend, which day remains?",
                "Tuesday",
            ),
            (
                "spatial-composition",
                f"Marker B is east of A. Marker C is north of B. Relative to A, where is C?",
                "northeast",
            ),
            (
                "constraint-assignment",
                f"Tiles red, blue, and green occupy slots 1–3. Red is in slot 1; blue is not "
                "in slot 3. Which color occupies slot 2?",
                "blue",
            ),
            (
                "necessary-condition",
                f"A valid pass requires both seal X and seal Y. Pass {c} lacks seal Y. Can it "
                "be valid?",
                "no",
            ),
        ),
        "coding": (
            (
                "generator-aggregation",
                f"Give a Python expression that sums the squares from 1 through {c}, with no "
                "imports.",
                f"sum(i*i for i in range(1, {c + 1}))",
            ),
            (
                "stable-deduplication",
                "Write one Python expression that removes duplicates from values while "
                "preserving first occurrence order.",
                "list(dict.fromkeys(values))",
            ),
            (
                "dictionary-inversion",
                "Provide a Python dictionary comprehension that swaps keys and values in "
                "mapping.",
                "{value: key for key, value in mapping.items()}",
            ),
            (
                "bounded-filter",
                f"Write a Python list comprehension selecting integers x from items where "
                f"{c} <= x < {a}.",
                f"[x for x in items if {c} <= x < {a}]",
            ),
            (
                "pairwise-differences",
                "Using zip only, give a Python expression for successive differences in "
                "sequence xs.",
                "[right - left for left, right in zip(xs, xs[1:])]",
            ),
            (
                "tuple-sort-key",
                "Write a Python expression sorting rows by their second field, then their "
                "first field.",
                "sorted(rows, key=lambda row: (row[1], row[0]))",
            ),
            (
                "balanced-delimiter-check",
                "Name the asymptotic time complexity of scanning a bracket string once with "
                "a stack.",
                "O(n)",
            ),
            (
                "safe-dictionary-access",
                f"Give a Python expression reading key 'port' from config with fallback {b}.",
                f"config.get('port', {b})",
            ),
            (
                "enumerated-index-map",
                "Write a dictionary comprehension mapping each item in names to its zero-based "
                "position.",
                "{name: index for index, name in enumerate(names)}",
            ),
        ),
        "instruction following": (
            (
                "ordered-csv",
                f"Output these labels in the stated order as comma-separated text with no "
                f"spaces: elm-{a}; quartz-{b}; tide-{c}.",
                f"elm-{a},quartz-{b},tide-{c}",
            ),
            (
                "uppercase-only",
                f"Return the phrase 'harbor gate {n}' in uppercase and add nothing else.",
                f"HARBOR GATE {n}",
            ),
            (
                "reverse-order-lines",
                f"Print these tokens on separate lines in reverse order: ash-{a}, birch-{b}, "
                f"cedar-{c}.",
                f"cedar-{c}\nbirch-{b}\nash-{a}",
            ),
            (
                "bracket-wrapper",
                f"Surround value luna-{a} with one pair of square brackets. No prose.",
                f"[luna-{a}]",
            ),
            (
                "alphabetic-selection",
                f"From zeta-{a}, alpha-{b}, and mu-{c}, return only the alphabetically first "
                "label.",
                f"alpha-{b}",
            ),
            (
                "exact-repetition",
                f"Repeat syllable ko-{n} exactly three times, separated by vertical bars.",
                f"ko-{n}|ko-{n}|ko-{n}",
            ),
            (
                "case-sensitive-choice",
                "Reply with ACCEPT only if the word 'Quartz' begins with an uppercase letter; "
                "otherwise reply REJECT.",
                "ACCEPT",
            ),
            (
                "word-limit-summary",
                f"Summarize 'the red buoy entered dock {a} before dawn' in exactly four words.",
                "Red buoy arrived predawn.",
            ),
            (
                "prefix-transformation",
                f"Prepend verified: to code nova-{b}, without inserting a space.",
                f"verified:nova-{b}",
            ),
        ),
        "tool formatting": (
            (
                "nested-call-object",
                f"Encode a JSON call to tool sum_pair with arguments left={a} and right={b}. "
                "Use top-level keys tool and arguments.",
                compact_json({"arguments": {"left": a, "right": b}, "tool": "sum_pair"}),
            ),
            (
                "boolean-argument",
                f"Produce compact JSON for function set_alarm with enabled=true and hour={c}.",
                compact_json({"enabled": True, "hour": c, "tool": "set_alarm"}),
            ),
            (
                "array-payload",
                f"Represent a batch_ids invocation as JSON whose ids array is [{a},{b},{c}].",
                compact_json({"ids": [a, b, c], "tool": "batch_ids"}),
            ),
            (
                "nullable-option",
                f"Write one compact JSON object calling lookup_{n}; query is 'cedar' and "
                "cursor is null.",
                compact_json({"cursor": None, "query": "cedar", "tool": f"lookup_{n}"}),
            ),
            (
                "string-escaping",
                "Serialize a tool call named quote_text whose text argument is the literal "
                "a\"b. Return valid compact JSON.",
                compact_json({"text": 'a"b', "tool": "quote_text"}),
            ),
            (
                "nested-coordinate",
                f"Create compact JSON for move_probe with position fields x={a} and y={b}.",
                compact_json({"position": {"x": a, "y": b}, "tool": "move_probe"}),
            ),
            (
                "empty-arguments",
                "Return a JSON tool envelope for heartbeat with an empty arguments object.",
                compact_json({"arguments": {}, "tool": "heartbeat"}),
            ),
            (
                "typed-result-envelope",
                f"Format compact JSON with ok=true and result containing count={c}; do not "
                "include a tool name.",
                compact_json({"ok": True, "result": {"count": c}}),
            ),
            (
                "multi-call-list",
                f"Emit a compact JSON array: first call ping with id={a}, then call close with "
                f"id={b}.",
                compact_json(
                    [{"id": a, "tool": "ping"}, {"id": b, "tool": "close"}]
                ),
            ),
        ),
        "agentic coding": (
            (
                "isolated-test-diagnosis",
                f"Test shard_{a} fails after normalize_{b}. Name the first read-only diagnostic "
                "action in six words or fewer.",
                f"Run test shard_{a} verbosely.",
            ),
            (
                "regression-bisection",
                f"A deterministic regression appeared between commits good-{a} and bad-{b}. "
                "What Git operation best locates the first bad commit?",
                "git bisect",
            ),
            (
                "type-error-localization",
                f"Static checking reports one error in parser_{c}. What should be inspected "
                "before editing code?",
                f"Inspect the full parser_{c} diagnostic.",
            ),
            (
                "resource-leak-observation",
                f"A worker's file-descriptor count grows each cycle. State one non-mutating "
                "first check.",
                "List the worker's open descriptors.",
            ),
            (
                "race-reproduction",
                "An intermittent concurrency test fails once per thousand runs. What should be "
                "captured before proposing a patch?",
                "A deterministic reproducer and trace.",
            ),
            (
                "schema-migration-audit",
                f"Migration {a} changes a required column. Which artifact should be read before "
                "executing it against data?",
                "The migration and rollback plan.",
            ),
            (
                "api-contract-inspection",
                f"Client tests reject response field item_{b}. What should be compared before "
                "changing serialization?",
                "Compare response against the API schema.",
            ),
            (
                "performance-profile",
                f"Endpoint /scan/{c} became slow without correctness failures. Name the evidence "
                "to collect before optimization.",
                "A representative performance profile.",
            ),
            (
                "dependency-version-check",
                "A build fails only on CI after a dependency update. Give the first comparison "
                "to make.",
                "Compare resolved dependency lockfiles.",
            ),
        ),
        "Chinese": (
            (
                "zh-inventory-arithmetic",
                f"温室里有{a}棵松树和{b}棵竹子。只写植物总数。",
                str(a + b),
            ),
            (
                "zh-keyword-extraction",
                f"句子“北港的灯塔编号是星-{c}”中，灯塔编号是什么？",
                f"星-{c}",
            ),
            (
                "zh-chronology",
                "甲先关门，乙随后熄灯，丙最后离开。谁第二个行动？",
                "乙",
            ),
            (
                "zh-classification",
                f"规则规定：青-{a}属于木类，赤-{b}属于石类。"
                f"青-{a}属于哪一类？",
                "木类",
            ),
            (
                "zh-concise-rewrite",
                f"把“由于下雨，编号{c}的比赛因此推迟了”"
                "改成更简洁的中文。",
                f"因下雨，编号{c}的比赛推迟了。",
            ),
            (
                "zh-conditional-reasoning",
                "若门开着，灯就亮。现在灯没有亮。门开着吗？"
                "只回答“是”或“否”。",
                "否",
            ),
            (
                "zh-format-following",
                f"按“名称|数量”的格式输出：名称是松果，数量是{b}。",
                f"松果|{b}",
            ),
            (
                "zh-spatial-relation",
                "小桥在塔的东边，花园在小桥的南边。"
                "花园相对塔在哪个方向？",
                "东南",
            ),
            (
                "zh-sum-two-records",
                f"记录甲为{a}，记录乙为{c}。计算二者之和，只写数字。",
                str(a + c),
            ),
        ),
        "English": (
            (
                "en-codeword-recall",
                f"A fictional register maps maple-{a} to harbor-{b}. Supply maple-{a}'s mapped "
                "value alone.",
                f"harbor-{b}",
            ),
            (
                "en-article-selection",
                "Choose the correct article for the phrase '__ hour': a or an.",
                "an",
            ),
            (
                "en-past-tense",
                "Give the simple past tense of the verb 'teach'.",
                "taught",
            ),
            (
                "en-plural-agreement",
                "Complete with is or are: 'The lanterns __ ready.'",
                "are",
            ),
            (
                "en-antonym",
                "Return one direct antonym of 'scarce'.",
                "abundant",
            ),
            (
                "en-punctuation",
                f"Add terminal punctuation to this declarative sentence: Beacon {c} is active",
                f"Beacon {c} is active.",
            ),
            (
                "en-compound-order",
                "Arrange these words as a grammatical sentence: silently / snow / fell / the.",
                "The snow fell silently.",
            ),
            (
                "en-contraction-expansion",
                "Expand the contraction in 'They've finished.' without changing tense.",
                "They have finished.",
            ),
            (
                "en-count-noun",
                f"Choose fewer or less: There were __ than {a} boats.",
                "fewer",
            ),
        ),
        "mixed-language": (
            (
                "zh-en-mapping",
                f"记录写着 river-{a} 对应 puerto-{b}. Return only the mapped value.",
                f"puerto-{b}",
            ),
            (
                "es-en-translation",
                "Traduce al inglés la palabra 'puente'. Return one word.",
                "bridge",
            ),
            (
                "fr-en-selection",
                "Le registre dit couleur=bleu. Answer in English with that color.",
                "blue",
            ),
            (
                "de-en-arithmetic",
                f"Im Bericht stehen {a} rote und {c} blaue Marken. Give the total as digits.",
                str(a + c),
            ),
            (
                "ja-en-label",
                f"ラベルは moon-{b} です。Return only the label.",
                f"moon-{b}",
            ),
            (
                "pt-en-boolean",
                "A regra diz ativo=true. Responda in English: true or false?",
                "true",
            ),
            (
                "ko-en-order",
                "순서는 alpha, beta, gamma입니다. Which item is second?",
                "beta",
            ),
            (
                "it-en-lookup",
                f"Nel catalogo chiave-{c} vale stella-{a}. Output the value alone.",
                f"stella-{a}",
            ),
            (
                "ar-en-count",
                f"السجل يحتوي على {c} عناصر. How many items are recorded?",
                str(c),
            ),
        ),
        "long-context retrieval": (
            (
                "mini-direct-key",
                f"Mini archive: accession pine-{a} stores reading rv-{b}. Retrieve that reading.",
                f"rv-{b}",
            ),
            (
                "mini-table-cell",
                f"Tiny table row [channel={c}; payload=px-{a}]. What payload occupies the row?",
                f"px-{a}",
            ),
            (
                "mini-cross-reference",
                f"Note A points to card cedar-{b}; card cedar-{b} says code q-{c}. Resolve Note "
                "A to its code.",
                f"q-{c}",
            ),
            (
                "mini-latest-version",
                f"Log versions for unit {n}: old=v-{a}, current=v-{b}. Return the current value.",
                f"v-{b}",
            ),
            (
                "mini-negative-filter",
                f"Entries are birch-{a}:inactive and elm-{c}:active. Name the active entry.",
                f"elm-{c}",
            ),
            (
                "mini-coordinate-field",
                f"Beacon record {{name:Orin, east:{a}, north:{b}}}. Extract the north field.",
                str(b),
            ),
            (
                "mini-quoted-span",
                f"The curator marked the exact phrase «silver tide {c}» as the answer span. "
                "Reproduce it.",
                f"silver tide {c}",
            ),
            (
                "mini-second-occurrence",
                f"Sequence tags: k-{a}, k-{b}, k-{c}. Which tag is second?",
                f"k-{b}",
            ),
            (
                "mini-zh-field",
                f"短记录中写着“目标值：光-{a}”。请只返回目标值。",
                f"光-{a}",
            ),
        ),
        "long-context synthesis": (
            (
                "mini-addition",
                f"Two notes report east={a} and west={b}. Combine them by addition.",
                str(a + b),
            ),
            (
                "mini-range",
                f"Sensor extremes are low={c} and high={b}. Return high minus low.",
                str(b - c),
            ),
            (
                "mini-majority",
                "Three reports label the gate open, closed, and open. Give the majority label.",
                "open",
            ),
            (
                "mini-chain-resolution",
                f"Alias dawn points to dusk; dusk points to node-{a}. Resolve dawn fully.",
                f"node-{a}",
            ),
            (
                "mini-intersection",
                f"List A contains pine-{a} and oak-{b}; list B contains ash-{c} and oak-{b}. "
                "Name their shared item.",
                f"oak-{b}",
            ),
            (
                "mini-weighted-total",
                f"Record X has count {a} with weight 2; record Y has count {c} with weight 1. "
                "Return the weighted total.",
                str(2 * a + c),
            ),
            (
                "mini-order-merge",
                "Fragment one says Mira preceded Nox. Fragment two says Nox preceded Oru. Who "
                "came first?",
                "Mira",
            ),
            (
                "mini-boolean-conjunction",
                "Audit A says seal=true; audit B says checksum=true. Is their conjunction true?",
                "true",
            ),
            (
                "mini-bilingual-join",
                f"记录甲给出前缀 star，record B gives suffix {c}. "
                "Join them with a hyphen.",
                f"star-{c}",
            ),
        ),
    }
    if domain == "rare tokens":
        raise AssertionError("rare-token tasks require a selected tokenizer token")
    return tasks[domain][partition_index]


def _rare_embedding_token_ids(bundle: TokenizerBundle, count: int) -> list[int]:
    """Select stable, non-special, single-ID round-trip tokens from the rare tail."""
    candidates: list[int] = []
    # Keep clear of the final added/special-token band while selecting from the tail.
    for token_id in range(bundle.vocab_size - 128, 100_000, -1):
        decoded = bundle.tokenizer.decode([token_id], skip_special_tokens=False)
        if not decoded or not decoded.strip() or any(ch in decoded for ch in "\r\n\x00"):
            continue
        if _encode(bundle, decoded) != (token_id,):
            continue
        candidates.append(token_id)
        if len(candidates) == count:
            return candidates
    raise CorpusIntegrityError(
        "EMBEDDING_TOKEN_SELECTION_FAILED", f"found {len(candidates)} of {count} IDs"
    )


def _rare_token_task(
    partition_index: int, token_text: str, token_id: int
) -> tuple[str, str, str]:
    payload = f"\n{token_text}\n"
    cases = (
        (
            "rare-id-recall",
            f"The line between the delimiters is one official tokenizer item. State its "
            f"vocabulary ID.---{payload}---",
        ),
        (
            "rare-entry-audit",
            f"Audit this isolated vocabulary entry and return only its integer index:"
            f"<entry>{payload}</entry>",
        ),
        (
            "rare-json-index",
            f"Identify the pinned ID for the payload below; the response must be the number, "
            f"not JSON.\nPAYLOAD:{payload}END",
        ),
        (
            "rare-boundary-probe",
            f"A boundary probe contains exactly one target piece between BEGIN and STOP. Which "
            f"token ID is targeted?\nBEGIN{payload}STOP",
        ),
        (
            "rare-decoder-cell",
            f"Look up the singleton decoder cell shown after the arrow and give its numeric "
            f"address.\narrow →{payload}done",
        ),
        (
            "rare-roundtrip-check",
            f"This text round-trips to one vocabulary element. Report that element's ID and "
            f"nothing else.\n{text_for_prompt(token_text)}",
        ),
        (
            "rare-embedding-row",
            f"Select the embedding row indexed by the standalone piece inside the box. Return "
            f"the row number.\n┌─{payload}─┘",
        ),
        (
            "rare-symbol-index",
            f"Determine the official integer index of this single encoded symbol sequence:"
            f"\n[{payload}]",
        ),
        (
            "rare-vocabulary-address",
            f"The specimen following SPEC is guaranteed to encode as one item. Give its "
            f"vocabulary address.\nSPEC{payload}ENDSPEC",
        ),
    )
    family, prompt = cases[partition_index]
    return family, prompt, str(token_id)


def text_for_prompt(value: str) -> str:
    """Keep a rare-token payload visibly isolated without escaping its bytes."""
    return f"<<<\n{value}\n>>>"


def build_core_records(bundle: TokenizerBundle) -> list[CorpusRecord]:
    selected_ids = _rare_embedding_token_ids(bundle, len(PARTITIONS))
    records: list[CorpusRecord] = []
    for partition_index, partition in enumerate(PARTITIONS):
        for domain in DOMAINS:
            record_id = f"core-{partition_index:02d}-{_slug(domain)}"
            embedding_ids: tuple[int, ...] = ()
            if domain == "rare tokens":
                token_id = selected_ids[partition_index]
                token_text = bundle.tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                family, prompt, answer = _rare_token_task(
                    partition_index, token_text, token_id
                )
                embedding_ids = (token_id,)
            else:
                family, prompt, answer = _domain_task(domain, partition_index)
            source_document_id = (
                f"urn:hawking:glm52:corpus:v2:{_slug(partition)}:{_slug(domain)}"
            )
            answer_segment = f"Expected-answer contract for {record_id}: {answer}"
            source_document = f"{prompt}\n{answer_segment}"
            records.append(
                _make_record(
                    bundle,
                    record_id=record_id,
                    partition=partition,
                    domain=domain,
                    kind="core",
                    document_family_id=f"{_slug(domain)}/{family}",
                    source_document_id=source_document_id,
                    source_document=source_document,
                    atomic_segments=(prompt, answer_segment),
                    context_window=prompt,
                    prompt=prompt,
                    expected_answer=answer,
                    provenance=_provenance(record_id, "domain-family-cell-v2"),
                    embedding_claim_token_ids=embedding_ids,
                )
            )
    return records


def _filler_segment(record_id: str, index: int, style_index: int) -> str:
    digest = _sha256_text(f"{GENERATOR_SEED}/{record_id}/filler/{index}")
    number = int(digest[:12], 16) % 10_000_019
    relation = int(digest[12:24], 16) % 1_000_003
    object_id = digest[24:40]
    styles = (
        f"Archive row {index:06d} in {record_id}: object {object_id} has reading {number} "
        f"and relation {relation}.",
        f"{record_id}\t{index:06d}\tobject={object_id}\treading={number}\tlink={relation}",
        f"档案{record_id}第{index:06d}行：对象{object_id}，"
        f"读数{number}，关联{relation}。",
        f'{{"doc":"{record_id}","row":{index},"object":"{object_id}",'
        f'"reading":{number},"relation":{relation}}}',
        f"Observation {object_id} / notebook {record_id}: measurement {number}; "
        f"reference channel {relation}; ordinal {index}.",
        f"LOG[{index:06d}] scope={record_id} entity={object_id} metric={number} "
        f"edge={relation}",
        f'<row n="{index}" doc="{record_id}"><object>{object_id}</object>'
        f"<reading>{number}</reading><relation>{relation}</relation></row>",
        f"Memo item {index} for {record_id} pairs catalog mark {object_id} with value "
        f"{number}; its cross-reference is {relation}.",
        f"Registro {index:06d} ({record_id}): clave {object_id}; valeur {number}; "
        f"连接 {relation}.",
    )
    return styles[style_index % len(styles)]


def _position_bucket(fraction: float) -> str:
    if fraction < 0.12:
        return "opening"
    if fraction < 0.35:
        return "early"
    if fraction < 0.65:
        return "middle"
    if fraction < 0.88:
        return "late"
    return "closing"


def _long_context_task(
    *,
    partition_index: int,
    domain: str,
    key_a: str,
    value_a: int,
    key_b: str | None,
    value_b: int | None,
) -> tuple[str, list[str], tuple[str, ...], str, str]:
    """Construct genuinely different query/evidence families for each split."""
    if domain == "long-context retrieval":
        marker = f"rv-{value_a}"
        cases = (
            (
                "accession-payload",
                f"Accession {key_a} resolves to payload {marker}.",
                f"Locate accession {key_a} in the archive and transcribe its payload without "
                "commentary.",
            ),
            (
                "dispatch-destination",
                f"Dispatch card {key_a} assigns destination {marker}.",
                f"Which destination is assigned on dispatch card {key_a}? Give only the code.",
            ),
            (
                "zh-ledger-reading",
                f"账本键 {key_a} 的读数为 {marker}。",
                f"在长账本中查找键 {key_a}，只抄写它的读数。",
            ),
            (
                "json-status-field",
                f'{{"lookup":"{key_a}","status":"{marker}"}}',
                f"Scan the records for lookup {key_a}; return the status field from its JSON "
                "entry.",
            ),
            (
                "observation-channel",
                f"Observation bearing tag {key_a} reports channel {marker}.",
                f"Recover the channel reported by the observation tagged {key_a}.",
            ),
            (
                "log-result-code",
                f"LOG target={key_a} result={marker} state=closed",
                f"From the diagnostic log, extract the result associated with target {key_a}; "
                "omit the field name.",
            ),
            (
                "xml-value-node",
                f'<target id="{key_a}"><value>{marker}</value></target>',
                f"Find the XML target whose id is {key_a} and reproduce the text of its value "
                "node.",
            ),
            (
                "memo-cross-reference",
                f"The memo's cross-reference for {key_a} is written as {marker}.",
                f"Consult the memo collection: what cross-reference belongs to {key_a}?",
            ),
            (
                "multilingual-catalog",
                f"Registro {key_a}: valeur cible {marker}; 状态已核验。",
                f"Search the mixed-language catalog for {key_a} and answer with its valeur "
                "cible only.",
            ),
        )
        family, evidence, prompt = cases[partition_index]
        return family, [evidence], (marker,), marker, prompt

    if key_b is None or value_b is None:
        raise AssertionError("synthesis needs two evidence keys")
    marker_a = f"{key_a}@{value_a}"
    marker_b = f"{key_b}@{value_b}"
    cases = (
        (
            "sum-distant-counters",
            f"Counter evidence {marker_a} records the eastern quantity.",
            f"Counter evidence {marker_b} records the western quantity.",
            str(value_a + value_b),
            f"Find counters {key_a} and {key_b} in the archive, add their quantities, and "
            "return the integer total.",
        ),
        (
            "absolute-gap",
            f"Lower-bound slip {marker_a} has been authenticated.",
            f"Upper-bound slip {marker_b} has been authenticated.",
            str(abs(value_a - value_b)),
            f"Using slips {key_a} and {key_b}, calculate the absolute difference between "
            "their recorded numbers.",
        ),
        (
            "maximum-reading",
            f"传感器甲记录 {marker_a}。",
            f"传感器乙记录 {marker_b}。",
            str(max(value_a, value_b)),
            f"比较长记录中 {key_a} 与 {key_b} 的数值，只返回较大的整数。",
        ),
        (
            "modular-composition",
            f'{{"factor":"{marker_a}","role":"left"}}',
            f'{{"factor":"{marker_b}","role":"right"}}',
            str((value_a + 2 * value_b) % 1_000_003),
            f"Resolve factors {key_a} and {key_b}; compute (left + 2×right) modulo 1000003.",
        ),
        (
            "greatest-common-divisor",
            f"Notebook alpha contains measurement {marker_a}.",
            f"Notebook beta contains measurement {marker_b}.",
            str(math.gcd(value_a, value_b)),
            f"Retrieve measurements {key_a} and {key_b}, then report their greatest common "
            "divisor.",
        ),
        (
            "bitwise-exclusive-or",
            f"TRACE operand_a={marker_a} accepted=true",
            f"TRACE operand_b={marker_b} accepted=true",
            str(value_a ^ value_b),
            f"Extract operands {key_a} and {key_b} from the trace and give their bitwise XOR "
            "as a decimal integer.",
        ),
        (
            "ordered-subtraction",
            f'<minuend key="{key_a}">{marker_a}</minuend>',
            f'<subtrahend key="{key_b}">{marker_b}</subtrahend>',
            str(value_a - value_b),
            f"Read the XML values identified by {key_a} and {key_b}; subtract the second from "
            "the first.",
        ),
        (
            "weighted-merge",
            f"Planning memo marks primary input as {marker_a}.",
            f"Planning memo marks auxiliary input as {marker_b}.",
            str(2 * value_a + value_b),
            f"Synthesize memo inputs {key_a} and {key_b} with rule 2×primary + auxiliary. "
            "Output only the result.",
        ),
        (
            "range-span",
            f"Registro bajo: {marker_a}; confirmé.",
            f"记录上限：{marker_b}；已确认。",
            str(max(value_a, value_b) - min(value_a, value_b)),
            f"Across the bilingual records for {key_a} and {key_b}, compute the numeric span "
            "from the smaller value to the larger.",
        ),
    )
    family, evidence_a, evidence_b, expected, prompt = cases[partition_index]
    return family, [evidence_a, evidence_b], (marker_a, marker_b), expected, prompt


def _make_long_record(
    bundle: TokenizerBundle,
    *,
    partition_index: int,
    rung: int,
) -> CorpusRecord:
    partition = PARTITIONS[partition_index]
    rung_label = CONTEXT_RUNG_LABELS[rung]
    domain = (
        "long-context retrieval"
        if (partition_index + ADMITTED_CONTEXT_RUNGS.index(rung)) % 2 == 0
        else "long-context synthesis"
    )
    record_id = f"ladder-{rung_label.lower()}-{partition_index:02d}-{_slug(domain)}"
    fraction = POSITION_FRACTIONS[
        (partition_index + 2 * ADMITTED_CONTEXT_RUNGS.index(rung))
        % len(POSITION_FRACTIONS)
    ]
    key_a = f"needle-{_sha256_text(record_id + '/a')[:16]}"
    value_a = 10_000 + int(_sha256_text(record_id + "/va")[:8], 16) % 800_000
    key_b = (
        f"needle-{_sha256_text(record_id + '/b')[:16]}"
        if domain == "long-context synthesis"
        else None
    )
    value_b = (
        10_000 + int(_sha256_text(record_id + "/vb")[:8], 16) % 800_000
        if domain == "long-context synthesis"
        else None
    )
    family, evidence_segments, evidence_markers, expected, prompt = _long_context_task(
        partition_index=partition_index,
        domain=domain,
        key_a=key_a,
        value_a=value_a,
        key_b=key_b,
        value_b=value_b,
    )

    # Calibrate filler length against the *full* tokenizer.  Per-line token counts are
    # only an initial estimate because byte-level BPE boundaries change at joins.
    sample = _filler_segment(record_id, 0, partition_index) + "\n"
    average = max(1, len(_encode(bundle, sample)))
    evidence_budget = sum(len(_encode(bundle, row + "\n")) for row in evidence_segments)
    filler_target = max(256, rung - evidence_budget)
    row_count = max(8, math.ceil(filler_target / average))
    for _ in range(8):
        segments = [
            _filler_segment(record_id, index, partition_index)
            for index in range(row_count)
        ]
        filler_actual = len(_encode(bundle, "\n".join(segments)))
        if filler_target <= filler_actual <= filler_target + average:
            break
        adjusted = max(8, round(row_count * filler_target / max(1, filler_actual)))
        if adjusted == row_count:
            adjusted += 1 if filler_actual < filler_target else -1
        row_count = adjusted
    segments = [
        _filler_segment(record_id, index, partition_index)
        for index in range(row_count)
    ]

    primary_index = min(len(segments) - 1, round(fraction * (len(segments) - 1)))
    segments.insert(primary_index, evidence_segments[0])
    evidence_indices = [primary_index]
    if len(evidence_segments) == 2:
        complement = 1.0 - fraction
        secondary_index = min(
            len(segments), max(0, round(complement * (len(segments) - 1)))
        )
        if secondary_index == primary_index:
            secondary_index = min(len(segments), secondary_index + 1)
        segments.insert(secondary_index, evidence_segments[1])
        # Insertion before the primary shifts its final index.
        if secondary_index <= evidence_indices[0]:
            evidence_indices[0] += 1
        evidence_indices.append(secondary_index)

    context = "\n".join(segments)
    actual = len(_encode(bundle, context))
    while actual < rung:
        gap_rows = max(1, math.ceil((rung - actual) / average))
        first = 100_000 + len(segments)
        segments.extend(
            _filler_segment(record_id, first + offset, partition_index)
            for offset in range(gap_rows)
        )
        context = "\n".join(segments)
        actual = len(_encode(bundle, context))
    while actual > rung + 256:
        excess_rows = max(1, math.floor((actual - rung) / average))
        removable = [
            index
            for index in range(len(segments) - 1, -1, -1)
            if segments[index] not in evidence_segments
        ][:excess_rows]
        if not removable:
            raise CorpusIntegrityError(
                "CONTEXT_CONSTRUCTION_FAILED", f"cannot trim {record_id}"
            )
        for index in sorted(removable, reverse=True):
            del segments[index]
        context = "\n".join(segments)
        actual = len(_encode(bundle, context))
        if actual < rung:
            first = 200_000 + len(segments)
            segments.append(_filler_segment(record_id, first, partition_index))
            context = "\n".join(segments)
            actual = len(_encode(bundle, context))

    # Recompute evidence indices and observed position after any tail growth.
    evidence_indices = [
        next(index for index, segment in enumerate(segments) if marker in segment)
        for marker in evidence_markers
    ]
    observed_fraction = evidence_indices[0] / max(1, len(segments) - 1)
    source_document = context
    return _make_record(
        bundle,
        record_id=record_id,
        partition=partition,
        domain=domain,
        kind="context_ladder",
        document_family_id=f"{_slug(domain)}/{family}",
        source_document_id=f"urn:hawking:glm52:corpus:v2:{record_id}",
        source_document=source_document,
        atomic_segments=segments,
        context_window=context,
        prompt=prompt,
        expected_answer=expected,
        provenance=_provenance(record_id, f"long-context-{rung_label}-v2"),
        context_rung_tokens=rung,
        evidence_markers=evidence_markers,
        evidence_segment_indices=evidence_indices,
        position_fraction=observed_fraction,
        position_bucket=_position_bucket(observed_fraction),
    )


def build_long_context_records(bundle: TokenizerBundle) -> list[CorpusRecord]:
    return [
        _make_long_record(bundle, partition_index=partition_index, rung=rung)
        for rung in ADMITTED_CONTEXT_RUNGS
        for partition_index in range(len(PARTITIONS))
    ]


def build_records(bundle: TokenizerBundle) -> list[CorpusRecord]:
    return build_core_records(bundle) + build_long_context_records(bundle)


def _record_hashes(record: CorpusRecord) -> dict[str, Any]:
    atomic_hashes = [_normalized_hash(segment) for segment in record.atomic_segments]
    return {
        "source_document_sha256": _sha256_text(record.source_document),
        "segment_hash": _merkleish_hash(atomic_hashes),
        "atomic_segment_count": len(atomic_hashes),
        "atomic_segment_hashes_sha256": _merkleish_hash(sorted(atomic_hashes)),
        "context_window_hash": _sha256_text(record.context_window),
        "prompt_hash": _normalized_hash(record.prompt),
        "expected_answer_hash": _sha256_text(record.expected_answer),
    }


def _raise(code: str, detail: str) -> None:
    raise CorpusIntegrityError(code, detail)


def _train_vs_evaluation(left: CorpusRecord, right: CorpusRecord) -> bool:
    return (
        left.partition in TRAIN_PARTITIONS and right.partition in EVALUATION_PARTITIONS
    ) or (
        right.partition in TRAIN_PARTITIONS and left.partition in EVALUATION_PARTITIONS
    )


def _similarity_witness(
    score: float, left: CorpusRecord, right: CorpusRecord
) -> dict[str, Any]:
    return {
        "score": round(score, 6),
        "left_record_id": left.record_id,
        "right_record_id": right.record_id,
        "left_partition": left.partition,
        "right_partition": right.partition,
    }


def _semantic_similarity_audit(
    records: Sequence[CorpusRecord], bundle: TokenizerBundle
) -> dict[str, Any]:
    """Hard-fail cross-split near duplicates using two independent shingle views."""
    prepared: dict[str, dict[str, Any]] = {}
    for record in records:
        skeleton = _semantic_skeleton(record.prompt)
        prepared[record.record_id] = {
            "skeleton": skeleton,
            "character": _shingles(tuple(skeleton), CHAR_SHINGLE_WIDTH),
            "token": _shingles(_encode(bundle, skeleton), TOKEN_SHINGLE_WIDTH),
        }

    maxima: dict[str, dict[str, dict[str, Any]]] = {
        method: {
            "all_cross_split": {"score": 0.0},
            "train_vs_evaluation": {"score": 0.0},
        }
        for method in ("character", "token")
    }
    pairs_evaluated = 0
    train_eval_pairs = 0
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left.partition == right.partition:
                continue
            pairs_evaluated += 1
            is_train_eval = _train_vs_evaluation(left, right)
            train_eval_pairs += int(is_train_eval)
            scores = {
                method: _jaccard(
                    prepared[left.record_id][method], prepared[right.record_id][method]
                )
                for method in ("character", "token")
            }
            for method, score in scores.items():
                if score > maxima[method]["all_cross_split"]["score"]:
                    maxima[method]["all_cross_split"] = _similarity_witness(
                        score, left, right
                    )
                if (
                    is_train_eval
                    and score > maxima[method]["train_vs_evaluation"]["score"]
                ):
                    maxima[method]["train_vs_evaluation"] = _similarity_witness(
                        score, left, right
                    )

            char_limit = (
                CHAR_SHINGLE_MAX_TRAIN_VS_EVAL
                if is_train_eval
                else CHAR_SHINGLE_MAX_ALL_CROSS_SPLIT
            )
            token_limit = (
                TOKEN_SHINGLE_MAX_TRAIN_VS_EVAL
                if is_train_eval
                else TOKEN_SHINGLE_MAX_ALL_CROSS_SPLIT
            )
            if scores["character"] > char_limit:
                _raise(
                    "CROSS_SPLIT_NEAR_DUPLICATE",
                    f"character-{CHAR_SHINGLE_WIDTH} Jaccard={scores['character']:.6f} "
                    f"> {char_limit:.6f}: {left.record_id} vs {right.record_id}",
                )
            if scores["token"] > token_limit:
                _raise(
                    "CROSS_SPLIT_NEAR_DUPLICATE",
                    f"token-{TOKEN_SHINGLE_WIDTH} Jaccard={scores['token']:.6f} "
                    f"> {token_limit:.6f}: {left.record_id} vs {right.record_id}",
                )

    return {
        "status": "PASS",
        "semantic_redaction": (
            "case/whitespace normalized; decimal numbers and long generated identifiers "
            "replaced by <variable>"
        ),
        "pairs_evaluated": pairs_evaluated,
        "train_vs_evaluation_pairs": train_eval_pairs,
        "character_shingle": {
            "width": CHAR_SHINGLE_WIDTH,
            "jaccard_threshold_all_cross_split": CHAR_SHINGLE_MAX_ALL_CROSS_SPLIT,
            "jaccard_threshold_train_vs_evaluation": (
                CHAR_SHINGLE_MAX_TRAIN_VS_EVAL
            ),
            "maximum_observed": maxima["character"],
        },
        "official_token_id_shingle": {
            "width": TOKEN_SHINGLE_WIDTH,
            "jaccard_threshold_all_cross_split": TOKEN_SHINGLE_MAX_ALL_CROSS_SPLIT,
            "jaccard_threshold_train_vs_evaluation": (
                TOKEN_SHINGLE_MAX_TRAIN_VS_EVAL
            ),
            "maximum_observed": maxima["token"],
        },
    }


def validate_records(
    records: Sequence[CorpusRecord],
    bundle: TokenizerBundle,
    *,
    verify_tokenization: bool = True,
) -> dict[str, Any]:
    """Validate every Part-IX split and integrity gate, failing on first defect."""
    if not records:
        _raise("EMPTY_CORPUS", "no records were provided")
    ids = [record.record_id for record in records]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        _raise("DUPLICATE_RECORD_ID", f"duplicate IDs: {duplicate_ids[:3]}")

    unknown_partitions = sorted({r.partition for r in records} - set(PARTITIONS))
    if unknown_partitions:
        _raise("UNKNOWN_PARTITION", repr(unknown_partitions))
    unknown_domains = sorted({r.domain for r in records} - set(DOMAINS))
    if unknown_domains:
        _raise("UNKNOWN_DOMAIN", repr(unknown_domains))

    provenance_fields = {
        "origin_type",
        "source_locator",
        "generator_id",
        "generator_seed_sha256",
        "recipe",
        "license",
        "network_access",
    }
    for record in records:
        missing = provenance_fields - set(record.provenance)
        if missing or record.provenance.get("network_access") is not False:
            _raise(
                "MISSING_PROVENANCE",
                f"{record.record_id}: missing={sorted(missing)}, "
                f"network_access={record.provenance.get('network_access')!r}",
            )
        if not record.source_document_id or not record.atomic_segments:
            _raise("MISSING_PROVENANCE", f"{record.record_id}: empty source/segments")

    # A declared document/task family may repeat across context rungs only inside one
    # partition.  Separately, a number/identifier-redacted prompt skeleton must not
    # cross a partition boundary; this defeats ID- or number-salted templates.
    family_owners: dict[str, set[str]] = defaultdict(set)
    family_records: dict[str, list[CorpusRecord]] = defaultdict(list)
    skeleton_records: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        if not record.document_family_id or "/" not in record.document_family_id:
            _raise("MISSING_DOCUMENT_FAMILY", record.record_id)
        family_owners[record.document_family_id].add(record.partition)
        family_records[record.document_family_id].append(record)
        skeleton_records[_sha256_text(_semantic_skeleton(record.prompt))].append(record)
    for family, partitions in family_owners.items():
        if len(partitions) > 1:
            _raise(
                "CROSS_SPLIT_DOCUMENT_FAMILY",
                f"family={family}, partitions={sorted(partitions)}",
            )
    for family, rows in family_records.items():
        if len(rows) <= 1:
            continue
        if any(row.kind != "context_ladder" for row in rows) or len(
            {row.context_rung_tokens for row in rows}
        ) != len(rows):
            _raise(
                "REPEATED_SEMANTIC_FAMILY_INFLATION",
                f"family={family}, records={[row.record_id for row in rows]}",
            )
    # Source-document identity and content are both disjoint across partitions.
    for axis_name, identity in (
        ("source_document_id", lambda r: r.source_document_id),
        ("source_document_sha256", lambda r: _sha256_text(r.source_document)),
    ):
        owners: dict[str, set[str]] = defaultdict(set)
        for record in records:
            owners[identity(record)].add(record.partition)
        leaked = [key for key, partitions in owners.items() if len(partitions) > 1]
        if leaked:
            _raise(
                "CROSS_SPLIT_SOURCE_DOCUMENT",
                f"{axis_name} crosses partitions: {leaked[0]}",
            )

    # Repeated atomic segments are forbidden even within one split: duplicates cannot
    # inflate sample counts.  Cross-split duplicates are additionally context leakage.
    atomic_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        for segment in record.atomic_segments:
            normalized = _normalized(segment)
            if not normalized:
                _raise("EMPTY_SEGMENT", record.record_id)
            atomic_owners[_sha256_text(normalized)].append(
                (record.partition, record.record_id)
            )
    repeated = {key: rows for key, rows in atomic_owners.items() if len(rows) > 1}
    if repeated:
        key, rows = next(iter(repeated.items()))
        _raise("REPEATED_SEGMENT_INFLATION", f"segment={key}, records={rows[:3]}")

    context_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        context_owners[_sha256_text(record.context_window)].append(
            (record.partition, record.record_id)
        )
    for digest, rows in context_owners.items():
        if len({partition for partition, _ in rows}) > 1:
            _raise("CROSS_SPLIT_CONTEXT_OVERLAP", f"context={digest}, records={rows}")

    prompt_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        prompt_owners[_normalized_hash(record.prompt)].append(
            (record.partition, record.record_id)
        )
    for digest, rows in prompt_owners.items():
        partitions = {partition for partition, _ in rows}
        if partitions & TRAIN_PARTITIONS and partitions & EVALUATION_PARTITIONS:
            _raise("EVALUATION_PROMPT_LEAKAGE", f"prompt={digest}, records={rows}")
        if len(partitions) > 1:
            _raise("CROSS_SPLIT_PROMPT_OVERLAP", f"prompt={digest}, records={rows}")

    for digest, record_rows in skeleton_records.items():
        rows = [(row.partition, row.record_id) for row in record_rows]
        if len({partition for partition, _ in rows}) > 1:
            _raise(
                "CROSS_SPLIT_SEMANTIC_FAMILY",
                f"redacted_prompt={digest}, records={rows}",
            )
        if len(record_rows) > 1 and (
            any(row.kind != "context_ladder" for row in record_rows)
            or len({row.context_rung_tokens for row in record_rows})
            != len(record_rows)
        ):
            _raise(
                "REPEATED_SEMANTIC_FAMILY_INFLATION",
                f"redacted_prompt={digest}, records={rows}",
            )

    similarity = _semantic_similarity_audit(records, bundle)

    # Every split has exactly one core item per domain.  Long probes are reported
    # separately, preventing their token volume from hiding a missing core domain.
    core_counts = Counter(
        (record.partition, record.domain) for record in records if record.kind == "core"
    )
    bad_cells = [
        (partition, domain, core_counts[(partition, domain)])
        for partition in PARTITIONS
        for domain in DOMAINS
        if core_counts[(partition, domain)] != 1
    ]
    if bad_cells:
        _raise("HIDDEN_DOMAIN_IMBALANCE", f"non-unit core cells: {bad_cells[:3]}")

    # Embedding-determined target IDs must be represented by the official tokenizer
    # and owned by one partition only.  General prompt tokens are intentionally not
    # treated as embedding claims.
    embedding_owners: dict[int, set[str]] = defaultdict(set)
    for record in records:
        for token_id in record.embedding_claim_token_ids:
            if token_id < 0 or token_id >= bundle.vocab_size:
                _raise(
                    "EMBEDDING_TOKEN_OUT_OF_RANGE", f"{record.record_id}: {token_id}"
                )
            if token_id not in _encode(bundle, record.context_window):
                _raise(
                    "EMBEDDING_TOKEN_NOT_OBSERVED", f"{record.record_id}: {token_id}"
                )
            embedding_owners[token_id].add(record.partition)
    if not embedding_owners:
        _raise("EMBEDDING_TOKEN_SPLIT_MISSING", "no embedding-claim token IDs")
    for token_id, partitions in embedding_owners.items():
        if len(partitions) != 1:
            _raise(
                "CROSS_SPLIT_EMBEDDING_TOKEN",
                f"token_id={token_id}, partitions={sorted(partitions)}",
            )

    long_records = [record for record in records if record.kind == "context_ladder"]
    rung_partition_counts = Counter(
        (record.context_rung_tokens, record.partition) for record in long_records
    )
    expected_rung_cells = {
        (rung, partition) for rung in ADMITTED_CONTEXT_RUNGS for partition in PARTITIONS
    }
    observed_rung_cells = set(rung_partition_counts)
    if observed_rung_cells != expected_rung_cells or any(
        count != 1 for count in rung_partition_counts.values()
    ):
        _raise(
            "CONTEXT_LADDER_INCOMPLETE",
            f"missing={sorted(expected_rung_cells - observed_rung_cells)[:3]}, "
            f"extra={sorted(observed_rung_cells - expected_rung_cells)[:3]}",
        )
    if any(record.context_rung_tokens in {262_144, 1_048_576} for record in records):
        _raise(
            "UNADMITTED_CONTEXT_RUNG",
            "256K requires resource evidence and 1M requires exact-runtime evidence",
        )

    prompt_position_words = re.compile(r"\b(position|offset|line\s+\d+)\b", re.I)
    rung_buckets: dict[int, set[str]] = defaultdict(set)
    partition_positions: dict[str, set[str]] = defaultdict(set)
    for record in long_records:
        assert record.context_rung_tokens is not None
        if record.token_count < record.context_rung_tokens:
            _raise(
                "CONTEXT_RUNG_UNDERSIZED",
                f"{record.record_id}: {record.token_count} < {record.context_rung_tokens}",
            )
        # A full generated row is never truncated (which would make provenance less
        # legible), so permit one conservative row of tokenizer overshoot.
        if record.token_count > record.context_rung_tokens + 256:
            _raise(
                "CONTEXT_RUNG_OVERSIZED",
                f"{record.record_id}: {record.token_count} exceeds tolerance",
            )
        if prompt_position_words.search(record.prompt):
            _raise("POSITION_ONLY_LEAKAGE", f"position hint in {record.record_id}")
        if not record.evidence_segment_indices or not record.evidence_markers:
            _raise("POSITION_ONLY_LEAKAGE", f"missing evidence in {record.record_id}")
        if len(record.evidence_segment_indices) != len(record.evidence_markers):
            _raise("POSITION_ONLY_LEAKAGE", f"evidence arity in {record.record_id}")
        for marker, index in zip(
            record.evidence_markers, record.evidence_segment_indices
        ):
            if index < 0 or index >= len(record.atomic_segments):
                _raise("POSITION_ONLY_LEAKAGE", f"bad evidence index in {record.record_id}")
            if marker not in record.atomic_segments[index]:
                _raise(
                    "POSITION_ONLY_LEAKAGE",
                    f"marker/index mismatch in {record.record_id}",
                )
            occurrences = sum(marker in segment for segment in record.atomic_segments)
            if occurrences != 1:
                _raise(
                    "POSITION_ONLY_LEAKAGE",
                    f"marker occurs {occurrences} times in {record.record_id}",
                )
        observed_fraction = record.evidence_segment_indices[0] / max(
            1, len(record.atomic_segments) - 1
        )
        if record.position_fraction is None or not math.isclose(
            observed_fraction, record.position_fraction, abs_tol=1e-12
        ):
            _raise("POSITION_ONLY_LEAKAGE", f"false position in {record.record_id}")
        observed_bucket = _position_bucket(observed_fraction)
        if observed_bucket != record.position_bucket:
            _raise("POSITION_ONLY_LEAKAGE", f"false bucket in {record.record_id}")
        rung_buckets[record.context_rung_tokens].add(observed_bucket)
        partition_positions[record.partition].add(observed_bucket)
    for rung, buckets in rung_buckets.items():
        if buckets != set(POSITION_BUCKETS):
            _raise(
                "POSITION_ONLY_LEAKAGE",
                f"{CONTEXT_RUNG_LABELS[rung]} buckets={sorted(buckets)}",
            )
    for partition, buckets in partition_positions.items():
        if len(buckets) < 3:
            _raise(
                "POSITION_ONLY_LEAKAGE",
                f"{partition} exercises only {sorted(buckets)}",
            )

    if verify_tokenization:
        for record in records:
            ids = _encode(bundle, record.context_window)
            if len(ids) != record.token_count or _token_id_hash(ids) != record.token_ids_sha256:
                _raise("TOKENIZATION_TAMPER", record.record_id)

    token_counts_by_domain = Counter()
    record_counts_by_domain = Counter()
    for record in records:
        token_counts_by_domain[record.domain] += record.token_count
        record_counts_by_domain[record.domain] += 1

    return {
        "status": "PASS",
        "record_count": len(records),
        "core_record_count": sum(record.kind == "core" for record in records),
        "long_context_record_count": len(long_records),
        "atomic_segment_count": sum(len(record.atomic_segments) for record in records),
        "unique_source_document_ids": len({r.source_document_id for r in records}),
        "unique_source_document_hashes": len(
            {_sha256_text(r.source_document) for r in records}
        ),
        "unique_segment_hashes": len(atomic_owners),
        "unique_context_window_hashes": len(context_owners),
        "unique_prompt_hashes": len(prompt_owners),
        "embedding_claim_unique_token_ids": len(embedding_owners),
        "document_family_count": len(family_owners),
        "semantic_prompt_skeleton_count": len(skeleton_records),
        "matched_ladder_repeat_records_not_independent_samples": sum(
            len(rows) - 1 for rows in family_records.values() if len(rows) > 1
        ),
        "semantic_similarity": similarity,
        "record_counts_by_domain": dict(sorted(record_counts_by_domain.items())),
        "token_counts_by_domain": dict(sorted(token_counts_by_domain.items())),
        "context_rung_actual_token_ranges": {
            CONTEXT_RUNG_LABELS[rung]: {
                "minimum": min(
                    r.token_count for r in long_records if r.context_rung_tokens == rung
                ),
                "maximum": max(
                    r.token_count for r in long_records if r.context_rung_tokens == rung
                ),
            }
            for rung in ADMITTED_CONTEXT_RUNGS
        },
    }


def _record_manifest_entry(record: CorpusRecord) -> dict[str, Any]:
    value: dict[str, Any] = {
        "record_id": record.record_id,
        "partition": record.partition,
        "domain": record.domain,
        "kind": record.kind,
        "document_family_id": record.document_family_id,
        "semantic_prompt_skeleton_sha256": _sha256_text(
            _semantic_skeleton(record.prompt)
        ),
        "source_document_id": record.source_document_id,
        **_record_hashes(record),
        "token_count": record.token_count,
        "token_ids_sha256": record.token_ids_sha256,
        "embedding_claim_token_ids": list(record.embedding_claim_token_ids),
        "provenance": record.provenance,
    }
    if record.context_rung_tokens is not None:
        value["context_probe"] = {
            "rung": CONTEXT_RUNG_LABELS[record.context_rung_tokens],
            "minimum_tokens": record.context_rung_tokens,
            "actual_tokens": record.token_count,
            "evidence_count": len(record.evidence_markers),
            "evidence_marker_hashes": [
                _sha256_text(marker) for marker in record.evidence_markers
            ],
            "evidence_segment_indices": list(record.evidence_segment_indices),
            "position_fraction": record.position_fraction,
            "position_bucket": record.position_bucket,
        }
    return value


def build_manifest(
    bundle: TokenizerBundle,
    records: Sequence[CorpusRecord] | None = None,
) -> dict[str, Any]:
    records = list(records) if records is not None else build_records(bundle)
    validation = validate_records(records, bundle, verify_tokenization=True)
    entries = [_record_manifest_entry(record) for record in records]
    core_matrix = {
        partition: {
            domain: sum(
                record.kind == "core"
                and record.partition == partition
                and record.domain == domain
                for record in records
            )
            for domain in DOMAINS
        }
        for partition in PARTITIONS
    }
    embedding_ownership = {
        str(token_id): record.partition
        for record in records
        for token_id in record.embedding_claim_token_ids
    }
    import tokenizers

    requirements_path = REPO_ROOT / "tools/condense/requirements-glm52.txt"
    common_path = REPO_ROOT / "tools/condense/glm52_common.py"
    tokenizers_module = Path(tokenizers.__file__).resolve()
    environment_root = Path(sys.prefix).resolve()
    environment_config = environment_root / "pyvenv.cfg"
    environment_text = (
        environment_config.read_text(encoding="utf-8")
        if environment_config.is_file() else ""
    )
    if sys.prefix == sys.base_prefix \
            or not re.search(
                r"(?im)^include-system-site-packages\s*=\s*false\s*$",
                environment_text,
            ) \
            or not tokenizers_module.is_relative_to(environment_root):
        _raise(
            "CORPUS_RUNTIME_NOT_ISOLATED",
            "the corpus manifest requires the pinned no-system-site GLM environment",
        )
    unsigned = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": {
            "model_payload_downloaded": False,
            "network_access_used": False,
            "artifact_kind": "offline deterministic corpus integrity manifest",
            "text_storage": "hashes and generator recipes; rebuild required for text",
            "capability_claim_permitted": False,
            "real_model_scores_required_for_capability_claim": True,
        },
        "official_tokenizer": {
            "repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "file": "tokenizer.json",
            # Host cache roots and content-addressed blob paths are deliberately
            # excluded from the seal.  The immutable repository/revision, file
            # name, byte count, and digest are the portable identity; recording
            # an absolute cache path made identical offline rebuilds differ
            # across machines (and even across two valid local cache views).
            "portable_locator": (
                f"hf://{MODEL_REPOSITORY}@{REVISION}/tokenizer.json"
            ),
            "host_path_sealed": False,
            "sha256": bundle.sha256,
            "bytes": bundle.byte_count,
            "vocabulary_size": bundle.vocab_size,
            "loader": "tokenizers.Tokenizer.from_file",
            "local_files_only": True,
        },
        "deterministic_builder": {
            "generator_id": GENERATOR_ID,
            "generator_seed_sha256": _sha256_text(GENERATOR_SEED),
            "builder_path": "tools/condense/glm52_corpus.py",
            "builder_sha256": sha256_file(Path(__file__)),
            "instrument_sha256": {
                "tools/condense/glm52_common.py": sha256_file(common_path),
                "tools/condense/requirements-glm52.txt": sha256_file(
                    requirements_path
                ),
                "tokenizers_import_module": sha256_file(tokenizers_module),
            },
            "runtime": {
                "python": platform.python_version(),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "fully_pinned_requirements": True,
                "system_site_packages": False,
                "tokenizers_import_within_environment": True,
                "host_paths_sealed": False,
            },
            "record_entries_sha256": _sha256_bytes(canonical(entries)),
        },
        "partitions": [
            {
                "name": partition,
                "role": (
                    "training"
                    if partition in TRAIN_PARTITIONS
                    else "evaluation"
                    if partition in EVALUATION_PARTITIONS
                    else "validation"
                ),
            }
            for partition in PARTITIONS
        ],
        "split_contract": {
            "source_document": "disjoint identity and content hash across partitions",
            "segment_hash": "all normalized atomic segments globally unique",
            "context_window_hash": "disjoint across partitions",
            "domain": "exact one-cell core stratification in every partition",
            "document_family": "declared semantic family owned by one partition only",
            "redacted_prompt_family": (
                "numeric/identifier-redacted skeleton disjoint across partitions"
            ),
            "near_duplicate": (
                "character and official-token shingle Jaccard hard limits; stricter for "
                "training versus evaluation"
            ),
            "embedding_claim_unique_token_id": "each target ID owned by one partition",
        },
        "domains": list(DOMAINS),
        "domain_balance": {
            "policy": "exactly one core record per partition/domain cell; long probes separate",
            "core_record_matrix": core_matrix,
            "raw_counts_required_with_every_aggregate": True,
            "per_domain_metrics_required": True,
        },
        "semantic_family_contract": {
            "core_family_reuse_permitted": False,
            "cross_partition_family_reuse_permitted": False,
            "matched_ladder_family_reuse": (
                "permitted only within one partition across distinct context rungs"
            ),
            "matched_ladder_repeat_records_count": validation[
                "matched_ladder_repeat_records_not_independent_samples"
            ],
            "matched_ladder_repeats_count_as_independent_capability_samples": False,
            "generated_long_context_distractor_rows_count_as_independent_samples": False,
            "near_duplicate_comparison_unit": "redacted query/prompt text",
        },
        "context_ladder": [
            {
                "rung": CONTEXT_RUNG_LABELS[rung],
                "tokens": rung,
                "admission": "ADMITTED",
                "records": len(PARTITIONS),
                "position_balancing": "all five disclosed buckets per rung",
            }
            for rung in ADMITTED_CONTEXT_RUNGS
        ]
        + [
            {
                "rung": "256K",
                "tokens": 262_144,
                "admission": "NOT_ADMITTED_RESOURCE_VALIDATION_PENDING",
                "records": 0,
                "condition": "admit only with sealed resource-valid execution evidence",
            },
            {
                "rung": "1M",
                "tokens": 1_048_576,
                "admission": "NOT_ADMITTED_EXACT_RUNTIME_PENDING",
                "records": 0,
                "condition": "admit only when the exact runtime executes it safely",
                "short_context_preservation_claim_permitted": False,
            },
        ],
        "embedding_claim_split": {
            "target_token_ownership": dict(sorted(embedding_ownership.items())),
            "general_prompt_token_overlap_is_not_an_embedding_claim": True,
        },
        "integrity_gates": {
            "repeated_segment_inflation": "PASS",
            "cross_split_context_overlap": "PASS",
            "evaluation_prompt_leakage": "PASS",
            "cross_split_document_family": "PASS",
            "number_or_identifier_salted_template": "PASS",
            "repeated_semantic_family_inflation": "PASS",
            "character_shingle_near_duplicate": "PASS",
            "official_token_shingle_near_duplicate": "PASS",
            "missing_provenance": "PASS",
            "position_only_leakage": "PASS",
            "domain_imbalance_hidden_by_averages": "PASS",
            "tamper_evident_seal": "PASS",
        },
        "quality_metric_contract": {
            "metrics": list(QUALITY_METRICS),
            "aggregate_only_reporting_permitted": False,
            "raw_sample_and_token_counts_required": True,
            "corpus_manifest_is_not_a_model_quality_result": True,
            "capability_claim_requires_real_scored_execution": True,
        },
        "validation": validation,
        "records": entries,
        "limitations": [
            "Corpus integrity is green; no model-quality result is implied.",
            "256K is withheld pending resource-valid execution evidence.",
            "1M is withheld pending safe execution by the exact runtime.",
            "No 1M preservation claim may be inferred from admitted shorter rungs.",
        ],
    }
    return seal(unsigned)


def render_markdown(manifest: dict[str, Any]) -> str:
    verify_sealed(manifest, label="GLM52_CORPUS_INTEGRITY")
    validation = manifest["validation"]
    similarity = validation["semantic_similarity"]
    character = similarity["character_shingle"]
    token = similarity["official_token_id_shingle"]
    lines = [
        "# GLM-5.2 Corpus Integrity",
        "",
        f"**Status:** {manifest['status']}",
        "",
        "This is an offline corpus-integrity admission, not a model-quality result. "
        "No model payload was fetched.",
        "",
        "## Bound tokenizer",
        "",
        f"- Repository: `{MODEL_REPOSITORY}`",
        f"- Revision: `{REVISION}`",
        f"- SHA-256: `{manifest['official_tokenizer']['sha256']}`",
        f"- Vocabulary: {manifest['official_tokenizer']['vocabulary_size']:,} IDs",
        "- Loader: direct local `tokenizer.json`; network disabled",
        "",
        "## Coverage",
        "",
        f"- Nine partitions: {len(manifest['partitions'])}",
        f"- Domains per core split: {len(manifest['domains'])}",
        f"- Core records: {validation['core_record_count']}",
        f"- Long-context records: {validation['long_context_record_count']}",
        f"- Atomic segments: {validation['atomic_segment_count']:,}",
        f"- Declared document families: {validation['document_family_count']}",
        "- Matched ladder repeats not counted as independent samples: "
        f"{validation['matched_ladder_repeat_records_not_independent_samples']}",
        f"- Embedding-claim target IDs: {validation['embedding_claim_unique_token_ids']}",
        "",
        "## Semantic and near-duplicate admission",
        "",
        "Numbers and long generated identifiers are redacted before comparison. Exact "
        "redacted skeleton reuse across splits fails, independently of the shingle gates.",
        "",
        "| View | Width | All-split limit | Maximum | Train/eval limit | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Character shingles | {character['width']} | "
        f"{character['jaccard_threshold_all_cross_split']:.3f} | "
        f"{character['maximum_observed']['all_cross_split']['score']:.6f} | "
        f"{character['jaccard_threshold_train_vs_evaluation']:.3f} | "
        f"{character['maximum_observed']['train_vs_evaluation']['score']:.6f} |",
        f"| Official-token-ID shingles | {token['width']} | "
        f"{token['jaccard_threshold_all_cross_split']:.3f} | "
        f"{token['maximum_observed']['all_cross_split']['score']:.6f} | "
        f"{token['jaccard_threshold_train_vs_evaluation']:.3f} | "
        f"{token['maximum_observed']['train_vs_evaluation']['score']:.6f} |",
        "",
        f"Pairs checked: {similarity['pairs_evaluated']:,} cross-split, including "
        f"{similarity['train_vs_evaluation_pairs']:,} training-versus-evaluation pairs.",
        "",
        "## Context ladder",
        "",
        "| Rung | Admission | Records | Exact policy |",
        "|---:|---|---:|---|",
    ]
    for rung in manifest["context_ladder"]:
        policy = rung.get("condition", rung.get("position_balancing", ""))
        lines.append(
            f"| {rung['rung']} | {rung['admission']} | {rung['records']} | {policy} |"
        )
    lines.extend(
        [
            "",
            "## Hard-fail gates",
            "",
        ]
    )
    # ``atomic_json`` canonicalizes object keys.  Sort here as well so the
    # rendered companion is byte-identical whether it is produced from the
    # in-memory manifest or reconstructed from the sealed JSON artifact.
    for gate, status in sorted(manifest["integrity_gates"].items()):
        lines.append(f"- {gate.replace('_', ' ')}: **{status}**")
    lines.extend(
        [
            "",
            "Every core split contains exactly one record for every required domain. "
            "Long-context token volume is reported separately and cannot hide a missing "
            "domain cell. Source-document identity/content, normalized segment, context "
            "window, prompt, and embedding-claim token ownership are independently checked.",
            "",
            "Matched query families reused across context lengths remain inside one partition "
            "and are counted as ladder controls, not independent capability samples. Generated "
            "distractor rows likewise contribute context length, not sample count.",
            "",
            "This admission establishes corpus hygiene only. A capability claim remains "
            "forbidden until the real model has produced sealed per-domain and per-rung scores.",
            "",
            "The 256K rung is not admitted without resource-valid execution evidence. "
            "The 1M rung is explicitly not admitted until the exact runtime executes it "
            "safely; shorter tests do not support a 1M preservation claim.",
            "",
            f"Manifest seal: `{manifest['seal_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    manifest: dict[str, Any],
    *,
    json_path: Path = OUTPUT_JSON,
    markdown_path: Path = OUTPUT_MARKDOWN,
) -> None:
    verify_sealed(manifest, label="GLM52_CORPUS_INTEGRITY")
    atomic_json(json_path, manifest)
    atomic_text(markdown_path, render_markdown(manifest))


def verify_artifact(path: Path, bundle: TokenizerBundle) -> dict[str, Any]:
    observed = read_sealed_json(path)
    expected = build_manifest(bundle)
    if canonical(observed) != canonical(expected):
        _raise(
            "DETERMINISTIC_REBUILD_MISMATCH",
            f"{path} does not match an offline reconstruction",
        )
    return observed


def adversarial_selfcheck(
    bundle: TokenizerBundle,
    records: Sequence[CorpusRecord] | None = None,
) -> dict[str, Any]:
    """Exercise representative fail-closed mutations without writing anything."""
    records = list(records) if records is not None else build_records(bundle)
    expected_codes: list[tuple[str, list[CorpusRecord]]] = []

    missing = list(records)
    missing[0] = replace(missing[0], provenance={})
    expected_codes.append(("MISSING_PROVENANCE", missing))

    overlap = list(records)
    source = overlap[0]
    target_index = next(
        index for index, row in enumerate(overlap) if row.partition != source.partition
    )
    overlap[target_index] = replace(
        overlap[target_index], context_window=source.context_window
    )
    expected_codes.append(("CROSS_SPLIT_CONTEXT_OVERLAP", overlap))

    position = list(records)
    long_index = next(i for i, row in enumerate(position) if row.kind == "context_ladder")
    wrong_bucket = (
        "closing" if position[long_index].position_bucket != "closing" else "opening"
    )
    position[long_index] = replace(position[long_index], position_bucket=wrong_bucket)
    expected_codes.append(("POSITION_ONLY_LEAKAGE", position))

    source = next(row for row in records if row.partition in TRAIN_PARTITIONS)
    evaluation_index = next(
        index
        for index, row in enumerate(records)
        if row.partition in EVALUATION_PARTITIONS
    )

    family = list(records)
    family[evaluation_index] = replace(
        family[evaluation_index], document_family_id=source.document_family_id
    )
    expected_codes.append(("CROSS_SPLIT_DOCUMENT_FAMILY", family))

    number_salted = re.sub(
        r"\d+", lambda match: str(int(match.group(0)) + 7001), source.prompt
    )
    semantic = list(records)
    semantic[evaluation_index] = replace(
        semantic[evaluation_index], prompt=number_salted
    )
    expected_codes.append(("CROSS_SPLIT_SEMANTIC_FAMILY", semantic))

    within_index = next(
        index
        for index, row in enumerate(records)
        if row.partition == source.partition and row.record_id != source.record_id
    )
    within = list(records)
    within[within_index] = replace(within[within_index], prompt=number_salted)
    expected_codes.append(("REPEATED_SEMANTIC_FAMILY_INFLATION", within))

    near = list(records)
    near[evaluation_index] = replace(
        near[evaluation_index],
        prompt=source.prompt + " Please preserve every instruction.",
    )
    expected_codes.append(("CROSS_SPLIT_NEAR_DUPLICATE", near))

    observed: list[str] = []
    for expected, mutated in expected_codes:
        try:
            validate_records(mutated, bundle, verify_tokenization=False)
        except CorpusIntegrityError as exc:
            observed.append(exc.code)
            if exc.code != expected:
                raise AssertionError(f"expected {expected}, observed {exc.code}") from exc
        else:  # pragma: no cover - a regression is intentionally fatal
            raise AssertionError(f"mutation was accepted: {expected}")

    tampered = seal({"schema": SCHEMA, "status": "PASS", "selfcheck": True})
    tampered["status"] = "FAIL"
    try:
        verify_sealed(tampered)
    except Glm52Error:
        seal_rejected = True
    else:  # pragma: no cover
        seal_rejected = False
    if not seal_rejected:
        raise AssertionError("seal accepted a tampered manifest")
    return {
        "status": "PASS",
        "mutation_codes_rejected": observed,
        "tampered_manifest_seal_rejected": seal_rejected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "verify", "selfcheck"), nargs="?", default="build"
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--json-output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=OUTPUT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = load_pinned_tokenizer(args.tokenizer_path)
    if args.command == "build":
        manifest = build_manifest(bundle)
        write_artifacts(
            manifest, json_path=args.json_output, markdown_path=args.markdown_output
        )
        result: dict[str, Any] = {
            "status": "PASS",
            "json": str(args.json_output),
            "markdown": str(args.markdown_output),
            "seal_sha256": manifest["seal_sha256"],
            "record_count": manifest["validation"]["record_count"],
        }
    elif args.command == "verify":
        manifest = verify_artifact(args.json_output, bundle)
        result = {
            "status": "PASS",
            "verified": str(args.json_output),
            "seal_sha256": manifest["seal_sha256"],
        }
    else:
        result = adversarial_selfcheck(bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
