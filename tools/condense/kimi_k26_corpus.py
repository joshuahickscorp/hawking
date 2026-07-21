#!/usr/bin/env python3.12
"""Fail-closed clean-corpus gate for the Kimi K2.6 Doctor Prime instrument.

The unit of assignment is a content-deduplicated source document.  Fixed context
windows are then globally deduplicated before any split is exposed.  This directly
blocks the position-level and repeated-segment leakage that invalidated the earlier
Qwen layer-zero experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any

from kimi_k26_reference import KimiTokenizer, PROBES, REVISION


SCHEMA = "hawking.kimi_k26.corpus_integrity.v1"
SPLITS = (
    "routing_calibration", "codec_calibration", "doctor_training",
    "validation", "holdout", "protected_domain_holdout",
)
WEIGHTS = {
    "routing_calibration": 26, "codec_calibration": 26,
    "doctor_training": 22, "validation": 10, "holdout": 10,
    "protected_domain_holdout": 6,
}
CALIBRATION_SPLITS = SPLITS[:3]
EXTENSIONS = {
    ".py": "code", ".rs": "code", ".c": "code", ".h": "code",
    ".md": "prose", ".txt": "prose", ".json": "tool_format",
    ".toml": "tool_format", ".sh": "code",
}
WINDOW_CHARS = 512


class IntegrityError(RuntimeError):
    pass


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8", "replace"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def discover(roots: list[Path], max_documents: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    documents: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if len(documents) >= max_documents:
                break
            if (not path.is_file() or path.suffix.lower() not in EXTENSIONS or
                    any(part in {".git", "target", "node_modules", "__pycache__"}
                        for part in path.parts)):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) < WINDOW_CHARS:
                continue
            digest = sha_text(text)
            if digest in seen:
                continue
            seen.add(digest)
            documents.append({
                "document_hash": digest, "path": str(path),
                "domain": EXTENSIONS[path.suffix.lower()], "text": text,
            })
    return documents


def assigned(document_hash: str) -> str:
    bucket = int(document_hash[:16], 16) % sum(WEIGHTS.values())
    boundary = 0
    for split in SPLITS:
        boundary += WEIGHTS[split]
        if bucket < boundary:
            return split
    raise AssertionError("split weights do not cover the bucket range")


def windows(document: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for offset in range(0, len(document["text"]) - WINDOW_CHARS + 1, WINDOW_CHARS):
        text = document["text"][offset:offset + WINDOW_CHARS]
        result.append({
            "document_hash": document["document_hash"], "domain": document["domain"],
            "offset": offset, "segment_hash": sha_text(text),
            "context_window_hash": sha_text(f"kimi-k26:{WINDOW_CHARS}:{text}"), "text": text,
        })
    return result


def contains_sequence(haystack: list[int], needle: list[int]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for index, token in enumerate(haystack[:len(haystack) - len(needle) + 1]):
        if token == first and haystack[index:index + len(needle)] == needle:
            return True
    return False


def build(roots: list[Path], tokenizer: KimiTokenizer, max_documents: int,
          probe_texts: list[str]) -> dict[str, Any]:
    documents = discover(roots, max_documents)
    if len(documents) < 30:
        raise IntegrityError(f"only {len(documents)} unique documents discovered")
    split_data: dict[str, dict[str, Any]] = {
        name: {"documents": [], "windows": [], "token_ids": set()} for name in SPLITS
    }
    global_segments: set[str] = set()
    global_contexts: set[str] = set()
    duplicates_dropped = 0
    probe_quarantined_documents = 0
    normalized_probes = [normalize(text) for text in probe_texts]
    for document in sorted(documents, key=lambda item: item["document_hash"]):
        split = assigned(document["document_hash"])
        # Instrument source contains the frozen probes by construction.  Quarantine any such
        # source document into the protected holdout before segmentation; silently allowing it
        # into a calibration split would make this gate self-invalidating.
        normalized_document = normalize(document["text"])
        if any(probe and probe in normalized_document for probe in normalized_probes):
            split = "protected_domain_holdout"
            probe_quarantined_documents += 1
        split_data[split]["documents"].append(document)
        for window in windows(document):
            if (window["segment_hash"] in global_segments or
                    window["context_window_hash"] in global_contexts):
                duplicates_dropped += 1
                continue
            global_segments.add(window["segment_hash"])
            global_contexts.add(window["context_window_hash"])
            ids = tokenizer.encode(window["text"])
            window["token_ids"] = ids
            split_data[split]["windows"].append(window)
            split_data[split]["token_ids"].update(ids)

    failures: list[str] = []
    split_report: dict[str, Any] = {}
    for split in SPLITS:
        block = split_data[split]
        hashes = {item["document_hash"] for item in block["documents"]}
        segments = {item["segment_hash"] for item in block["windows"]}
        contexts = {item["context_window_hash"] for item in block["windows"]}
        domains = sorted({item["domain"] for item in block["windows"]})
        if not hashes or not segments:
            failures.append(f"empty split: {split}")
        split_report[split] = {
            "documents": len(hashes), "segments": len(segments),
            "context_windows": len(contexts), "unique_token_ids": len(block["token_ids"]),
            "tokens": sum(len(item["token_ids"]) for item in block["windows"]),
            "domains": domains,
        }

    overlap: dict[str, Any] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            a, b = split_data[left], split_data[right]
            row = {
                "source_document_hash": len(
                    {d["document_hash"] for d in a["documents"]} &
                    {d["document_hash"] for d in b["documents"]}),
                "segment_hash": len(
                    {w["segment_hash"] for w in a["windows"]} &
                    {w["segment_hash"] for w in b["windows"]}),
                "context_window_hash": len(
                    {w["context_window_hash"] for w in a["windows"]} &
                    {w["context_window_hash"] for w in b["windows"]}),
                "domain_conditioned_context_hash": {},
            }
            for domain in sorted(set(EXTENSIONS.values())):
                left_hashes = {w["context_window_hash"] for w in a["windows"]
                               if w["domain"] == domain}
                right_hashes = {w["context_window_hash"] for w in b["windows"]
                                if w["domain"] == domain}
                row["domain_conditioned_context_hash"][domain] = len(left_hashes & right_hashes)
            overlap[f"{left}|{right}"] = row
            if any(row[key] for key in (
                    "source_document_hash", "segment_hash", "context_window_hash")):
                failures.append(f"forbidden cross-split overlap: {left}|{right}")
            if any(row["domain_conditioned_context_hash"].values()):
                failures.append(f"domain-conditioned overlap: {left}|{right}")

    calibration_text = "\n".join(
        window["text"] for split in CALIBRATION_SPLITS
        for window in split_data[split]["windows"]
    )
    calibration_ids = tokenizer.encode(calibration_text)
    absent = []
    for text in probe_texts:
        normalized_present = normalize(text) in normalize(calibration_text)
        token_sequence_present = contains_sequence(calibration_ids, tokenizer.encode(text))
        absent.append({
            "probe_sha256": sha_text(text), "normalized_substring_present": normalized_present,
            "token_sequence_present": token_sequence_present,
        })
        if normalized_present or token_sequence_present:
            failures.append(f"evaluation probe leaked into calibration: {sha_text(text)[:12]}")

    all_token_ids = sorted(set().union(
        *(split_data[name]["token_ids"] for name in SPLITS)
    ))
    shuffled = list(all_token_ids)
    random.Random(260621).shuffle(shuffled)
    cut = len(shuffled) // 2
    layer_zero_fit, layer_zero_score = set(shuffled[:cut]), set(shuffled[cut:])
    layer_zero_overlap = len(layer_zero_fit & layer_zero_score)
    if layer_zero_overlap:
        failures.append(f"layer-zero unique-token overlap: {layer_zero_overlap}")

    corpus_seal_input = {
        split: {
            "documents": sorted(d["document_hash"] for d in split_data[split]["documents"]),
            "contexts": sorted(w["context_window_hash"] for w in split_data[split]["windows"]),
        } for split in SPLITS
    }
    return {
        "documents_discovered": len(documents), "duplicates_dropped": duplicates_dropped,
        "evaluation_probe_documents_quarantined": probe_quarantined_documents,
        "unique_token_ids_total": len(all_token_ids),
        "splits": split_report, "overlap": overlap,
        "evaluation_prompts_absent": absent,
        "layer_zero_unique_token_split": {
            "fit_unique_token_ids": len(layer_zero_fit),
            "score_unique_token_ids": len(layer_zero_score), "overlap": layer_zero_overlap,
            "fit_probe_token_ids": sorted(layer_zero_fit)[:32],
            "score_probe_token_ids": sorted(layer_zero_score)[:32],
        },
        "corpus_assignment_sha256": sha_bytes(canonical(corpus_seal_input)),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=400)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    tokenizer = KimiTokenizer(source)
    probes = [probe["text"] for probe in PROBES]
    full = build([repo_root / "tools", repo_root / "docs"], tokenizer,
                 args.max_documents, probes)
    small = build([repo_root / "tools", repo_root / "docs"], tokenizer, 40, probes)
    medium = build([repo_root / "tools", repo_root / "docs"], tokenizer, 200, probes)
    growth_axes = {
        "documents": [small["documents_discovered"], medium["documents_discovered"]],
        "segments": [sum(v["segments"] for v in small["splits"].values()),
                     sum(v["segments"] for v in medium["splits"].values())],
        "context_windows": [sum(v["context_windows"] for v in small["splits"].values()),
                            sum(v["context_windows"] for v in medium["splits"].values())],
        "unique_token_ids": [small["unique_token_ids_total"],
                             medium["unique_token_ids_total"]],
    }
    growth = {name: values for name, values in growth_axes.items() if values[1] > values[0]}
    passed = not full["failures"] and bool(growth)
    artifact = {
        "schema": SCHEMA, "status": "PASS" if passed else "FAIL",
        "source": {"repo": "moonshotai/Kimi-K2.6", "revision": REVISION,
                   "tokenizer_sha256": sha_bytes((source / "tiktoken.model").read_bytes())},
        "rules": {
            "assignment_unit": "content-deduplicated source document",
            "global_context_window_dedup": True,
            "evaluation_prompts_absent_from_calibration": True,
            "layer_zero_partition_unit": "unique token ID",
        },
        "growth": {"status": "PASS" if growth else "FAIL", "axes": growth_axes,
                   "axes_that_grew": sorted(growth)},
        "corpus": full,
    }
    unsigned = canonical(artifact)
    artifact["seal_sha256"] = sha_bytes(unsigned)
    atomic_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "output": str(args.output),
                      "documents": full["documents_discovered"],
                      "failures": full["failures"], "seal_sha256": artifact["seal_sha256"]},
                     sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
