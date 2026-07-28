#!/usr/bin/env python3.12
"""Run the frozen GLM-5.2 2K/8K/32K context ladder against one exact artifact.

This is a quality gate, not a speed benchmark. It rebuilds the deterministic
long-context records from the pinned tokenizer, sends greedy requests to the
base runtime, and grades exact planted retrieval/synthesis answers. The server
must attest the candidate index SHA-256 and ``fallback_present=false`` before
and after the run.

Without ``--run`` this command only prints the future heavy plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "tools/eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import glm52_corpus as corpus  # noqa: E402
import support_halo_gate as halo  # noqa: E402


RUNG_TOKENS = {
    "2K": 2_048,
    "8K": 8_192,
    "32K": 32_768,
    "128K": 131_072,
}
DEFAULT_RUNGS = ("2K", "8K", "32K")
CORPUS_MANIFEST = ROOT / "GLM52_CORPUS_INTEGRITY.json"


class LongContextGateError(RuntimeError):
    """The gate could not produce attributable context-quality evidence."""


def score_completion(completion: str, expected: str) -> bool:
    text = completion.strip()
    first = text.splitlines()[0].strip() if text else ""
    return first.casefold() == expected.strip().casefold()


def summarize(rows: list[dict[str, Any]], rung_labels: list[str]) -> dict[str, Any]:
    by_rung: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rung[row["rung"]].append(row)
    expected = set(rung_labels)
    if set(by_rung) != expected:
        raise LongContextGateError("result rung set differs from the requested set")
    rungs = []
    for label in rung_labels:
        samples = by_rung[label]
        if len(samples) != len(corpus.PARTITIONS):
            raise LongContextGateError(
                f"{label}: expected {len(corpus.PARTITIONS)} partition samples, "
                f"found {len(samples)}"
            )
        passes = sum(bool(row["passed"]) for row in samples)
        rungs.append(
            {
                "rung": label,
                "tokens": RUNG_TOKENS[label],
                "passes": passes,
                "total": len(samples),
                "pass_rate": passes / len(samples),
                "all_passed": passes == len(samples),
            }
        )
    return {
        "rungs": rungs,
        "verdict": "PASS" if all(row["all_passed"] for row in rungs) else "FAIL",
    }


def selected_records(
    bundle: corpus.TokenizerBundle,
    rung_labels: list[str],
) -> list[corpus.CorpusRecord]:
    all_records = corpus.build_records(bundle)
    corpus.validate_records(all_records, bundle, verify_tokenization=True)
    selected_tokens = {RUNG_TOKENS[label] for label in rung_labels}
    return [
        record
        for record in all_records
        if record.kind == "context_ladder"
        and record.context_rung_tokens in selected_tokens
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8899")
    parser.add_argument("--model", help="optional expected model id")
    parser.add_argument(
        "--rung",
        action="append",
        choices=tuple(RUNG_TOKENS),
        help="repeat to select rungs; default is 2K, 8K, and 32K",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute whole-model generation; default prints the plan only",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    artifact = args.artifact.expanduser().resolve()
    index = halo.artifact_index(artifact)
    index_sha = halo.sha256_file(index)
    rung_labels = list(dict.fromkeys(args.rung or DEFAULT_RUNGS))
    if args.max_new_tokens <= 0:
        raise LongContextGateError("--max-new-tokens must be positive")

    serve_address = (
        args.endpoint.removeprefix("http://")
        if args.endpoint.startswith("http://127.0.0.1:")
        else "<endpoint-address>"
    )
    serve_command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "hawking",
        "--",
        "serve",
        "--gravity",
        str(artifact),
        "--addr",
        serve_address,
    ]
    plan = {
        "mode": "dry-run",
        "heavy_execution_started": False,
        "artifact": str(artifact),
        "artifact_index": index.name,
        "artifact_index_sha256": index_sha,
        "rungs": rung_labels,
        "tasks_per_rung": len(corpus.PARTITIONS),
        "total_tasks": len(rung_labels) * len(corpus.PARTITIONS),
        "corpus_manifest": str(CORPUS_MANIFEST),
        "endpoint": args.endpoint,
        "serve_command": serve_command,
        "note": "rerun with --run only after capability and runtime agreement pass",
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.out is None:
        raise LongContextGateError("--out is required with --run")

    before = halo.fetch_runtime_attestation(
        args.endpoint,
        expected_index_sha256=index_sha,
        requested_model=args.model,
    )
    model = before["model_id"]
    bundle = corpus.load_pinned_tokenizer()
    records = selected_records(bundle, rung_labels)
    manifest = json.loads(CORPUS_MANIFEST.read_text())
    corpus.verify_sealed(manifest, label="GLM52_CORPUS_INTEGRITY")

    rows = []
    completions = []
    for record in records:
        prompt = f"{record.context_window}\n\nQUESTION\n{record.prompt}"
        completion, _elapsed = halo.call_endpoint(
            args.endpoint,
            model,
            prompt,
            args.max_new_tokens,
            args.timeout,
        )
        passed = score_completion(completion, record.expected_answer)
        rung = corpus.CONTEXT_RUNG_LABELS[record.context_rung_tokens]
        completions.append({"id": record.record_id, "completion": completion})
        rows.append(
            {
                "record_id": record.record_id,
                "partition": record.partition,
                "domain": record.domain,
                "rung": rung,
                "context_tokens": record.token_count,
                "position_bucket": record.position_bucket,
                "expected_answer_sha256": hashlib.sha256(
                    record.expected_answer.encode()
                ).hexdigest(),
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "passed": passed,
            }
        )

    after = halo.fetch_runtime_attestation(
        args.endpoint,
        expected_index_sha256=index_sha,
        requested_model=model,
    )
    summary = summarize(rows, rung_labels)
    completions_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in completions
    )
    out_path = args.out.expanduser().resolve()
    completions_path = out_path.with_name(f"{out_path.stem}.completions.jsonl")
    completions_path.parent.mkdir(parents=True, exist_ok=True)
    completions_path.write_text(completions_text)

    receipt = {
        "schema": "hawking.glm52.long_context_gate.v1",
        "artifact": {
            "address_kind": "model_index_sha256",
            "index": index.name,
            "index_sha256": index_sha,
        },
        "runtime": {
            "model": model,
            "temperature": 0,
            "pre_attestation": before,
            "post_attestation": after,
        },
        "corpus": {
            "schema": manifest["schema"],
            "seal_sha256": manifest["seal_sha256"],
            "builder_sha256": manifest["deterministic_builder"]["builder_sha256"],
            "tokenizer_sha256": bundle.sha256,
            "rungs": rung_labels,
            "matched_ladder_repeats_are_independent_samples": False,
        },
        "completion_evidence": {
            "sha256": hashlib.sha256(completions_text.encode()).hexdigest(),
            "tasks": len(completions),
        },
        "quality": summary,
        "samples": rows,
        "claims": {
            "quality_only": True,
            "speed": False,
            "sustained_runtime": False,
            "rungs_above_tested": False,
        },
        "verdict": summary["verdict"],
    }
    halo.write_json(out_path, receipt)
    print(
        json.dumps(
            {
                "wrote": str(out_path),
                "completions": str(completions_path),
                "quality": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        LongContextGateError,
        corpus.CorpusIntegrityError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"verdict": "ERROR", "error": str(error)}, indent=2))
        raise SystemExit(2) from error
