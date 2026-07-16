#!/usr/bin/env python3
"""Build and verify fail-closed per-tier/rate block-thread profiles.

Only production receipts covering every required candidate (8/12/16/20 by
default) can qualify an entry. Selection re-hashes and revalidates every bound
receipt; there is no nearest-tier, nearest-rate, or synthetic fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

RECEIPT_SCHEMA = "hawking.strand.tier-rate-thread-canary.v1"
PROFILE_SCHEMA = "hawking.strand.thread-profile.v1"
PIPELINE_RECEIPT_SCHEMA = "hawking.strand.quantize-model-ordered-pipeline-parity.v1"
DEFAULT_THREADS = (8, 12, 16, 20)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{field} must be a non-empty, whitespace-stable string")
    return value


def validate_receipt(
    receipt: dict[str, Any],
    *,
    expected_binary_sha256: str,
    allowed_threads: Iterable[int],
) -> dict[str, Any]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ContractError(f"receipt schema must be {RECEIPT_SCHEMA!r}")
    if receipt.get("status") != "pass":
        raise ContractError("receipt status must be 'pass'")
    if receipt.get("scope") != "production" or receipt.get("synthetic") is not False:
        raise ContractError("receipt must explicitly bind scope='production' and synthetic=false")
    tier = require_nonempty_string(receipt.get("tier"), "tier")
    rate = require_nonempty_string(receipt.get("rate"), "rate")
    threads = receipt.get("threads")
    allowed = tuple(int(value) for value in allowed_threads)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads not in allowed:
        raise ContractError(f"threads must be one of {allowed}")
    binary_sha256 = require_sha256(receipt.get("binary_sha256"), "binary_sha256")
    if binary_sha256 != expected_binary_sha256:
        raise ContractError("receipt binary SHA does not match the profile binary")
    source_sha256 = require_sha256(receipt.get("source_sha256"), "source_sha256")
    canonical_sha256 = require_sha256(
        receipt.get("canonical_output_sha256"), "canonical_output_sha256"
    )
    output_sha256 = require_sha256(receipt.get("output_sha256"), "output_sha256")
    if receipt.get("exact_output") is not True or output_sha256 != canonical_sha256:
        raise ContractError("receipt does not prove exact canonical output")
    wall_seconds = receipt.get("wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0
    ):
        raise ContractError("wall_seconds must be finite and greater than zero")
    peak_rss_bytes = receipt.get("peak_rss_bytes")
    scratch_budget_bytes = receipt.get("scratch_budget_bytes")
    if not isinstance(peak_rss_bytes, int) or isinstance(peak_rss_bytes, bool) or peak_rss_bytes <= 0:
        raise ContractError("peak_rss_bytes must be a positive integer")
    if (
        not isinstance(scratch_budget_bytes, int)
        or isinstance(scratch_budget_bytes, bool)
        or scratch_budget_bytes <= 0
    ):
        raise ContractError("scratch_budget_bytes must be a positive integer")
    mode = receipt.get("mode", "block_parallel")
    if mode not in {"block_parallel", "ordered_pipeline"}:
        raise ContractError("mode must be block_parallel or ordered_pipeline")
    pipeline_receipt_sha256 = receipt.get("pipeline_receipt_sha256")
    pipeline_receipt_path = receipt.get("pipeline_receipt_path")
    if mode == "ordered_pipeline":
        require_sha256(pipeline_receipt_sha256, "pipeline_receipt_sha256")
        require_nonempty_string(pipeline_receipt_path, "pipeline_receipt_path")
    elif pipeline_receipt_sha256 is not None:
        require_sha256(pipeline_receipt_sha256, "pipeline_receipt_sha256")
        if pipeline_receipt_path is not None:
            require_nonempty_string(pipeline_receipt_path, "pipeline_receipt_path")
    return {
        "tier": tier,
        "rate": rate,
        "threads": threads,
        "binary_sha256": binary_sha256,
        "source_sha256": source_sha256,
        "canonical_output_sha256": canonical_sha256,
        "output_sha256": output_sha256,
        "wall_seconds": float(wall_seconds),
        "peak_rss_bytes": peak_rss_bytes,
        "scratch_budget_bytes": scratch_budget_bytes,
        "mode": mode,
        "pipeline_receipt_sha256": pipeline_receipt_sha256,
        "pipeline_receipt_path": pipeline_receipt_path,
    }


def validate_pipeline_binding(receipt: dict[str, Any], receipt_path: Path) -> None:
    if receipt["mode"] != "ordered_pipeline":
        return
    pipeline_path = Path(receipt["pipeline_receipt_path"])
    if not pipeline_path.is_absolute():
        pipeline_path = receipt_path.parent / pipeline_path
    pipeline_path = pipeline_path.resolve()
    if not pipeline_path.is_file():
        raise ContractError(f"bound pipeline receipt is missing: {pipeline_path}")
    if sha256_file(pipeline_path) != receipt["pipeline_receipt_sha256"]:
        raise ContractError(f"bound pipeline receipt changed: {pipeline_path}")
    pipeline = load_json(pipeline_path)
    if (
        pipeline.get("schema") != PIPELINE_RECEIPT_SCHEMA
        or pipeline.get("status") != "pass"
        or pipeline.get("scope") != "production"
        or pipeline.get("exact_output") is not True
        or pipeline.get("canonical_order") is not True
        or pipeline.get("dense_exact_match") is not True
        or pipeline.get("sidecar_exact_match") is not True
        or pipeline.get("packed_v2_exact_match") is not True
        or pipeline.get("production_promotion_allowed") is not True
    ):
        raise ContractError(
            "pipeline receipt is not an exact, ordered quantize-model production receipt"
        )


def _entry_key(tier: str, rate: str) -> str:
    return json.dumps([tier, rate], separators=(",", ":"), ensure_ascii=False)


def build_profile(
    receipt_paths: Iterable[Path],
    *,
    expected_binary_sha256: str,
    rss_limit_bytes: int,
    required_threads: Iterable[int] = DEFAULT_THREADS,
) -> dict[str, Any]:
    expected_binary_sha256 = require_sha256(expected_binary_sha256, "expected_binary_sha256")
    requested = tuple(int(value) for value in required_threads)
    required = tuple(sorted(set(requested)))
    if required != requested or not required:
        raise ContractError("required thread candidates must be non-empty, unique, and sorted")
    if any(value not in DEFAULT_THREADS for value in required):
        raise ContractError(f"required thread candidates must be drawn from {DEFAULT_THREADS}")
    if not isinstance(rss_limit_bytes, int) or isinstance(rss_limit_bytes, bool) or rss_limit_bytes <= 0:
        raise ContractError("rss_limit_bytes must be a positive integer")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for original_path in receipt_paths:
        path = Path(original_path).resolve()
        receipt = load_json(path)
        normalized = validate_receipt(
            receipt,
            expected_binary_sha256=expected_binary_sha256,
            allowed_threads=required,
        )
        validate_pipeline_binding(normalized, path)
        normalized["receipt_path"] = str(path)
        normalized["receipt_sha256"] = sha256_file(path)
        groups.setdefault((normalized["tier"], normalized["rate"]), []).append(normalized)

    entries: dict[str, Any] = {}
    all_qualified = bool(groups)
    for (tier, rate), receipts in sorted(groups.items()):
        blockers: list[str] = []
        by_threads: dict[int, dict[str, Any]] = {}
        for receipt in receipts:
            threads = receipt["threads"]
            if threads in by_threads:
                blockers.append(f"duplicate receipt for {threads} threads")
            else:
                by_threads[threads] = receipt
        missing = [threads for threads in required if threads not in by_threads]
        if missing:
            blockers.append(f"missing exact production receipts for threads {missing}")
        source_hashes = {receipt["source_sha256"] for receipt in by_threads.values()}
        output_hashes = {receipt["canonical_output_sha256"] for receipt in by_threads.values()}
        modes = {receipt["mode"] for receipt in by_threads.values()}
        scratch_budgets = {receipt["scratch_budget_bytes"] for receipt in by_threads.values()}
        if len(source_hashes) > 1:
            blockers.append("candidate receipts do not bind the same source")
        if len(output_hashes) > 1:
            blockers.append("candidate receipts do not bind the same canonical output")
        if len(modes) > 1:
            blockers.append("candidate receipts mix execution modes")
        if len(scratch_budgets) > 1:
            blockers.append("candidate receipts mix scratch budgets")

        admissible = [
            receipt
            for receipt in by_threads.values()
            if receipt["peak_rss_bytes"] <= rss_limit_bytes
        ]
        if not admissible:
            blockers.append("no candidate satisfies the bound RSS limit")
        selected = min(
            admissible,
            key=lambda receipt: (receipt["wall_seconds"], receipt["threads"]),
            default=None,
        )
        baseline = by_threads.get(required[0])
        qualified = not blockers and selected is not None
        all_qualified &= qualified
        entry: dict[str, Any] = {
            "tier": tier,
            "rate": rate,
            "qualified": qualified,
            "blockers": blockers,
            "required_threads": list(required),
            "receipt_bindings": [
                {
                    "threads": threads,
                    "path": receipt["receipt_path"],
                    "sha256": receipt["receipt_sha256"],
                }
                for threads, receipt in sorted(by_threads.items())
            ],
        }
        if qualified and selected is not None:
            entry.update(
                {
                    "selected_threads": selected["threads"],
                    "selected_wall_seconds": selected["wall_seconds"],
                    "selected_peak_rss_bytes": selected["peak_rss_bytes"],
                    "scratch_budget_bytes": selected["scratch_budget_bytes"],
                    "mode": selected["mode"],
                    "source_sha256": selected["source_sha256"],
                    "canonical_output_sha256": selected["canonical_output_sha256"],
                    "speedup_vs_lowest_thread_candidate": (
                        baseline["wall_seconds"] / selected["wall_seconds"]
                        if baseline is not None
                        else None
                    ),
                }
            )
        entries[_entry_key(tier, rate)] = entry

    return {
        "schema": PROFILE_SCHEMA,
        "status": "qualified" if all_qualified else "partial",
        "generated_unix_ns": time.time_ns(),
        "expected_binary_sha256": expected_binary_sha256,
        "rss_limit_bytes": rss_limit_bytes,
        "required_threads": list(required),
        "entry_count": len(entries),
        "entries": entries,
    }


def verify_selection(
    profile: dict[str, Any],
    *,
    tier: str,
    rate: str,
    binary_sha256: str,
) -> dict[str, Any]:
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ContractError(f"profile schema must be {PROFILE_SCHEMA!r}")
    binary_sha256 = require_sha256(binary_sha256, "binary_sha256")
    if profile.get("expected_binary_sha256") != binary_sha256:
        raise ContractError("runtime binary SHA does not match the profile")
    rss_limit_bytes = profile.get("rss_limit_bytes")
    if (
        not isinstance(rss_limit_bytes, int)
        or isinstance(rss_limit_bytes, bool)
        or rss_limit_bytes <= 0
    ):
        raise ContractError("profile RSS limit must be a positive integer")
    required = tuple(profile.get("required_threads", []))
    if required != DEFAULT_THREADS:
        raise ContractError(f"production selector requires the complete {DEFAULT_THREADS} candidate set")
    key = _entry_key(
        require_nonempty_string(tier, "tier"),
        require_nonempty_string(rate, "rate"),
    )
    entry = profile.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        raise ContractError("no exact tier/rate profile entry; fallback is forbidden")
    if entry.get("qualified") is not True or entry.get("blockers"):
        raise ContractError("tier/rate profile entry is not qualified")
    bindings = entry.get("receipt_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(DEFAULT_THREADS):
        raise ContractError("profile does not bind all required candidate receipts")
    seen_threads: set[int] = set()
    candidates: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ContractError("receipt binding must be an object")
        path = Path(require_nonempty_string(binding.get("path"), "receipt path"))
        expected_receipt_sha = require_sha256(binding.get("sha256"), "receipt sha256")
        if not path.is_file() or sha256_file(path) != expected_receipt_sha:
            raise ContractError(f"bound receipt is missing or changed: {path}")
        normalized = validate_receipt(
            load_json(path),
            expected_binary_sha256=binary_sha256,
            allowed_threads=DEFAULT_THREADS,
        )
        validate_pipeline_binding(normalized, path)
        if normalized["tier"] != tier or normalized["rate"] != rate:
            raise ContractError("bound receipt tier/rate does not match the selected entry")
        if normalized["threads"] != binding.get("threads"):
            raise ContractError("bound receipt thread count does not match its profile binding")
        seen_threads.add(normalized["threads"])
        candidates.append(normalized)
    if seen_threads != set(DEFAULT_THREADS):
        raise ContractError("bound receipts do not cover exactly 8/12/16/20 threads")
    if len({candidate["source_sha256"] for candidate in candidates}) != 1:
        raise ContractError("bound receipts do not share one source")
    if len({candidate["canonical_output_sha256"] for candidate in candidates}) != 1:
        raise ContractError("bound receipts do not share one canonical output")
    if len({candidate["mode"] for candidate in candidates}) != 1:
        raise ContractError("bound receipts do not share one execution mode")
    if len({candidate["scratch_budget_bytes"] for candidate in candidates}) != 1:
        raise ContractError("bound receipts do not share one scratch budget")
    admissible = [
        candidate
        for candidate in candidates
        if candidate["peak_rss_bytes"] <= rss_limit_bytes
    ]
    if not admissible:
        raise ContractError("no bound candidate satisfies the profile RSS limit")
    winner = min(
        admissible,
        key=lambda candidate: (candidate["wall_seconds"], candidate["threads"]),
    )
    expected_entry = {
        "selected_threads": winner["threads"],
        "selected_wall_seconds": winner["wall_seconds"],
        "selected_peak_rss_bytes": winner["peak_rss_bytes"],
        "scratch_budget_bytes": winner["scratch_budget_bytes"],
        "mode": winner["mode"],
        "source_sha256": winner["source_sha256"],
        "canonical_output_sha256": winner["canonical_output_sha256"],
    }
    for field, expected in expected_entry.items():
        if entry.get(field) != expected:
            raise ContractError(f"profile entry {field} does not match bound receipts")
    return {
        "eligible": True,
        "tier": tier,
        "rate": rate,
        "threads": winner["threads"],
        "scratch_budget_bytes": winner["scratch_budget_bytes"],
        "mode": winner["mode"],
        "binary_sha256": binary_sha256,
        "source_sha256": winner["source_sha256"],
        "canonical_output_sha256": winner["canonical_output_sha256"],
        "profile_entry_key": key,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def parse_threads(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("threads must be comma-separated integers") from error
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--receipt", action="append", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--expected-binary-sha256", required=True)
    build.add_argument("--rss-limit-bytes", type=int, required=True)
    build.add_argument("--required-threads", type=parse_threads, default=DEFAULT_THREADS)
    build.add_argument("--allow-partial", action="store_true")
    select = commands.add_parser("select")
    select.add_argument("--profile", type=Path, required=True)
    select.add_argument("--tier", required=True)
    select.add_argument("--rate", required=True)
    select.add_argument("--binary-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            profile = build_profile(
                args.receipt,
                expected_binary_sha256=args.expected_binary_sha256,
                rss_limit_bytes=args.rss_limit_bytes,
                required_threads=args.required_threads,
            )
            atomic_write_json(args.output, profile)
            print(json.dumps(profile, sort_keys=True))
            return 0 if profile["status"] == "qualified" or args.allow_partial else 2
        selection = verify_selection(
            load_json(args.profile),
            tier=args.tier,
            rate=args.rate,
            binary_sha256=args.binary_sha256,
        )
        print(json.dumps(selection, sort_keys=True))
        return 0
    except ContractError as error:
        print(json.dumps({"eligible": False, "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
