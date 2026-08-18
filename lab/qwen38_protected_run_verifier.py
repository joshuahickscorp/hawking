#!/usr/bin/env python3
"""Read-only protected verification for a captured Qwen3.8 decode run.

The candidate capture is evidence, not authority.  This module hashes the
artifact manifest, runtime executable, and protected kernel source from
controller-supplied paths, then re-derives the complete-token wall from raw
warm-repetition samples.  It never launches the runtime and never writes
campaign or promotion state.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


VERIFICATION_SCHEMA = "hawking.protected.qwen38_run_verification.v1"
CAPTURE_SCHEMA = "hawking.ascent.qwen38_complete_token_wall.v1"
QWEN38_MODEL = "PocketAiHub/Qwen3.8-27B-Abliterated-MLX"
QWEN38_SAY_HI_PROMPT_IDS = (
    248045,
    846,
    198,
    44240,
    15131,
    13,
    248046,
    198,
    248045,
    74455,
    198,
)
QWEN38_SAY_HI_GREEDY_IDS = (
    248068,
    198,
    760,
    1156,
    4777,
    6587,
    728,
    310,
    1910,
    328,
    5834,
    1149,
    1061,
    369,
    264,
    1546,
    4145,
    11,
    2050,
    1622,
    13,
    353,
    3172,
    1066,
    1910,
    15131,
    303,
    264,
    11321,
    11,
    5629,
    1560,
)


class Qwen38VerificationError(ValueError):
    """The capture or its independently derived binding failed closed."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Qwen38VerificationError(f"{label} must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Qwen38VerificationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Qwen38VerificationError(f"{label} must be a non-negative integer")
    return value


def _int_sequence(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise Qwen38VerificationError(f"{label} must be a non-empty token-id list")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or item < 0:
            raise Qwen38VerificationError(
                f"{label}[{index}] must be a non-negative integer"
            )
        result.append(item)
    return tuple(result)


def _upper_median(values: Sequence[int], label: str) -> int:
    if not values:
        raise Qwen38VerificationError(f"{label} must not be empty")
    return sorted(values)[len(values) // 2]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen38VerificationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Qwen38VerificationError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Qwen38VerificationError(f"invalid {label} JSON {path}: {exc}") from exc
    return _mapping(value, label), raw


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise Qwen38VerificationError(f"cannot resolve {label} {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise Qwen38VerificationError(f"{label} is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise Qwen38VerificationError(f"runtime is not executable: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Qwen38VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _optional_claim(binding: Mapping[str, Any], name: str) -> str | None:
    value = binding.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise Qwen38VerificationError(f"candidate binding {name} must be 64 hex chars")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Qwen38VerificationError(
            f"candidate binding {name} must be lowercase hexadecimal"
        ) from exc
    if value != value.lower():
        raise Qwen38VerificationError(
            f"candidate binding {name} must be lowercase hexadecimal"
        )
    return value


def _check_claim(name: str, claimed: str | None, actual: str) -> None:
    if claimed is not None and not hmac.compare_digest(claimed, actual):
        raise Qwen38VerificationError(
            f"candidate binding {name} does not match independently hashed file"
        )


def _canonical_expected_ids(max_new_tokens: int) -> tuple[int, ...]:
    if max_new_tokens < 16 or max_new_tokens > len(QWEN38_SAY_HI_GREEDY_IDS):
        raise Qwen38VerificationError(
            "identity.max_new_tokens must be between 16 and 32 for the protected probe"
        )
    return QWEN38_SAY_HI_GREEDY_IDS[:max_new_tokens]


def _derive_wall(capture: Mapping[str, Any]) -> tuple[list[int], int]:
    warm_reps = capture.get("warm_reps")
    if not isinstance(warm_reps, list) or len(warm_reps) < 3:
        raise Qwen38VerificationError("warm_reps must contain at least 3 repetitions")

    identity = _mapping(capture.get("identity"), "identity")
    expected_ids = _canonical_expected_ids(
        _positive_int(identity.get("max_new_tokens"), "identity.max_new_tokens")
    )
    rep_medians: list[int] = []
    for rep_index, rep_value in enumerate(warm_reps):
        rep = _mapping(rep_value, f"warm_reps[{rep_index}]")
        summary = _mapping(rep.get("summary"), f"warm_reps[{rep_index}].summary")
        fallbacks = _nonnegative_int(
            summary.get("fallbacks"), f"warm_reps[{rep_index}].summary.fallbacks"
        )
        if fallbacks != 0:
            raise Qwen38VerificationError(
                f"warm_reps[{rep_index}] recorded {fallbacks} silent fallbacks"
            )
        rep_ids = _int_sequence(
            summary.get("new_token_ids"),
            f"warm_reps[{rep_index}].summary.new_token_ids",
        )
        if rep_ids != expected_ids:
            raise Qwen38VerificationError(
                f"warm_reps[{rep_index}] greedy ids differ from protected probe"
            )

        steady = _mapping(
            summary.get("steady_decode"),
            f"warm_reps[{rep_index}].summary.steady_decode",
        )
        wall = _mapping(
            steady.get("complete_wall_ns"),
            f"warm_reps[{rep_index}].summary.steady_decode.complete_wall_ns",
        )
        raw_values = wall.get("all")
        if not isinstance(raw_values, list) or not raw_values:
            raise Qwen38VerificationError(
                f"warm_reps[{rep_index}] has no raw complete-wall samples"
            )
        samples = [
            _positive_int(value, f"warm_reps[{rep_index}].wall[{sample_index}]")
            for sample_index, value in enumerate(raw_values)
        ]
        declared_n = _positive_int(
            summary.get("n_steady_decode_steps"),
            f"warm_reps[{rep_index}].summary.n_steady_decode_steps",
        )
        if declared_n != len(samples):
            raise Qwen38VerificationError(
                f"warm_reps[{rep_index}] sample count {len(samples)} != {declared_n}"
            )
        rep_medians.append(_upper_median(samples, f"warm_reps[{rep_index}] wall"))

    return rep_medians, _upper_median(rep_medians, "rep medians")


def verify_qwen38_capture(
    *,
    capture_path: Path,
    artifact_root: Path,
    runtime_executable: Path,
    kernel_source: Path,
    max_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Verify one captured run without executing it or trusting its hashes.

    ``artifact_root``, ``runtime_executable``, and ``kernel_source`` are
    protected-controller inputs, not paths selected from the candidate capture.
    If the candidate supplies hash claims, they are checked against independently
    read bytes; their absence does not prevent the verifier from deriving and
    reporting the binding.
    """

    capture_file = _regular_file(Path(capture_path), "capture")
    try:
        artifact_dir = Path(artifact_root).resolve(strict=True)
    except OSError as exc:
        raise Qwen38VerificationError(
            f"cannot resolve artifact root {artifact_root}: {exc}"
        ) from exc
    if not artifact_dir.is_dir():
        raise Qwen38VerificationError(f"artifact root is not a directory: {artifact_dir}")
    manifest_file = _regular_file(artifact_dir / "manifest.json", "artifact manifest")
    runtime_file = _regular_file(Path(runtime_executable), "runtime", executable=True)
    kernel_file = _regular_file(Path(kernel_source), "kernel source")

    capture, capture_raw = _load_json(capture_file, "capture")
    manifest, _ = _load_json(manifest_file, "artifact manifest")
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise Qwen38VerificationError(
            f"capture schema must be {CAPTURE_SCHEMA!r}, got {capture.get('schema')!r}"
        )
    if manifest.get("schema") != "hawking.ascent.qwen38_language_uniform_q4.v1":
        raise Qwen38VerificationError("artifact manifest is not uniform-q4-v1 Qwen3.8")

    identity = _mapping(capture.get("identity"), "identity")
    if identity.get("model") != QWEN38_MODEL:
        raise Qwen38VerificationError("identity.model is not the protected Qwen3.8 model")
    prompt_ids = _int_sequence(identity.get("prompt_ids"), "identity.prompt_ids")
    if prompt_ids != QWEN38_SAY_HI_PROMPT_IDS:
        raise Qwen38VerificationError("identity.prompt_ids differ from the protected probe")
    expected_ids = _canonical_expected_ids(
        _positive_int(identity.get("max_new_tokens"), "identity.max_new_tokens")
    )
    greedy_ids = _int_sequence(
        identity.get("greedy_new_token_ids"), "identity.greedy_new_token_ids"
    )
    if greedy_ids != expected_ids:
        raise Qwen38VerificationError(
            "identity.greedy_new_token_ids differ from the protected probe"
        )
    fallbacks = _nonnegative_int(identity.get("fallbacks"), "identity.fallbacks")
    if fallbacks != 0:
        raise Qwen38VerificationError(f"capture recorded {fallbacks} silent fallbacks")

    vehicle = _mapping(capture.get("vehicle"), "vehicle")
    claimed_root = vehicle.get("artifact_root")
    if not isinstance(claimed_root, str):
        raise Qwen38VerificationError("vehicle.artifact_root must be a path string")
    try:
        candidate_artifact_root = Path(claimed_root).resolve(strict=True)
    except OSError as exc:
        raise Qwen38VerificationError(
            f"cannot resolve candidate vehicle.artifact_root: {exc}"
        ) from exc
    if candidate_artifact_root != artifact_dir:
        raise Qwen38VerificationError(
            "candidate vehicle.artifact_root differs from protected artifact root"
        )

    artifact_sha = _sha256_file(manifest_file)
    runtime_sha = _sha256_file(runtime_file)
    kernel_sha = _sha256_file(kernel_file)
    binding_value = capture.get("binding", {})
    binding = _mapping(binding_value, "binding")
    _check_claim(
        "artifact_manifest_sha256",
        _optional_claim(binding, "artifact_manifest_sha256"),
        artifact_sha,
    )
    _check_claim(
        "runtime_executable_sha256",
        _optional_claim(binding, "runtime_executable_sha256"),
        runtime_sha,
    )
    _check_claim(
        "kernel_source_sha256",
        _optional_claim(binding, "kernel_source_sha256"),
        kernel_sha,
    )

    rep_medians, derived_wall = _derive_wall(capture)
    authority = _mapping(capture.get("authority"), "authority")
    declared_rep_medians = authority.get("rep_median_complete_wall_ns")
    if not isinstance(declared_rep_medians, list):
        raise Qwen38VerificationError(
            "authority.rep_median_complete_wall_ns must be a list"
        )
    declared_rep_medians_int = [
        _positive_int(value, f"authority.rep_median_complete_wall_ns[{index}]")
        for index, value in enumerate(declared_rep_medians)
    ]
    if declared_rep_medians_int != rep_medians:
        raise Qwen38VerificationError(
            "declared rep medians do not match raw complete-wall samples"
        )
    declared_wall = _positive_int(
        authority.get("headline_complete_wall_ns_per_token"),
        "authority.headline_complete_wall_ns_per_token",
    )
    if declared_wall != derived_wall:
        raise Qwen38VerificationError(
            f"declared headline wall {declared_wall} != derived {derived_wall}"
        )
    if max_wall_ns is not None:
        limit = _positive_int(max_wall_ns, "max_wall_ns")
        if derived_wall > limit:
            raise Qwen38VerificationError(
                f"derived wall {derived_wall} ns exceeds protected limit {limit} ns"
            )
    declared_tps = authority.get("headline_complete_tps")
    if not isinstance(declared_tps, (int, float)) or isinstance(declared_tps, bool):
        raise Qwen38VerificationError("authority.headline_complete_tps must be numeric")
    derived_tps = 1_000_000_000.0 / derived_wall
    if not math.isclose(float(declared_tps), derived_tps, rel_tol=1e-12, abs_tol=1e-12):
        raise Qwen38VerificationError(
            "authority.headline_complete_tps does not match the derived wall"
        )

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "PASS",
        "capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "protected_binding": {
            "artifact_root": str(artifact_dir),
            "artifact_manifest_path": str(manifest_file),
            "artifact_manifest_sha256": artifact_sha,
            "runtime_executable_path": str(runtime_file),
            "runtime_executable_sha256": runtime_sha,
            "kernel_source_path": str(kernel_file),
            "kernel_source_sha256": kernel_sha,
            "candidate_hashes_were_not_authority": True,
        },
        "measurement": {
            "model": QWEN38_MODEL,
            "prompt_ids": list(prompt_ids),
            "greedy_new_token_ids": list(greedy_ids),
            "fallbacks": 0,
            "warm_reps": len(rep_medians),
            "derived_rep_median_complete_wall_ns": rep_medians,
            "derived_headline_complete_wall_ns_per_token": derived_wall,
            "derived_complete_tps": derived_tps,
        },
        "claim_boundary": {
            "gpu_work_launched": False,
            "promotion_state_mutated": False,
            "runtime_executed": False,
            "wall_rederived_from_raw_capture": True,
            "capture_origin_attested": False,
            "note": (
                "This validates protected identities and internal capture consistency. "
                "Only a protected launcher can prove that this runtime produced the capture."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--runtime-executable", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--max-wall-ns", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = verify_qwen38_capture(
            capture_path=args.capture,
            artifact_root=args.artifact_root,
            runtime_executable=args.runtime_executable,
            kernel_source=args.kernel_source,
            max_wall_ns=args.max_wall_ns,
        )
    except Qwen38VerificationError as exc:
        print(f"qwen38 protected verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
