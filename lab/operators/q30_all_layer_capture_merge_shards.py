#!/usr/bin/env python3
"""Merge N all-layer Q30 capture shard directories into one single-process-shaped run.

Each shard is produced by:

    ascension_qwen30_broad_activation_all_layer_route_capture
        --probe-shard i --probe-shard-count N --output-dir <shard_i>

Hidden-position selection is global and deterministic over the full probe list
inside each capture process, so shards agree without communicating. This merge
is a pure union:

- probes concatenated in ORIGINAL global probe order (not shard order)
- hidden/ tree union (probe ids are unique)
- totals recomputed: total_tokens_executed, hidden_tokens_retained, layers_executed

Does not claim coherence, HCLI, TPS, or capability. Diagnostic capture only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


RESULT_SCHEMA = (
    "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_result.v1"
)
CAPTURE_PROTOCOL_REVISION = "all-layer-route-hidden-capture-stratified-subsample-v1"


def _die(message: str, code: int = 2) -> None:
    print(f"q30 all-layer capture merge refused: {message}", file=sys.stderr)
    raise SystemExit(code)


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _die(f"missing capture-result.json under {path.parent}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        _die(f"{path} is not a JSON object")
    if document.get("schema") != RESULT_SCHEMA:
        _die(
            f"{path} schema is {document.get('schema')!r}, expected {RESULT_SCHEMA!r}"
        )
    if document.get("capture_protocol_revision") != CAPTURE_PROTOCOL_REVISION:
        _die(
            f"{path} capture_protocol_revision is "
            f"{document.get('capture_protocol_revision')!r}, expected "
            f"{CAPTURE_PROTOCOL_REVISION!r}"
        )
    probes = document.get("probes")
    if not isinstance(probes, list):
        _die(f"{path} lacks probes array")
    return document


def _count_hidden_tokens(probes: list[dict[str, Any]]) -> int:
    retained = 0
    for probe in probes:
        steps = probe.get("steps") or []
        for step in steps:
            if step.get("hidden_retained_for_this_token") is True:
                retained += 1
    return retained


def _count_tokens(probes: list[dict[str, Any]]) -> int:
    total = 0
    for probe in probes:
        steps = probe.get("steps") or []
        total += len(steps)
        declared = probe.get("source_one_user_native_prompt_token_count")
        if declared is not None and declared != len(steps):
            _die(
                f"probe {probe.get('probe_id')!r} token count mismatch: "
                f"declared {declared}, steps {len(steps)}"
            )
    return total


def _probe_order_from_input(input_path: Path | None, probes: list[dict[str, Any]]) -> list[str]:
    """Prefer the sealed capture input's probe order; else first-seen order."""
    if input_path is not None and input_path.is_file():
        with input_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        rows = document.get("probes") or []
        order = []
        for row in rows:
            probe_id = row.get("probe_id")
            if not probe_id:
                _die(f"capture input {input_path} has a probe without probe_id")
            order.append(probe_id)
        return order
    # Fall back to order of first appearance across shards (stable if shards
    # were launched with the same input).
    seen: list[str] = []
    for probe in probes:
        probe_id = probe.get("probe_id")
        if probe_id and probe_id not in seen:
            seen.append(probe_id)
    return seen


def _copy_hidden_tree(src: Path, dst: Path) -> int:
    """Copy hidden/ from src into dst. Returns number of files copied."""
    hidden = src / "hidden"
    if not hidden.is_dir():
        return 0
    copied = 0
    for root, _dirs, files in os.walk(hidden):
        root_path = Path(root)
        rel = root_path.relative_to(hidden)
        target_dir = dst / "hidden" / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            source_file = root_path / name
            target_file = target_dir / name
            if target_file.exists():
                # Probe ids are unique; a collision means overlapping shards.
                _die(
                    f"hidden path collision while merging: {target_file.relative_to(dst)}"
                )
            shutil.copy2(source_file, target_file)
            copied += 1
    return copied


def merge_shards(
    shard_dirs: list[Path],
    output_dir: Path,
    *,
    input_json: Path | None = None,
) -> dict[str, Any]:
    if not shard_dirs:
        _die("at least one --shard-dir is required")
    if output_dir.exists():
        _die(f"refusing to reuse or overwrite output directory {output_dir}")
    parent = output_dir.parent
    if parent is None or not parent.is_dir():
        _die(f"output parent must already exist: {parent}")

    results: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            _die(f"shard directory does not exist: {shard_dir}")
        results.append(_load_result(shard_dir / "capture-result.json"))

    # Cross-shard consistency: same input, same runtime seals, same budget.
    def _field(document: dict[str, Any], *path: str) -> Any:
        cursor: Any = document
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor

    reference = results[0]
    for index, document in enumerate(results[1:], start=1):
        for path in (
            ("input", "sha256"),
            ("input", "schema"),
            ("runtime_binding", "manifest_seal_sha256"),
            ("runtime_binding", "source_revision"),
            ("runtime_binding", "layers"),
            ("runtime_binding", "hidden"),
            ("bounded_storage", "max_hidden_tokens_per_layer"),
            ("bounded_storage", "strategy"),
            ("bounded_storage", "hidden_position_selection"),
        ):
            left = _field(reference, *path)
            right = _field(document, *path)
            if left != right:
                _die(
                    f"shard {index} disagrees with shard 0 on {'.'.join(path)}: "
                    f"{right!r} vs {left!r}"
                )

    # Build probe map; refuse duplicate probe_ids across shards.
    probe_by_id: dict[str, dict[str, Any]] = {}
    for document in results:
        for probe in document.get("probes") or []:
            probe_id = probe.get("probe_id")
            if not probe_id:
                _die("a shard probe lacks probe_id")
            if probe_id in probe_by_id:
                _die(f"probe_id {probe_id!r} appears in more than one shard")
            probe_by_id[probe_id] = probe

    # Original global order from the sealed input when available.
    if input_json is None:
        input_path_value = _field(reference, "input", "path")
        if isinstance(input_path_value, str) and input_path_value:
            candidate = Path(input_path_value)
            if candidate.is_file():
                input_json = candidate
    order = _probe_order_from_input(input_json, list(probe_by_id.values()))
    missing = [probe_id for probe_id in order if probe_id not in probe_by_id]
    if missing:
        _die(
            f"merged shards are missing probes present in input order: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    extras = [probe_id for probe_id in probe_by_id if probe_id not in order]
    if extras:
        # Append extras after known order so nothing is dropped, but flag them.
        order = order + extras

    merged_probes = [probe_by_id[probe_id] for probe_id in order if probe_id in probe_by_id]
    total_tokens = _count_tokens(merged_probes)
    hidden_tokens_retained = _count_hidden_tokens(merged_probes)
    layers_executed = int(_field(reference, "runtime_binding", "layers") or 48)
    hidden_width = int(_field(reference, "runtime_binding", "hidden") or 2048)
    max_hidden = int(
        _field(reference, "bounded_storage", "max_hidden_tokens_per_layer") or 0
    )

    # Prefer the global selection figure recorded by any shard (they must agree).
    global_selected = _field(
        reference, "bounded_storage", "global_hidden_tokens_selected_per_layer"
    )
    if global_selected is None:
        global_selected = hidden_tokens_retained
    if int(global_selected) != hidden_tokens_retained:
        _die(
            f"merged hidden token count {hidden_tokens_retained} does not match "
            f"global_hidden_tokens_selected_per_layer {global_selected}; "
            f"shards likely used independent position selection"
        )

    retained_hidden_budget_bytes = (
        hidden_tokens_retained * layers_executed * hidden_width * 4
    )
    naive_hidden_bytes = total_tokens * layers_executed * hidden_width * 4

    output_dir.mkdir(parents=False)
    hidden_files = 0
    for shard_dir in shard_dirs:
        hidden_files += _copy_hidden_tree(shard_dir, output_dir)

    # Sum written bytes from shards (each shard reports only its own writes).
    retained_hidden_bytes_written = 0
    for document in results:
        written = _field(document, "bounded_storage", "retained_hidden_bytes_written")
        if isinstance(written, int):
            retained_hidden_bytes_written += written

    expected_written = retained_hidden_budget_bytes
    if retained_hidden_bytes_written != expected_written:
        _die(
            f"retained_hidden_bytes_written sum {retained_hidden_bytes_written} "
            f"!= expected {expected_written} "
            f"({hidden_tokens_retained} tokens × {layers_executed} layers × "
            f"{hidden_width} × 4)"
        )

    claim_boundary = reference.get("claim_boundary")
    logit_provenance = reference.get("logit_provenance")
    runtime_binding = dict(reference.get("runtime_binding") or {})
    # Merged run is not a single executable; keep seals, drop per-process exe hash
    # only if shards disagree.
    exe_hashes = {
        _field(document, "runtime_binding", "runtime_executable_sha256")
        for document in results
    }
    if len(exe_hashes) == 1:
        runtime_binding["runtime_executable_sha256"] = next(iter(exe_hashes))
    else:
        runtime_binding["runtime_executable_sha256"] = sorted(
            h for h in exe_hashes if h is not None
        )

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": reference.get("status"),
        "capture_protocol_revision": CAPTURE_PROTOCOL_REVISION,
        "input": reference.get("input"),
        "runtime_binding": runtime_binding,
        "bounded_storage": {
            "strategy": "stratified_token_subsample_raw_hiddens_plus_full_route_membership",
            "why": _field(reference, "bounded_storage", "why"),
            "max_hidden_tokens_per_layer": max_hidden,
            "hidden_tokens_retained_per_layer": hidden_tokens_retained,
            "global_hidden_tokens_selected_per_layer": int(global_selected),
            "layers": layers_executed,
            "hidden_width": hidden_width,
            "total_tokens_executed": total_tokens,
            "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
            "retained_hidden_budget_bytes": retained_hidden_budget_bytes,
            "retained_hidden_bytes_written": retained_hidden_bytes_written,
            "full_route_membership_for_every_token_every_layer": True,
            "hidden_position_selection": "global_stratified_over_full_probe_list",
            "rejected_alternatives": _field(
                reference, "bounded_storage", "rejected_alternatives"
            ),
        },
        "probe_shard": {
            "index": 0,
            "count": 1,
            "ownership": "probe_index % count == index",
            "global_probe_count": len(merged_probes),
            "shard_probe_count": len(merged_probes),
            "merged_from_shard_count": len(shard_dirs),
            "merged_from": [str(path) for path in shard_dirs],
            "hidden_files_copied": hidden_files,
        },
        "capture_summary": {
            "probe_count": len(merged_probes),
            "total_tokens": total_tokens,
            "layers_executed": layers_executed,
            "broad_activation_diversity": True,
            "all_layer_activation_capture": True,
            "hidden_tokens_retained": hidden_tokens_retained,
        },
        "probes": merged_probes,
        "logit_provenance": logit_provenance,
        "claim_boundary": claim_boundary,
    }

    result_path = output_dir / "capture-result.json"
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with result_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    summary = {
        "status": "MERGED",
        "schema": RESULT_SCHEMA,
        "output_dir": str(output_dir),
        "shard_count": len(shard_dirs),
        "capture_summary": result["capture_summary"],
        "bounded_storage": {
            "hidden_tokens_retained_per_layer": hidden_tokens_retained,
            "total_tokens_executed": total_tokens,
            "retained_hidden_bytes_written": retained_hidden_bytes_written,
            "hidden_files_copied": hidden_files,
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge N all-layer Q30 capture shard output directories into one "
            "single-process-shaped capture run."
        )
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        dest="shard_dirs",
        required=True,
        help="Absolute path to one shard output directory (repeat N times)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Absolute path for the merged capture run (must not exist)",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help=(
            "Optional sealed capture input JSON used to recover original probe "
            "order. Defaults to the path recorded in shard capture-result.json."
        ),
    )
    args = parser.parse_args(argv)

    shard_dirs = [Path(value) for value in args.shard_dirs]
    for path in shard_dirs:
        if not path.is_absolute():
            _die(f"--shard-dir must be absolute: {path}")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        _die(f"--output-dir must be absolute: {output_dir}")
    input_json = Path(args.input_json) if args.input_json else None
    if input_json is not None and not input_json.is_absolute():
        _die(f"--input-json must be absolute: {input_json}")

    merge_shards(shard_dirs, output_dir, input_json=input_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
