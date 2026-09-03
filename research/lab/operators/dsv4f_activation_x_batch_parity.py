"""Compare two DSV4F activation-X captures for the batched-vs-old correctness gate.

The set of retained (layer, expert, row) keys and the row order must be
identical. Per-element drift is reported (max abs, cosine) per organ class.
Doctor6 must read the batched tree with no collector change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lab.operators.dsv4f_activation_capture import load_capture_result
from lab.operators.dsv4f_activation_x_source_verify import verify_doctor6


def _read_f32le(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    return raw


def _layer_meta_keys(run_dir: Path) -> list[tuple[int, int, int, tuple[int, ...]]]:
    keys = []
    meta_dir = run_dir / "layer_meta"
    for path in sorted(meta_dir.glob("L*.json")):
        doc = json.loads(path.read_text())
        layer = int(doc["layer"])
        for row in doc["tokens"]:
            pos = int(row["pos"])
            ids = tuple(int(x) for x in row["selected_expert_ids"])
            retained = bool(row.get("hidden_retained"))
            keys.append((layer, pos, int(retained), ids))
    return keys


def _retained_hidden_files(run_dir: Path) -> list[Path]:
    hidden = run_dir / "hidden"
    if not hidden.is_dir():
        return []
    return sorted(p for p in hidden.rglob("*.f32le") if p.is_file())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_runs(old_dir: Path, new_dir: Path) -> dict[str, Any]:
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    old_keys = _layer_meta_keys(old_dir)
    new_keys = _layer_meta_keys(new_dir)
    key_set_identical = old_keys == new_keys
    old_files = _retained_hidden_files(old_dir)
    new_files = _retained_hidden_files(new_dir)
    old_rel = [p.relative_to(old_dir).as_posix() for p in old_files]
    new_rel = [p.relative_to(new_dir).as_posix() for p in new_files]
    files_identical = old_rel == new_rel
    max_abs = 0.0
    min_cos = 1.0
    compared = 0
    worst = None
    for rel in old_rel:
        a = _read_f32le(old_dir / rel)
        b = _read_f32le(new_dir / rel)
        if a.shape != b.shape:
            raise SystemExit(f"shape mismatch {rel}: {a.shape} vs {b.shape}")
        absdiff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        cos = _cosine(a.astype(np.float64), b.astype(np.float64))
        max_abs = max(max_abs, absdiff)
        min_cos = min(min_cos, cos)
        compared += 1
        if worst is None or absdiff > worst["max_abs"]:
            worst = {"path": rel, "max_abs": absdiff, "cosine": cos}
    doctor6 = verify_doctor6(new_dir)
    old_cap = load_capture_result(old_dir)
    new_cap = load_capture_result(new_dir)
    report = {
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "key_set_and_order_identical": key_set_identical,
        "retained_file_set_identical": files_identical,
        "retained_rows_compared": compared,
        "router_input_h_post_ffn_norm": {
            "max_abs_diff": max_abs,
            "min_cosine": min_cos,
            "worst": worst,
        },
        "doctor6_batched": {
            "key_count": doctor6["doctor6_key_count"],
            "all_organs_finite": doctor6["all_organs_finite"],
            "sample_organ": doctor6["sample_organ"],
        },
        "old_tokens": (old_cap.get("source_run") or {}).get("tokens"),
        "new_tokens": (new_cap.get("source_run") or {}).get("tokens"),
        "old_execution_path": (old_cap.get("source_run") or {}).get("execution_path"),
        "new_execution_path": (new_cap.get("source_run") or {}).get("execution_path"),
        "pass": key_set_identical and files_identical and doctor6["all_organs_finite"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    args = parser.parse_args()
    report = compare_runs(args.old_dir, args.new_dir)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
