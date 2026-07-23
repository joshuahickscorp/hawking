"""Fit the MoE student on real teacher trajectories and score it on a disjoint partition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/scammermike/Downloads/hawking/tools/condense")
import glm52_moe_student as ms  # noqa: E402

BASE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/capsules_generation_b"
LAYER = 38
# Routed experts plus the shared expert, the weights the student replaces.
REPLACED = (256 * 2048 * 6144 * 3) + (2048 * 6144 * 3)

TRAIN = [BASE / "L38_L38.npz",
         BASE / "teacher_router/L38_L38.npz",
         BASE / "teacher_doctor/L38_L38.npz"]
SCORE = BASE / "teacher_score/L38_L38.npz"


def pairs(path: Path):
    data = np.load(path)
    x = np.asarray(data[f"layer_{LAYER:02d}/pre_router_hidden"], dtype=np.float32)
    y = np.asarray(data[f"layer_{LAYER:02d}/post_moe"], dtype=np.float32)
    return x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])


def main() -> int:
    xs, ys = zip(*[pairs(p) for p in TRAIN if p.exists()])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    xs_score, ys_score = pairs(SCORE)
    print(f"train samples {x.shape[0]}, score samples {xs_score.shape[0]}, "
          f"width {x.shape[1]}", flush=True)

    rows = []
    for hidden in (1024, 4096, 8192, 16384):
        fitted = ms.fit(x, y, hidden=hidden, seed=17, replaced_weights=REPLACED)
        predicted = ms.apply_student(fitted["blob"], xs_score)
        cosine = float(np.dot(predicted.ravel(), ys_score.ravel())
                       / max(np.linalg.norm(predicted) * np.linalg.norm(ys_score), 1e-12))
        relative = float(np.linalg.norm(predicted - ys_score)
                         / max(np.linalg.norm(ys_score), 1e-12))
        row = {k: v for k, v in fitted.items() if k != "blob"}
        row.update({"score_split_cosine": cosine, "score_split_relative_error": relative})
        rows.append(row)
        print(json.dumps({k: row[k] for k in
                          ("hidden", "bpw", "ridge", "in_sample_cosine",
                           "score_split_cosine", "score_split_relative_error",
                           "samples_per_stored_parameter", "fit_seconds")}), flush=True)
        Path(f"/private/tmp/claude-503/-Users-scammermike-Downloads-hawking/"
             f"4ad7f3da-1d29-4770-9680-d102475f345e/scratchpad/student_h{hidden}.bin"
             ).write_bytes(fitted["blob"])

    Path("/private/tmp/claude-503/-Users-scammermike-Downloads-hawking/"
         "4ad7f3da-1d29-4770-9680-d102475f345e/scratchpad/student_sweep.json"
         ).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
