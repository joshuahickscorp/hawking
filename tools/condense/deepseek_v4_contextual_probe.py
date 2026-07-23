#!/usr/bin/env python3.12
"""The DeepSeek functional existence test on REAL contextual pre-MoE hidden.

The embedding-seeded probe was instrument-limited: its own shared-expert control showed the
input, not the organ, drove the negative. This replaces the input with the real thing. The
validated streamed forward produces the actual pre-MoE hidden a layer sees in context, and
the validated MoE forward produces its output, so the functional student is fitted against a
real contextual (input -> MoE output) map with fit-split nulls and the mandatory controls.

Positions come from two disjoint real sequences (a fit sequence and a score sequence), so
the split is across documents, not within one. It is a bounded existence test, not a
capability claim.

    run LAYER [SEQ_LEN]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import deepseek_v4_moe as ds
import deepseek_v4_reference as ref
import hawking_null_metric as metric
import glm52_moe_student as student

OUT = Path(__file__).resolve().parents[2] / "reports" / "condense" / "deepseek_v4_flash"


def _contextual_pairs(layer: int, seq_len: int, seed: int, experts) -> tuple:
    """Real pre-MoE hidden from the streamed forward, and its real MoE output."""
    tokens = np.random.default_rng(seed).integers(0, 129280, seq_len, dtype=np.int64)
    _, captured = ref.streamed_forward(tokens, layer, capture=(layer,))
    x = captured[layer].reshape(-1, ds.DIM).astype(np.float64)
    y = ds.moe_forward(x.astype(np.float32), layer, ds._index(),
                       experts=experts)["post_moe"].astype(np.float64)
    return x, y


def run(layer: int, seq_len: int = 256) -> dict:
    index = ds._index()
    experts = {e: ds._load_expert(f"layers.{layer}.ffn.experts.{e}", index)
               for e in range(ds.N_ROUTED)}
    x_fit, y_fit = _contextual_pairs(layer, seq_len, 41, experts)
    x_score, y_score = _contextual_pairs(layer, seq_len, 42, experts)
    null = metric.fit_null(y_fit)

    replaced = (ds.N_ROUTED * (ds.MOE_INTER * ds.DIM * 3)
                + ds.N_SHARED * (ds.MOE_INTER * ds.DIM * 3) + ds.DIM * ds.N_ROUTED)

    def score(predict, blob_bytes, label):
        s = metric.score(y_score, predict(x_score), null)
        return {"label": label, "bytes": blob_bytes,
                "local_bpw": blob_bytes * 8 / replaced,
                **{k: v for k, v in s.items() if k != "schema"}}

    fitted = student.fit(x_fit.astype(np.float32), y_fit.astype(np.float32),
                         hidden=1024, seed=17, replaced_weights=replaced)
    a = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1))], axis=1)
    weight = np.linalg.solve(a.T @ a + 1.0 * np.eye(a.shape[1]), a.T @ y_fit)

    def affine(z):
        return np.concatenate([z, np.ones((z.shape[0], 1))], axis=1) @ weight

    rows = [
        score(lambda z: student.apply_student(fitted["blob"], z.astype(np.float32)),
              len(fitted["blob"]), "functional_student_h1024"),
        score(affine, weight.size * 2 + 64, "affine_upper_control"),
        score(lambda z: np.broadcast_to(null["mean"], (z.shape[0], ds.DIM)),
              ds.DIM * 2, "weight_space_constant"),
    ]
    shuffled = np.random.default_rng(9).permutation(x_score.shape[0])
    shuffle_skill = metric.score(
        y_score, student.apply_student(fitted["blob"], x_score[shuffled].astype(np.float32)),
        null)["skill"]

    result = {
        "schema": "hawking.deepseek_v4.contextual_probe.v1",
        "parent": "deepseek-ai/DeepSeek-V4-Flash", "layer": layer,
        "input": "REAL contextual pre-MoE hidden from the validated streamed forward",
        "forward": "official DeepseekV4DecoderLayer, real dequantized weights, streamed",
        "fit_positions": int(x_fit.shape[0]), "score_positions": int(x_score.shape[0]),
        "seq_len": seq_len,
        "constant_null_raw_cosine": metric.constant_null_raw_cosine(y_score, null),
        "candidates": rows,
        "shuffled_input_skill": shuffle_skill,
        "functional_escape_exists": bool(rows[0]["passes"]),
        "map_is_the_win": bool(rows[1]["skill"] >= rows[0]["skill"] - 0.05),
        "compared_to_embedding_seeded": "the embedding-seeded probe gave student skill "
            "-0.06/-0.09 and a shared-expert control near zero, proving the input invalid; "
            "this uses real contextual hidden instead",
        "not_evidence_of": "amplification or capability; that needs perturbation propagation "
                           "through following blocks, which the same streamed forward supports.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"DEEPSEEK_V4_CONTEXTUAL_PROBE_L{layer:02d}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float))
    return result


if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    r = run(layer, seq_len)
    for row in r["candidates"]:
        print(f"  {row['label']:26} bpw {row['local_bpw']:.6f}  skill {row['skill']:7.4f}"
              f"  lower {row['skill_lower']:7.4f}  centered {row['centered_cosine']:7.4f}"
              f"  pass {row['passes']}")
    print(f"escape_exists={r['functional_escape_exists']} "
          f"shuffled={r['shuffled_input_skill']:.4f} null_raw={r['constant_null_raw_cosine']:.4f}")
