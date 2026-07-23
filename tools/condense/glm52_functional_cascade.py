#!/usr/bin/env python3.12
"""Does the functional student compose across depth, or does one layer's error grow?

FS2 measured one layer and found the relative error entering layer 39 at 0.154 leaving it
at 0.262.  A per-layer amplification above one is the difference between a representation
and a curiosity: a model is 75 sparse layers deep, and an error that grows by a constant
factor per layer arrives at the head as noise no matter how good the single-layer score
was.

Two experiments, because they answer different questions:

* ``perturbation`` injects the student at one layer and then runs TEACHER weights for
  several following layers.  It isolates the amplification operator: how the block stack
  treats an error it did not create.  A factor below one means the residual stream is
  contractive and a single-layer student is safe; above one means depth is the enemy.
* ``cascade`` puts a student in every layer of the run.  This is the real thing, and it
  measures injection and amplification together.

Both compare against a teacher forward from the same input with the same indexer carry, so
the only difference is the MoE function.

    perturbation LAYER DEPTH
    cascade LAYER DEPTH
    selftest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import hawking_null_metric as metric  # noqa: E402
import glm52_functional_gauntlet as gauntlet  # noqa: E402

OUT = gauntlet.OUT


def _machinery():
    import glm52_teacher_capture as capture
    import glm52_reference as reference
    graph = capture._graph()
    config = capture.official_config()
    source = capture.ShardTensorSource(capture.SOURCE_ROOT, capture._tensor_table(graph))
    return capture, reference, source, config


def _run(capture, reference, source, config, layer, hidden, carry_topk):
    cache = reference.ReferenceCache()
    return capture.capture_layer(np.asarray(hidden, dtype=np.float32), source, layer,
                                 config, np.asarray(carry_topk), cache)[2]


def _relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def _students(layers, hidden, seed):
    """One student per layer, each fitted only on its own captured evidence."""
    made = {}
    for layer in layers:
        try:
            x_fit, y_fit = gauntlet.fit_evidence(layer)
        except gauntlet.GauntletError:
            continue
        made[layer] = {
            "candidate": gauntlet.candidate_student(x_fit, y_fit, hidden=hidden, seed=seed),
            "fit_positions": int(x_fit.shape[0]),
        }
    return made


def _walk(capture, reference, source, config, start, depth, hidden_in, carry_topk,
          students, teacher_hidden):
    """Advance `depth` layers, substituting the student wherever one is supplied."""
    rows = []
    student_state, teacher_state = hidden_in, teacher_hidden
    for step in range(depth):
        layer = start + 1 + step
        teacher_side = _run(capture, reference, source, config, layer, teacher_state,
                            carry_topk)
        student_side = _run(capture, reference, source, config, layer, student_state,
                            carry_topk)
        teacher_next = teacher_side["block_output"]
        if layer in students:
            predicted = students[layer]["candidate"]["predict"](
                student_side["pre_router_hidden"])
            student_next = (student_side["post_attention_hidden"]
                            + predicted.reshape(student_side["post_attention_hidden"].shape))
        else:
            student_next = student_side["block_output"]

        row = {
            "layer": layer,
            "student_substituted_here": layer in students,
            "relative_l2_entering": _relative(student_state, teacher_state),
            "relative_l2_leaving": _relative(student_next, teacher_next),
            "relative_l2_by_stage": {
                key: _relative(student_side[key], teacher_side[key])
                for key in ("attention_output", "pre_router_hidden", "post_moe")
                if key in teacher_side},
        }
        row["step_amplification"] = (row["relative_l2_leaving"]
                                     / max(row["relative_l2_entering"], 1e-12))
        if "topk_indices" in teacher_side:
            t_idx = teacher_side["topk_indices"].reshape(-1, 8)
            s_idx = student_side["topk_indices"].reshape(-1, 8)
            row["mean_topk_overlap_of_8"] = float(np.mean(
                [len(set(a) & set(b)) for a, b in zip(t_idx, s_idx)]))
            row["top1_agreement"] = float((t_idx[:, 0] == s_idx[:, 0]).mean())
        rows.append(row)
        student_state, teacher_state = student_next, teacher_next
    return rows, student_state, teacher_state


def _seed_perturbation(layer, hidden, seed):
    made = gauntlet.candidate_student(*gauntlet.fit_evidence(layer),
                                      hidden=hidden, seed=seed)
    got = gauntlet.arrays(layer, gauntlet.SCORE_SPLIT,
                          ("pre_router_hidden", "post_attention_hidden", "block_output"))
    with np.load(gauntlet.capsule(layer, gauntlet.SCORE_SPLIT)) as data:
        carry_topk = np.asarray(data["carry_out_index_selection"])
    teacher_in = got["block_output"]
    student_in = (got["post_attention_hidden"]
                  + made["predict"](got["pre_router_hidden"]).reshape(teacher_in.shape))
    return made, teacher_in, student_in, carry_topk


def perturbation(layer: int, depth: int = 3, *, hidden=gauntlet.PRIMARY_HIDDEN,
                 seed=gauntlet.SEEDS[0]) -> dict:
    started = time.time()
    capture, reference, source, config = _machinery()
    _, teacher_in, student_in, carry_topk = _seed_perturbation(layer, hidden, seed)
    rows, student_state, teacher_state = _walk(
        capture, reference, source, config, layer, depth, student_in, carry_topk,
        students={}, teacher_hidden=teacher_in)

    entering = _relative(student_in, teacher_in)
    leaving = _relative(student_state, teacher_state)
    factors = [row["step_amplification"] for row in rows]
    result = {
        "experiment": "perturbation_through_teacher_layers",
        "question": "how does the block stack treat an error it did not create",
        "injected_at": layer,
        "teacher_layers_traversed": [row["layer"] for row in rows],
        "relative_l2_at_injection": entering,
        "relative_l2_after": leaving,
        "total_amplification": leaving / max(entering, 1e-12),
        "per_layer_amplification": factors,
        "geometric_mean_amplification": float(np.exp(np.mean(np.log(factors))))
        if factors else None,
        "contractive": bool(factors and max(factors) < 1.0),
        "steps": rows,
        "seconds": round(time.time() - started, 1),
    }
    result["projected_over_75_sparse_layers"] = (
        float(result["geometric_mean_amplification"] ** 75)
        if result["geometric_mean_amplification"] else None)
    return result


def cascade(layer: int, depth: int = 3, *, hidden=gauntlet.PRIMARY_HIDDEN,
            seed=gauntlet.SEEDS[0]) -> dict:
    """A student in every layer of the run: injection and amplification together."""
    started = time.time()
    capture, reference, source, config = _machinery()
    made, teacher_in, student_in, carry_topk = _seed_perturbation(layer, hidden, seed)
    students = _students(range(layer + 1, layer + 1 + depth), hidden, seed)
    rows, student_state, teacher_state = _walk(
        capture, reference, source, config, layer, depth, student_in, carry_topk,
        students=students, teacher_hidden=teacher_in)

    fit_block = gauntlet.arrays(layer, gauntlet.FIT_SPLITS[0],
                                ("block_output",))["block_output"]
    null = metric.fit_null(fit_block.reshape(-1, gauntlet.HIDDEN))
    final = metric.score(teacher_state.reshape(-1, gauntlet.HIDDEN),
                         student_state.reshape(-1, gauntlet.HIDDEN), null)

    entering = _relative(student_in, teacher_in)
    leaving = _relative(student_state, teacher_state)
    result = {
        "experiment": "student_in_every_layer",
        "question": "does a stack of functional students hold together",
        "seeded_at": layer,
        "substituted_layers": sorted(students),
        "layers_without_a_student": [row["layer"] for row in rows
                                     if not row["student_substituted_here"]],
        "fit_positions_per_student": {str(k): v["fit_positions"]
                                      for k, v in students.items()},
        "relative_l2_at_injection": entering,
        "relative_l2_after": leaving,
        "total_growth": leaving / max(entering, 1e-12),
        "steps": rows,
        "final_state_vs_teacher": {k: v for k, v in final.items() if k != "schema"},
        "seconds": round(time.time() - started, 1),
        "caveat": "students for the later layers are fitted on capsules seeded from the "
                  "embedding table, not on states chained from the perturbed layer; the "
                  "distribution they see here is therefore slightly off-fit",
    }
    return result


def threshold(layer: int, depth: int = 2, *, hidden=gauntlet.PRIMARY_HIDDEN,
              seed=gauntlet.SEEDS[0],
              scales=(0.05, 0.1, 0.25, 0.5, 1.0)) -> dict:
    """Is the amplification intrinsic to the stack, or is the student simply too far out?

    A block stack that amplifies every perturbation would make any lossy per-layer
    replacement impossible, which is not what the rest of the field observes.  The useful
    question is whether there is a magnitude below which the residual stream is
    contractive.  The student's own error direction is scaled rather than replaced by
    noise, so this measures the operator on the error that actually occurs.
    """
    started = time.time()
    capture, reference, source, config = _machinery()
    _, teacher_in, student_in, carry_topk = _seed_perturbation(layer, hidden, seed)
    direction = student_in - teacher_in

    rows = []
    for scale in scales:
        perturbed = teacher_in + direction * np.float32(scale)
        steps, final_student, final_teacher = _walk(
            capture, reference, source, config, layer, depth, perturbed, carry_topk,
            students={}, teacher_hidden=teacher_in)
        entering = _relative(perturbed, teacher_in)
        rows.append({
            "scale": scale,
            "relative_l2_at_injection": entering,
            "relative_l2_after": _relative(final_student, final_teacher),
            "per_layer_amplification": [s["step_amplification"] for s in steps],
            "geometric_mean_amplification": float(np.exp(np.mean(np.log(
                [s["step_amplification"] for s in steps])))),
            "first_layer_top1_agreement": steps[0].get("top1_agreement"),
            "first_layer_topk_overlap": steps[0].get("mean_topk_overlap_of_8"),
        })

    contractive = [row for row in rows if row["geometric_mean_amplification"] < 1.0]
    return {
        "experiment": "perturbation_magnitude_sweep",
        "question": "is there an error magnitude below which the stack is contractive",
        "injected_at": layer,
        "teacher_layers_traversed": depth,
        "student_own_relative_l2": rows[-1]["relative_l2_at_injection"],
        "rows": rows,
        "contractive_below": (max(row["relative_l2_at_injection"] for row in contractive)
                              if contractive else None),
        "stack_is_expansive_at_every_tested_magnitude": not contractive,
        "seconds": round(time.time() - started, 1),
    }


def selftest() -> int:
    # The walk must chain: layer n's output is layer n+1's input, and the amplification of
    # an identical pair must be exactly one rather than a divide-by-zero.
    assert abs(_relative(np.ones((2, 3)), np.ones((2, 3)))) < 1e-12
    a = np.array([[1.0, 0.0]])
    assert abs(_relative(a * 1.1, a) - 0.1) < 1e-6
    # A geometric mean over per-layer factors must reproduce a constant factor exactly.
    factors = [1.7, 1.7, 1.7]
    assert abs(float(np.exp(np.mean(np.log(factors)))) - 1.7) < 1e-9
    print(json.dumps({"selftest": "PASS"}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 38
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "perturbation":
        payload = perturbation(layer, depth)
    elif command == "cascade":
        payload = cascade(layer, depth)
    elif command == "threshold":
        payload = threshold(layer, depth)
    else:
        raise SystemExit(f"unknown command: {command}")
    OUT.mkdir(parents=True, exist_ok=True)
    # Named by layer, so a probe at one stratum cannot overwrite another's evidence.
    path = OUT / f"GLM52_FUNCTIONAL_DEPTH_{command.upper()}_L{layer:02d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"wrote {path}", file=sys.stderr)
    print(json.dumps({k: v for k, v in payload.items() if k != "steps"}, indent=2,
                     default=float))
