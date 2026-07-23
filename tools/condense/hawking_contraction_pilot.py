#!/usr/bin/env python3.12
"""Parent-agnostic, contraction-first functional pilot.

GLM-5.2 taught the ordering this encodes: a functional student can pass the block
and replicate across layers and still be worthless, because the residual stream
amplifies the error it introduces. So the decisive question is asked early and
cheaply, before any rate ladder: does the target architecture have a regime a
per-layer functional student can survive in.

The protocol, in order:

    1. fit-split nulls
    2. full affine functional upper control
    3. tiny functional student
    4. weight-space control (a constant, standing in for the closed families)
    5. real block insertion
    6. next-layer propagation
    7. perturbation amplification at multiple magnitudes      <- the gate
    8. three-to-four-layer rollout
    9. router-margin and route-agreement propagation
    10. exact 0.75 / 0.50 / 0.333 artifacts, ONLY if the functional direction
        survives propagation

Promotion never follows from a local fit. It requires positive null-relative
block skill, rollout stability, cross-depth replication, exact physical
accounting, and a direct runtime path -- all of them.

The pilot is driven by a ``TeacherProvider``: anything that can return, for a
layer and split, the real trajectory tensors and can run one real block forward
from an arbitrary input. A synthetic provider with a tunable per-layer gain
exercises the whole harness with no model, and the selftest uses it to prove the
gate reports CONTRACTIVE when the operator contracts and EXPANSIVE when it does
not -- the discrimination GLM's residual stream failed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import hawking_null_metric as metric  # noqa: E402
import glm52_moe_student as student  # noqa: E402

# The magnitudes are fractions of the student's own error, so a stable regime below some
# fraction is detectable rather than assumed away.
MAGNITUDE_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
CONTRACTION_GATE = 1.0  # geometric-mean per-layer amplification below this is contractive
RIDGE_GRID = student.RIDGE_GRID


class TeacherProvider(Protocol):
    hidden_size: int

    def pairs(self, layer: int, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(pre_router_hidden, post_moe) for a layer and split, flattened to [N, hidden]."""

    def block_pieces(self, layer: int, split: str) -> dict:
        """post_attention_hidden, block_output, post_moe, and router evidence if MoE."""

    def forward_block(self, layer: int, hidden_in: np.ndarray) -> dict:
        """Run one real block from an arbitrary input state; return its trajectory."""

    def replaced_weights(self, layer: int) -> int:
        """Source logical weights the functional organ stands in for at this layer."""


# --------------------------------------------------------------------------- candidates


def _ridge(a: np.ndarray, b: np.ndarray, holdout: float = 0.2) -> tuple[np.ndarray, float]:
    cut = int(a.shape[0] * (1.0 - holdout))
    gram, cross = a[:cut].T @ a[:cut], a[:cut].T @ b[:cut]
    eye = np.eye(a.shape[1], dtype=np.float64)
    best, best_err = None, np.inf
    for ridge in RIDGE_GRID:
        weight = np.linalg.solve(gram + ridge * eye, cross)
        err = float(np.linalg.norm(a[cut:] @ weight - b[cut:]))
        if err < best_err:
            best, best_err = ridge, err
    weight = np.linalg.solve(a.T @ a + best * eye, a.T @ b)
    return weight, best


def affine_control(x_fit, y_fit, hidden_size):
    a = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1))], axis=1)
    weight, ridge = _ridge(a, y_fit)

    def predict(z):
        flat = z.reshape(-1, z.shape[-1])
        flat = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
        return (flat @ weight).reshape(*z.shape[:-1], hidden_size)

    return {"family": "affine_upper_control", "predict": predict,
            "bytes": weight.size * 2 + 64, "ridge": ridge, "stored": int(weight.size)}


def functional_student(x_fit, y_fit, hidden_size, *, hidden=1024, seed=17):
    phi = student.features(x_fit.astype(np.float32), seed, hidden).astype(np.float64)
    readout, ridge = _ridge(phi, y_fit)

    def predict(z, readout=readout, seed=seed, hidden=hidden):
        flat = z.reshape(-1, z.shape[-1]).astype(np.float32)
        return (student.features(flat, seed, hidden).astype(np.float64) @ readout
                ).reshape(*z.shape[:-1], hidden_size)

    return {"family": "random_feature_student", "predict": predict,
            "bytes": readout.size * 2 + 64, "ridge": ridge, "stored": int(readout.size),
            "seed": seed, "hidden": hidden,
            "expanded_bytes": hidden_size * hidden * 4 + readout.size * 2}


def weight_space_control(y_fit, null, hidden_size):
    """The closed weight-space families collapse to their best legal behaviour: a constant.

    On GLM every weight-space rung scored at or below the constant, so the honest stand-in
    for 'the best a weight blueprint did' is the fit-split mean.
    """
    mean = null["mean"]

    def predict(z):
        return np.broadcast_to(mean, (*z.shape[:-1], hidden_size))

    return {"family": "weight_space_constant", "predict": predict, "bytes": hidden_size * 2,
            "stored": hidden_size}


# --------------------------------------------------------------------------- steps


@dataclass
class PilotResult:
    parent: str
    steps: dict = field(default_factory=dict)
    verdict: str = "PENDING"
    promotion: dict = field(default_factory=dict)


def _score(provider, layer, candidate, x_score, y_score, null):
    prediction = candidate["predict"](x_score)
    scored = metric.score(y_score, prediction, null)
    return {"family": candidate["family"], "bytes": candidate["bytes"],
            "local_bpw": candidate["bytes"] * 8 / provider.replaced_weights(layer),
            **{k: v for k, v in scored.items() if k != "schema"}}


def steps_1_to_6(provider: TeacherProvider, layer: int, *, fit_splits, score_split,
                 hidden=1024, seed=17) -> dict:
    """Nulls, controls, student, block insertion and next-layer propagation."""
    xs, ys = zip(*[provider.pairs(layer, s) for s in fit_splits])
    x_fit = np.concatenate(xs).astype(np.float64)
    y_fit = np.concatenate(ys).astype(np.float64)
    x_score, y_score = (a.astype(np.float64) for a in provider.pairs(layer, score_split))
    null = metric.fit_null(y_fit)                                              # step 1
    hs = provider.hidden_size

    affine = affine_control(x_fit, y_fit, hs)                                  # step 2
    fitted = functional_student(x_fit, y_fit, hs, hidden=hidden, seed=seed)    # step 3
    weightless = weight_space_control(y_fit, null, hs)                         # step 4

    candidates = {name: _score(provider, layer, c, x_score, y_score, null)
                  for name, c in (("affine_upper_control", affine),
                                  ("functional_student", fitted),
                                  ("weight_space_control", weightless))}

    # step 5: real block insertion. block_output = post_attention + post_moe, so the block
    # score follows from substituting the student's post_moe.
    pieces = provider.block_pieces(layer, score_split)
    residual = pieces["post_attention_hidden"].reshape(-1, hs)
    teacher_block = pieces["block_output"].reshape(-1, hs)
    fit_block = provider.block_pieces(layer, fit_splits[0])["block_output"].reshape(-1, hs)
    block_null = metric.fit_null(fit_block)
    student_block = residual + fitted["predict"](x_score).reshape(-1, hs)
    block = {k: v for k, v in metric.score(teacher_block, student_block, block_null).items()
             if k != "schema"}
    no_moe = {k: v for k, v in metric.score(teacher_block, residual, block_null).items()
              if k != "schema"}

    # step 6: next-layer propagation through a real following block.
    student_next = provider.forward_block(layer + 1, student_block.reshape(pieces["block_output"].shape))
    teacher_next = provider.forward_block(layer + 1, pieces["block_output"])
    entering = float(np.linalg.norm(student_block - teacher_block)
                     / max(np.linalg.norm(teacher_block), 1e-12))
    leaving = float(np.linalg.norm(student_next["block_output"] - teacher_next["block_output"])
                    / max(np.linalg.norm(teacher_next["block_output"]), 1e-12))

    return {
        "layer": layer,
        "null_constant_raw_cosine": metric.constant_null_raw_cosine(y_score, null),
        "candidates": candidates,
        "block": block, "block_no_moe_control": no_moe,
        "block_gate_passes": bool(block["skill_lower"] > 0.0
                                  and block["skill"] > no_moe["skill"]),
        "propagation": {"entering_relative_l2": entering, "leaving_relative_l2": leaving,
                        "single_step_amplification": leaving / max(entering, 1e-12)},
        "_student": fitted, "_x_score": x_score, "_pieces": pieces,
    }


def amplification_gate(provider: TeacherProvider, layer: int, prep: dict, *,
                       depth=2) -> dict:
    """Step 7. Inject the student's error, scale it, carry it through teacher blocks."""
    hs = provider.hidden_size
    pieces = prep["_pieces"]
    teacher_in = pieces["block_output"]
    student_in = (pieces["post_attention_hidden"]
                  + prep["_student"]["predict"](prep["_x_score"]).reshape(teacher_in.shape))
    direction = student_in - teacher_in

    rows = []
    for fraction in MAGNITUDE_FRACTIONS:
        state_s = teacher_in + direction * fraction
        state_t = teacher_in
        factors, entering = [], float(np.linalg.norm(state_s - state_t)
                                      / max(np.linalg.norm(state_t), 1e-12))
        for step in range(depth):
            nxt = layer + 1 + step
            out_s = provider.forward_block(nxt, state_s)["block_output"]
            out_t = provider.forward_block(nxt, state_t)["block_output"]
            before = float(np.linalg.norm(state_s - state_t)
                           / max(np.linalg.norm(state_t), 1e-12))
            after = float(np.linalg.norm(out_s - out_t) / max(np.linalg.norm(out_t), 1e-12))
            factors.append(after / max(before, 1e-12))
            state_s, state_t = out_s, out_t
        rows.append({"fraction": fraction, "entering_relative_l2": entering,
                     "geometric_mean_amplification":
                         float(np.exp(np.mean(np.log(factors)))),
                     "per_layer": factors})

    gains = [r["geometric_mean_amplification"] for r in rows]
    contractive = [r for r in rows if r["geometric_mean_amplification"] < CONTRACTION_GATE]
    verdict = ("CONTRACTIVE" if all(g < CONTRACTION_GATE for g in gains)
               else "THRESHOLD_STABLE" if contractive
               else "EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE")
    return {"rows": rows, "verdict": verdict,
            "contractive_below_relative_l2":
                max((r["entering_relative_l2"] for r in contractive), default=None),
            "min_amplification": min(gains), "max_amplification": max(gains),
            "worse_at_smaller": gains[0] > gains[-1]}


def rollout(provider: TeacherProvider, layer: int, prep: dict, *, depth=4) -> dict:
    """Step 8/9. A student in every layer of the run, with route agreement tracked."""
    hs = provider.hidden_size
    pieces = prep["_pieces"]
    state_s = (pieces["post_attention_hidden"]
               + prep["_student"]["predict"](prep["_x_score"]).reshape(pieces["block_output"].shape))
    state_t = pieces["block_output"]
    fit_block = provider.block_pieces(layer, "teacher_fit")["block_output"].reshape(-1, hs) \
        if _has_split(provider, layer, "teacher_fit") else state_t.reshape(-1, hs)
    null = metric.fit_null(fit_block)

    steps = []
    for step in range(depth):
        nxt = layer + 1 + step
        side_s = provider.forward_block(nxt, state_s)
        side_t = provider.forward_block(nxt, state_t)
        row = {"layer": nxt,
               "relative_l2": float(np.linalg.norm(side_s["block_output"] - side_t["block_output"])
                                    / max(np.linalg.norm(side_t["block_output"]), 1e-12))}
        if "topk_indices" in side_t and "topk_indices" in side_s:
            k = side_t["topk_indices"].shape[-1]
            t_idx = side_t["topk_indices"].reshape(-1, k)
            s_idx = side_s["topk_indices"].reshape(-1, k)
            row["mean_topk_overlap"] = float(np.mean(
                [len(set(a) & set(b)) for a, b in zip(t_idx, s_idx)]))
            row["top1_agreement"] = float((t_idx[:, 0] == s_idx[:, 0]).mean())
            if "router_margin" in side_t:
                margin = side_t["router_margin"].reshape(-1)
                low = margin <= np.quantile(margin, 0.1)
                overlap = np.array([len(set(a) & set(b)) for a, b in zip(t_idx, s_idx)])
                row["overlap_on_low_margin"] = float(overlap[low].mean())
        steps.append(row)
        state_s, state_t = side_s["block_output"], side_t["block_output"]

    final = metric.score(state_t.reshape(-1, hs), state_s.reshape(-1, hs), null)
    return {"steps": steps, "depth": depth,
            "final_state_skill": final["skill"], "final_state_skill_lower": final["skill_lower"],
            "relative_l2_trajectory": [s["relative_l2"] for s in steps],
            "route_agreement_trajectory": [s.get("top1_agreement") for s in steps],
            "rollout_stable": bool(final["skill_lower"] > 0.0
                                   and steps[-1]["relative_l2"] < steps[0]["relative_l2"] * depth)}


def _has_split(provider, layer, split) -> bool:
    try:
        provider.pairs(layer, split)
        return True
    except Exception:  # noqa: BLE001
        return False


def run(provider: TeacherProvider, *, parent: str, strata, fit_splits, score_split,
        replication_splits=(), hidden=1024, seed=17) -> PilotResult:
    result = PilotResult(parent=parent)
    middle = strata[len(strata) // 2]

    prep = steps_1_to_6(provider, middle, fit_splits=fit_splits, score_split=score_split,
                        hidden=hidden, seed=seed)
    gate = amplification_gate(provider, middle, prep)
    result.steps["steps_1_to_6"] = {k: v for k, v in prep.items() if not k.startswith("_")}
    result.steps["amplification_gate"] = gate

    # The gate decides whether propagation-dependent work is worth doing.
    if gate["verdict"] == "EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE":
        result.verdict = "FUNCTIONAL_PARTIAL_ONLY_LIKELY"
        result.steps["rollout"] = rollout(provider, middle, prep)
        result.promotion = {"promoted": False,
                            "reason": "expansive at every magnitude; a per-layer student "
                                      "cannot compose. Rate ladder withheld."}
        return result

    # Contractive or threshold-stable: the functional direction may survive depth, so the
    # rollout and cross-depth replication decide promotion.
    result.steps["rollout"] = rollout(provider, middle, prep)
    strata_skill = {}
    for layer in strata:
        p = steps_1_to_6(provider, layer, fit_splits=fit_splits, score_split=score_split,
                         hidden=hidden, seed=seed)
        strata_skill[layer] = p["candidates"]["functional_student"]["skill"]
    result.steps["cross_depth_replication"] = strata_skill

    block_ok = prep["block_gate_passes"]
    rollout_ok = result.steps["rollout"]["rollout_stable"]
    replicated = all(s > 0 for s in strata_skill.values())
    result.verdict = ("FUNCTIONAL_CONTRACTIVE_CANDIDATE"
                      if block_ok and rollout_ok and replicated else "FUNCTIONAL_PARTIAL_ONLY_LIKELY")
    result.promotion = {
        "promoted": False,
        "requires_all_of": ["positive null-relative block skill", "rollout stability",
                            "cross-depth replication", "exact physical accounting",
                            "direct runtime path"],
        "block_skill_positive": block_ok,
        "rollout_stable": rollout_ok,
        "cross_depth_replicated": replicated,
        "exact_physical_accounting": "PENDING_RATE_LADDER",
        "direct_runtime_path": "PENDING_CODEC",
        "note": "local fit never promotes; the 0.75/0.50/0.333 rung artifacts are built "
                "only once this candidate holds through propagation.",
    }
    return result


# --------------------------------------------------------------------------- synthetic


@dataclass
class SyntheticProvider:
    """A stand-in teacher with a tunable per-layer gain, for exercising the harness.

    ``gain`` below 1 makes the block contractive, above 1 expansive. The MoE output is a
    fixed nonlinear function of the input so a student can actually fit it; the block adds
    a residual and applies a linear operator whose spectral radius is ``gain``.
    """
    hidden_size: int = 64
    gain: float = 0.8
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._w = rng.standard_normal((self.hidden_size, self.hidden_size)) / self.hidden_size
        q, _ = np.linalg.qr(rng.standard_normal((self.hidden_size, self.hidden_size)))
        self._op = q * self.gain  # perturbation operator with spectral radius = gain
        self._moe = rng.standard_normal((self.hidden_size, self.hidden_size)) / 8.0
        # A large fixed DC direction: the residual stream is mean-dominated, so the state
        # norm is roughly constant while a perturbation evolves under the operator. This is
        # what makes relative-L2 amplification track the gain rather than cancel to one, and
        # it is the same mean-dominance that broke raw cosine on the real parents.
        self._center = rng.standard_normal(self.hidden_size) * 6.0

    def _corpus(self, split, n=2048):
        rng = np.random.default_rng(abs(hash((split, self.seed))) % (2**32))
        return self._center + rng.standard_normal((n, self.hidden_size))

    def _moe_out(self, x):
        return np.tanh((x - self._center) @ self._moe) @ self._w * 4.0

    def pairs(self, layer, split):
        x = self._corpus(f"{layer}:{split}")
        return x, self._moe_out(x)

    def block_pieces(self, layer, split):
        x = self._corpus(f"{layer}:{split}")
        post_moe = self._moe_out(x)
        post_attn = self._center + (x - self._center) * 1.5
        return {"post_attention_hidden": post_attn, "block_output": post_attn + post_moe,
                "post_moe": post_moe}

    def forward_block(self, layer, hidden_in):
        flat = hidden_in.reshape(-1, self.hidden_size)
        # Deviations from the DC center evolve under the gain operator; the center is held,
        # so the output norm is stable and a small nonlinear term does not dominate.
        deviation = flat - self._center
        out = (self._center + deviation @ self._op.T
               + np.tanh(deviation @ self._moe) @ self._w * 0.1)
        return {"block_output": out.reshape(hidden_in.shape)}

    def replaced_weights(self, layer):
        return self.hidden_size * self.hidden_size * 4


def selftest() -> int:
    strata = [1, 2, 3]
    common = dict(strata=strata, fit_splits=("teacher_fit", "teacher_router"),
                  score_split="teacher_score", hidden=256, seed=17)

    # A contractive operator must be detected as such, and its student must not be dismissed
    # as expansive. This is the discrimination GLM's residual stream could not offer.
    contractive = run(SyntheticProvider(gain=0.6), parent="synthetic_contractive", **common)
    assert contractive.steps["amplification_gate"]["verdict"] == "CONTRACTIVE", \
        contractive.steps["amplification_gate"]["verdict"]
    assert contractive.verdict == "FUNCTIONAL_CONTRACTIVE_CANDIDATE", contractive.verdict

    # An expansive operator must trip the gate and withhold the rate ladder, exactly as GLM.
    expansive = run(SyntheticProvider(gain=1.6), parent="synthetic_expansive", **common)
    assert expansive.steps["amplification_gate"]["verdict"] \
        == "EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE", \
        expansive.steps["amplification_gate"]["verdict"]
    assert not expansive.promotion["promoted"]
    assert "rollout" in expansive.steps

    # A student must actually fit the synthetic MoE, or the harness is testing nothing.
    fs = contractive.steps["steps_1_to_6"]["candidates"]["functional_student"]
    assert fs["skill"] > 0.3, fs["skill"]
    # The block gate and no-MoE control must both be present and ordered.
    s16 = contractive.steps["steps_1_to_6"]
    assert s16["block"]["skill"] > s16["block_no_moe_control"]["skill"]

    # Promotion never fires from local fit alone: even the contractive candidate is not
    # promoted, because physical accounting and the runtime path are still pending.
    assert contractive.promotion["promoted"] is False
    assert contractive.promotion["exact_physical_accounting"] == "PENDING_RATE_LADDER"

    print(json.dumps({
        "selftest": "PASS",
        "contractive_gate": contractive.steps["amplification_gate"]["verdict"],
        "contractive_min_amp": round(contractive.steps["amplification_gate"]["min_amplification"], 3),
        "expansive_gate": expansive.steps["amplification_gate"]["verdict"],
        "expansive_min_amp": round(expansive.steps["amplification_gate"]["min_amplification"], 3),
        "student_skill": round(fs["skill"], 3),
    }))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    raise SystemExit(f"unknown command: {command}")
