#!/usr/bin/env python3
"""G053: oracle-first bounds for the three unjudged DeltaNet families.

S020 §16 named six families. Two have curves and both came back
MEASURED_NEGATIVE. Three more are blocked on a fitted program. Oracle-first
is a standing Gravity law:

    IF AN UNFAIR ORACLE CANNOT MAKE THE ECONOMICS WORK,
    DO NOT BUILD THE REAL VERSION.

This module fits the strongest cheap unfair oracle for each remaining family
on the same teacher-corpus holdout the multi-step judge used (L38
post_attn_norm, prompt_ids code:14 and code:15) and reports the bound.
An oracle may be unfair — perfect routing, perfect per-block codes, fitted
on the tokens it is scored on — because its job is an UPPER BOUND. If that
bound cannot reconstruct the transition at the 0.01 bar, or cannot clear
the 1.813 ms residual at MEASURED per-stream rates, the family is
ORACLE_NEGATIVE and nobody writes a runtime.

    python3 tools/future/deltanet_program_oracles.py --build
    python3 -m pytest tools/future/test_deltanet_program_oracles.py -q

evidence_class STATIC_ONLY. No GPU lease. Does not touch crates/.
CPU / numpy only. Capability is UNMEASURED for every family.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import causal_budget_71 as cb
from tools.future import deltanet_multistep as dnm
from tools.future import deltanet_state_function as dsf
from tools.future import deltanet_state_machine_economics as sme
from tools.future._common import REPO, write_receipt


RECEIPT = "DELTANET_PROGRAM_ORACLES.json"
SCHEMA = "hawking.future.deltanet_program_oracles.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_program_oracles.py"
EVIDENCE_CLASS = "STATIC_ONLY"

PREDECESSOR_MULTISTEP = "receipts/future/DELTANET_MULTISTEP.json"
PREDECESSOR_STATE = "receipts/future/DELTANET_STATE_FUNCTION.json"
PREDECESSOR_ECON = "receipts/future/DELTANET_STATE_MACHINE_ECONOMICS.json"
CALIBRATION_REL = "receipts/future/ECONOMICS_CALIBRATION.json"

ORACLE_NEGATIVE = "ORACLE_NEGATIVE"
ORACLE_PERMITS_INVESTIGATION = "ORACLE_PERMITS_INVESTIGATION"
BLOCKED = "BLOCKED"
VERDICTS = (ORACLE_NEGATIVE, ORACLE_PERMITS_INVESTIGATION, BLOCKED)

REQUIRED_FAMILIES: tuple[str, ...] = (
    "generated_coefficients",
    "learned_recurrence",
    "conditional_recurrence",
)

JUDGE_HOLD_PROMPT_IDS: tuple[str, ...] = ("code:14", "code:15")
JUDGE_LAYER = dnm.DN_PROBE_LAYER  # 38
JUDGE_HEADS = dnm.PROBE_HEADS  # 8
JUDGE_DIM = dnm.PROBE_DIM  # 128
JUDGE_KIND = "held_out_prompt_teacher_corpus_L38_post_attn_norm"
MIN_HOLD_TOKENS = 256
MAX_HOLD_SEQUENCES = 2
MAX_TRAIN_SEQUENCES = 4

TRANSITION_BAR = 0.01  # same relative-L2 bar the multi-step judge uses
ENERGY_BAR_99 = 0.99
GENERATOR_RANKS: tuple[int, ...] = (64, 128, 256)
NAMED_GENERATOR_RANK = 256
LEARNED_DSTATES: tuple[int, ...] = (8, 16, 32, 64, 128)
NAMED_DSTATE = 16
CONDITIONAL_KS: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
HORIZONS: tuple[int, ...] = (1, 16, 64, 256)
Q4_GROUP = 64
RNG_SEED = dnm.RNG_SEED
F32 = 4
F16 = 2
HEADER_PER_LAYER = 40

BOUND_KIND = "OPPORTUNITY_BOUND_ON_PERFECT_SUCCESS"
UNMEASURED = "UNMEASURED"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Milliseconds are CITED: catalog / state bytes billed at the MEASURED "
    "per-stream rates from ECONOMICS_CALIBRATION (weight_codes 0.547282 ms/GB, "
    "broadcast_aux 0.000, activation 2.906132), never the organ average. "
    "Recurrent state S is resident ACTIVATION. This is an "
    "OPPORTUNITY_BOUND_ON_PERFECT_SUCCESS, not a measured speedup. Capability "
    "is UNMEASURED for every family. Coefficients are the same seeded STATIC "
    "map of real residual-stream X the multi-step judge used; that map is not "
    "W_qkvz and the trained q/k/v trajectory is UNMEASURED. Held-out is by "
    "prompt_id (code:14, code:15). A train figure cannot be reported as "
    "held-out. A missing corpus raises rather than synthesising X (NNS-001). "
    "An absent family is a refuse, not a pass."
)


class OracleRefuse(ValueError):
    """The oracle lane refused rather than guessing."""


class CorpusUnavailable(OracleRefuse):
    """Real residual-stream rows are not readable; synthesising X is NNS-001."""

    def __init__(self, paths: Sequence[str | Path], *, detail: str = "") -> None:
        self.missing_paths = [str(p) for p in paths]
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: teacher-corpus payload is not readable; "
            f"missing {self.missing_paths}{extra}; "
            "refusing to synthesise X (NNS-001)"
        )


class HeldOutRefuse(OracleRefuse):
    """A train figure cannot be reported as held-out."""


class FamilyAbsentRefuse(OracleRefuse):
    """An absent family reads as a judged one. Raise, do not skip."""


class FamilyInputMissing(OracleRefuse):
    """A family with a missing input raises rather than being skipped."""


def _py(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(dnm.relative_l2(a, b))


def stream_rates() -> dict[str, float]:
    """MEASURED per-stream rates. Never the organ average."""
    return dict(cb.STREAM_MS_PER_GB)


def cited_ms(n_bytes: int, stream: str, *, rates: Mapping[str, float] | None = None) -> float:
    rates = dict(rates or stream_rates())
    if stream not in rates:
        raise OracleRefuse(f"REFUSED: unknown stream {stream!r}; refusing to bill at the organ average")
    return float(n_bytes) / 1e9 * float(rates[stream])


def residual_gap_ms() -> float:
    """The gap a NEW lever has to close. Cited from the economics receipt, not frozen."""
    return float(sme.price()["gap_to_71_residual_after_everything_on_record_ms"])


def _as_int_breakdown(row: Mapping[str, Any], *keys: str) -> dict[str, int]:
    out = {k: int(row.get(k) or 0) for k in keys}
    out["total"] = int(row["total"]) if "total" in row else int(sum(out.values()))
    return out


def load_named_candidate(cid: str) -> dict[str, Any]:
    path = REPO / PREDECESSOR_STATE
    if not path.is_file():
        raise OracleRefuse(f"REFUSED: {PREDECESSOR_STATE} is not readable")
    doc = json.loads(path.read_text())
    for row in doc.get("candidates") or []:
        if str(row.get("id")) == cid:
            return dict(row)
    raise OracleRefuse(f"REFUSED: candidate {cid!r} missing from {PREDECESSOR_STATE}")


# ---------------------------------------------------------------------------
# Corpus. Same payload and same holdout as DELTANET_MULTISTEP.
# ---------------------------------------------------------------------------


def payload_candidates() -> tuple[Path, ...]:
    return tuple(dnm.PAYLOAD_CANDIDATES)


def resolve_payload_dir(payload: Path | None = None) -> Path:
    if payload is not None:
        root = Path(payload)
        if (root / "rows.jsonl").is_file() and (
            root / f"L{JUDGE_LAYER:02d}_x.f32"
        ).is_file():
            return root
        raise CorpusUnavailable([root], detail=f"L{JUDGE_LAYER:02d}_x.f32 + rows.jsonl required")
    found = dnm.resolve_payload_dir()
    if found is not None:
        return Path(found)
    raise CorpusUnavailable(payload_candidates())


def assert_held_out_sequences(
    sequences: Sequence[Mapping[str, Any]],
    *,
    train_ids: Sequence[str] | None = None,
    allowed_ids: Sequence[str] | None = None,
) -> None:
    """A train-set figure cannot be reported as held-out. Loud exception."""
    if not sequences:
        raise HeldOutRefuse("REFUSED: no sequences to score as held-out")
    train = set(str(x) for x in (train_ids or []))
    allowed = set(str(x) for x in allowed_ids) if allowed_ids is not None else None
    for seq in sequences:
        split = str(seq.get("split") or "")
        pid = str(seq.get("prompt_id") or "")
        if split == "train":
            raise HeldOutRefuse(
                f"REFUSED: train figure cannot be reported as held-out "
                f"(prompt_id={pid!r} split={split!r})"
            )
        if pid in train:
            raise HeldOutRefuse(
                f"REFUSED: prompt {pid} is a train id reported as held-out"
            )
        if allowed is not None and pid not in allowed:
            raise HeldOutRefuse(
                f"REFUSED: prompt {pid} is not in the hold set {sorted(allowed)}"
            )
        if seq.get("synthetic") is True:
            raise CorpusUnavailable(
                ["synthetic-row"],
                detail=f"synthetic row in hold set prompt={pid} (NNS-001)",
            )


def assert_judge_holdout(sequences: Sequence[Mapping[str, Any]]) -> None:
    """The scored holdout must be the same two prompts the judge used."""
    assert_held_out_sequences(
        sequences, allowed_ids=JUDGE_HOLD_PROMPT_IDS
    )
    got = tuple(str(s["prompt_id"]) for s in sequences)
    if set(got) != set(JUDGE_HOLD_PROMPT_IDS):
        raise HeldOutRefuse(
            "REFUSED: scored prompt_ids "
            f"{list(got)} are not the judge holdout {list(JUDGE_HOLD_PROMPT_IDS)}"
        )
    for seq, want_pid in zip(sequences, JUDGE_HOLD_PROMPT_IDS):
        # Order is "longest first", which for this corpus is code:14 then code:15.
        if str(seq["prompt_id"]) != want_pid:
            # Permutation is acceptable; identity of the set was checked above.
            break
    for seq in sequences:
        if int(seq.get("layer", -1)) != JUDGE_LAYER:
            raise HeldOutRefuse(
                f"REFUSED: scored layer {seq.get('layer')} is not judge layer {JUDGE_LAYER}"
            )


def load_hidden_sequences(
    *,
    split: str,
    layer: int = JUDGE_LAYER,
    min_tokens: int = MIN_HOLD_TOKENS,
    max_sequences: int = MAX_HOLD_SEQUENCES,
    payload: Path | None = None,
    prompt_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Real post_attn_norm rows. Train and hold are disjoint by prompt_id."""
    if split not in {"hold", "train"}:
        raise OracleRefuse(f"REFUSED: split must be hold or train, got {split!r}")
    root = resolve_payload_dir(payload)
    split_doc = dnm._corpus_split(root)
    hold_ids = set(str(x) for x in (split_doc.get("hold_prompt_ids") or []))
    train_ids = set(str(x) for x in (split_doc.get("train_prompt_ids") or []))
    if not hold_ids or not train_ids:
        raise CorpusUnavailable(
            [root / "CAPTURE.json"],
            detail="hold_prompt_ids / train_prompt_ids missing from corpus split",
        )
    overlap = hold_ids & train_ids
    if overlap:
        raise HeldOutRefuse(
            f"REFUSED: hold/train prompt_id leak {sorted(overlap)[:8]}"
        )
    want_ids = set(str(x) for x in prompt_ids) if prompt_ids is not None else None
    if split == "hold":
        if want_ids is None:
            want_ids = set(JUDGE_HOLD_PROMPT_IDS)
        leak = want_ids & train_ids
        if leak:
            raise HeldOutRefuse(
                f"REFUSED: requested hold ids {sorted(leak)} sit on the train side"
            )
    else:
        if want_ids is None:
            want_ids = set(train_ids)
        leak = want_ids & hold_ids
        if leak:
            raise HeldOutRefuse(
                f"REFUSED: requested train ids {sorted(leak)} sit on the hold side"
            )

    rows_path = root / "rows.jsonl"
    x_path = root / f"L{int(layer):02d}_x.f32"
    if not rows_path.is_file() or not x_path.is_file():
        raise CorpusUnavailable([rows_path, x_path])

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    with rows_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if int(rec.get("layer", -1)) != int(layer):
                continue
            pid = str(rec["prompt_id"])
            if pid not in want_ids:
                continue
            declared = str(rec.get("split") or "")
            if split == "hold" and (pid in train_ids or declared == "train"):
                continue
            if split == "train" and (pid in hold_ids or declared == "hold"):
                continue
            if rec.get("synthetic") is True:
                raise CorpusUnavailable(
                    [rows_path],
                    detail=f"synthetic row {rec.get('row_id')} in {split} set (NNS-001)",
                )
            by_prompt.setdefault(pid, []).append(rec)

    n_rows = x_path.stat().st_size // (dsf.HIDDEN * F32)
    mm = np.memmap(x_path, dtype="<f4", mode="r", shape=(n_rows, dsf.HIDDEN))
    ranked = sorted(by_prompt.items(), key=lambda kv: -len(kv[1]))
    sequences: list[dict[str, Any]] = []
    for pid, recs in ranked:
        recs = sorted(recs, key=lambda r: int(r["token_position"]))
        if len(recs) < int(min_tokens):
            continue
        if split == "hold" and any(str(r.get("split")) == "train" for r in recs):
            raise HeldOutRefuse(
                f"REFUSED: prompt {pid} is in the hold id set but a row is split=train"
            )
        idx = [int(r["x_row_index"]) for r in recs]
        x = np.asarray(mm[idx], dtype=np.float32)
        sequences.append(
            {
                "prompt_id": pid,
                "layer": int(layer),
                "split": split,
                "n_tokens": int(x.shape[0]),
                "capability_domain": recs[0].get("capability_domain"),
                "x": x,
                "x_path": str(x_path),
                "held_out_unit": "prompt_id",
                "synthetic": False,
            }
        )
        if len(sequences) >= int(max_sequences):
            break
    if not sequences:
        raise CorpusUnavailable(
            [x_path],
            detail=f"no {split} prompt at L{layer} has >= {min_tokens} tokens",
        )
    if split == "hold":
        assert_judge_holdout(sequences)
    return sequences


def load_hold_sequences(payload: Path | None = None) -> list[dict[str, Any]]:
    return load_hidden_sequences(split="hold", payload=payload)


def load_train_sequences(payload: Path | None = None) -> list[dict[str, Any]]:
    return load_hidden_sequences(
        split="train",
        payload=payload,
        max_sequences=MAX_TRAIN_SEQUENCES,
    )


# ---------------------------------------------------------------------------
# Coefficients and trajectories. Same map as the judge. Not W_qkvz.
# ---------------------------------------------------------------------------


def coefficients_of(seq: Mapping[str, Any]) -> dict[str, np.ndarray]:
    x = np.asarray(seq["x"], dtype=np.float32)
    return dnm.coefficients_from_hidden(
        x, n_heads=JUDGE_HEADS, dim=JUDGE_DIM, seed=RNG_SEED
    )


def _stack_coeffs(bundle: Mapping[str, np.ndarray]) -> np.ndarray:
    t = int(bundle["q"].shape[0])
    parts = [
        np.asarray(bundle["q"], dtype=np.float32).reshape(t, -1),
        np.asarray(bundle["k"], dtype=np.float32).reshape(t, -1),
        np.asarray(bundle["v"], dtype=np.float32).reshape(t, -1),
        np.asarray(bundle["z"], dtype=np.float32).reshape(t, -1),
        np.asarray(bundle["decay"], dtype=np.float32).reshape(t, -1),
        np.asarray(bundle["beta"], dtype=np.float32).reshape(t, -1),
    ]
    return np.concatenate(parts, axis=1)


def _unstacK_coeffs(
    y: np.ndarray, *, n_heads: int = JUDGE_HEADS, dim: int = JUDGE_DIM
) -> dict[str, np.ndarray]:
    t = int(y.shape[0])
    hd = int(n_heads) * int(dim)
    q, k, v, z, rest = np.split(y, [hd, 2 * hd, 3 * hd, 4 * hd], axis=1)
    decay, beta = np.split(rest, [int(n_heads)], axis=1)
    return {
        "q": q.reshape(t, n_heads, dim).astype(np.float32, copy=False),
        "k": k.reshape(t, n_heads, dim).astype(np.float32, copy=False),
        "v": v.reshape(t, n_heads, dim).astype(np.float32, copy=False),
        "z": z.reshape(t, n_heads, dim).astype(np.float32, copy=False),
        "decay": decay.astype(np.float32, copy=False),
        "beta": beta.astype(np.float32, copy=False),
    }


def _concat_xy(
    sequences: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[str, np.ndarray]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    lengths: list[int] = []
    bundles: list[dict[str, np.ndarray]] = []
    for seq in sequences:
        bundle = coefficients_of(seq)
        x = np.asarray(seq["x"], dtype=np.float32)
        xs.append(x)
        ys.append(_stack_coeffs(bundle))
        lengths.append(int(x.shape[0]))
        bundles.append(bundle)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        lengths,
        bundles,
    )


def _pca_basis(x: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, components (rank, dim), singular_values)."""
    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0)
    xc = x64 - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    r = min(int(rank), int(vt.shape[0]))
    return mean.astype(np.float64), vt[:r].astype(np.float64), s[:r].astype(np.float64)


def _energy_ranks(singular: np.ndarray) -> dict[str, Any]:
    s = np.asarray(singular, dtype=np.float64)
    if s.size == 0 or float(np.square(s).sum()) <= 0.0:
        return {"n": 0, "rank_at_0.99": 0, "rank_at_store_bar": 0, "energy": {}}
    energy = np.cumsum(np.square(s)) / np.square(s).sum()
    def _rank(bar: float) -> int:
        return int(np.searchsorted(energy, bar) + 1)
    want = (8, 16, 32, 64, 128, 256)
    return {
        "n": int(s.size),
        "rank_at_0.99": _rank(ENERGY_BAR_99),
        "rank_at_store_bar": _rank(1.0 - TRANSITION_BAR * TRANSITION_BAR),
        "energy": {
            str(r): float(energy[r - 1])
            for r in want
            if r <= energy.size
        },
    }


def roll_state(
    bundle: Mapping[str, np.ndarray],
    *,
    n_heads: int = JUDGE_HEADS,
    dim: int = JUDGE_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """S and h along one prompt, from S=0. Returns (T,H,D,D), (T,H,D)."""
    t = int(bundle["q"].shape[0])
    s = np.zeros((n_heads, dim, dim), dtype=np.float32)
    states = np.empty((t, n_heads, dim, dim), dtype=np.float32)
    hs = np.empty((t, n_heads, dim), dtype=np.float32)
    for i in range(t):
        c = dnm._coeffs_at(bundle, i)
        s, h = dnm.reference_step(s, c)
        states[i] = s
        hs[i] = h
    return states, hs


def transition_errors(
    true_bundle: Mapping[str, np.ndarray],
    hat_bundle: Mapping[str, np.ndarray],
    *,
    n_heads: int = JUDGE_HEADS,
    dim: int = JUDGE_DIM,
    horizons: Sequence[int] = HORIZONS,
) -> dict[str, Any]:
    """One-step transition reconstruction AND a rolled trajectory.

    One-step applies hat coefficients to the TRUE previous S. That is
    reconstruction of the TRANSITION, not of an integrated drift.
    """
    t = int(true_bundle["q"].shape[0])
    s_true = np.zeros((n_heads, dim, dim), dtype=np.float32)
    s_roll = np.zeros((n_heads, dim, dim), dtype=np.float32)
    one_step: list[float] = []
    one_step_h: list[float] = []
    rolled: dict[int, dict[str, float]] = {}
    want = {int(h) for h in horizons}
    for i in range(t):
        c_true = dnm._coeffs_at(true_bundle, i)
        c_hat = dnm._coeffs_at(hat_bundle, i)
        s_next, h_true = dnm.reference_step(s_true, c_true)
        s_hat, h_hat = dnm.reference_step(s_true, c_hat)
        one_step.append(relative_l2(s_hat, s_next))
        one_step_h.append(relative_l2(h_hat, h_true))
        s_true = s_next
        s_roll, h_roll = dnm.reference_step(s_roll, c_hat)
        step = i + 1
        if step in want or i == t - 1:
            rolled[step] = {
                "state_relative_l2": relative_l2(s_roll, s_true),
                "output_h_relative_l2": relative_l2(h_roll, h_true),
            }
    return {
        "n_tokens": t,
        "one_step_state_relative_l2_mean": float(np.mean(one_step)),
        "one_step_state_relative_l2_max": float(np.max(one_step)),
        "one_step_h_relative_l2_mean": float(np.mean(one_step_h)),
        "one_step_h_relative_l2_max": float(np.max(one_step_h)),
        "rolled_by_horizon": {str(h): rolled[h] for h in sorted(rolled)},
        "clears_transition_bar": float(np.max(one_step)) <= TRANSITION_BAR,
        "bar": TRANSITION_BAR,
    }


# ---------------------------------------------------------------------------
# Billing. Per-stream rates. S is activation.
# ---------------------------------------------------------------------------


def opportunity_bound(
    *,
    removed_weight_codes: int,
    removed_activation: int,
    added_weight_codes: int,
    added_activation: int,
    residual: float | None = None,
) -> dict[str, Any]:
    rates = stream_rates()
    if rates["weight_codes"] == rates["activation"]:
        raise OracleRefuse(
            "REFUSED: weight_codes and activation billed at the same rate; "
            "that is the organ-average error this lane exists to prevent"
        )
    rem_w = cited_ms(removed_weight_codes, "weight_codes", rates=rates)
    rem_a = cited_ms(removed_activation, "activation", rates=rates)
    add_w = cited_ms(added_weight_codes, "weight_codes", rates=rates)
    add_a = cited_ms(added_activation, "activation", rates=rates)
    # broadcast_aux is measured at 0; an oracle router billed there is free
    # in cited ms, which is part of the unfairness.
    rem = rem_w + rem_a
    add = add_w + add_a
    net = rem - add
    gap = float(residual if residual is not None else residual_gap_ms())
    return {
        "kind": BOUND_KIND,
        "not_a_speedup": True,
        "ms_are_cited_not_measured": True,
        "capability": UNMEASURED,
        "bytes": {
            "removed": {
                "weight_codes": int(removed_weight_codes),
                "activation": int(removed_activation),
                "total": int(removed_weight_codes) + int(removed_activation),
            },
            "added": {
                "weight_codes": int(added_weight_codes),
                "activation": int(added_activation),
                "total": int(added_weight_codes) + int(added_activation),
            },
            "net_removed": (int(removed_weight_codes) + int(removed_activation))
            - (int(added_weight_codes) + int(added_activation)),
        },
        "cited_ms_removed": {
            "weight_codes": round(rem_w, 4),
            "activation": round(rem_a, 4),
            "total": round(rem, 4),
        },
        "cited_ms_added": {
            "weight_codes": round(add_w, 4),
            "activation": round(add_a, 4),
            "total": round(add, 4),
        },
        "opportunity_bound_ms": round(net, 4),
        "residual_gap_ms_cited": round(gap, 4),
        "clears_residual_gap": bool(net >= gap),
        "rates_used": dict(rates),
        "rates_source": "ECONOMICS_CALIBRATION via causal_budget_71.STREAM_MS_PER_GB",
        "state_billed_as": "activation",
        "why_state_is_activation": (
            "Recurrent state S is resident activation, read and written every "
            "token. Billing it at the weight_codes rate is a whole-organ error."
        ),
    }


def generator_byte_model(rank: int) -> dict[str, Any]:
    """Shared T1/T2 plus per-layer f16 diagonals, billed at model scope.

    Rank a multiple of 64 uses the recorded HQ30UQ4 packing; otherwise f16.
    At rank 256 this reconciles to the generated_transition_coefficients
    candidate (4,548,560 added, 2,139,096,960 removed).
    """
    r = int(rank)
    if r <= 0:
        raise OracleRefuse(f"REFUSED: generator rank {r} is not positive")
    if r % Q4_GROUP == 0:
        t1 = int(dsf.q4_stored(r, dsf.HIDDEN))
        t2 = int(dsf.q4_stored(dsf.QKVZ_ROWS, r))
        packing = "hq30uq4_uniform_q4_group64"
    else:
        t1 = int(r * dsf.HIDDEN * F16)
        t2 = int(dsf.QKVZ_ROWS * r * F16)
        packing = "f16"
    diag = int(dsf.N_DN_LAYERS * dsf.QKVZ_ROWS * F16)
    embed = int(dsf.N_DN_LAYERS * r * F32)
    meta = int(dsf.N_DN_LAYERS * HEADER_PER_LAYER)
    added = dsf.added(
        generator=t1 + t2, embeddings=embed, residuals=diag, metadata=meta
    )
    removed = dsf.removed(catalog_weights=int(dsf.QKVZ_ACTIVE_TARGET))
    return {
        "rank": r,
        "packing": packing,
        "t1_bytes": t1,
        "t2_bytes": t2,
        "bytes_added": added,
        "bytes_removed": removed,
        "net_bytes": int(removed["total"]) - int(added["total"]),
        "scope": "model (48 DN layers; T1/T2 shared; diagonals and embeddings per-layer)",
    }


def learned_byte_model(d_state: int) -> dict[str, Any]:
    """Scale the recorded learned_recurrence candidate to the d_state the oracle needed."""
    named = load_named_candidate("learned_recurrence")
    d = int(d_state)
    if d <= 0:
        raise OracleRefuse(f"REFUSED: d_state {d} is not positive")
    scale = d / float(NAMED_DSTATE)
    named_added = named["bytes_added"]
    named_inproj = int(named_added["generator"]) - int(
        dsf.N_DN_LAYERS * dsf.VALUE_HEADS * NAMED_DSTATE * F32
    )
    in_proj = int(round(named_inproj * scale))
    a_diag = int(dsf.N_DN_LAYERS * dsf.VALUE_HEADS * d * F32)
    state = int(dsf.N_DN_LAYERS * dsf.VALUE_HEADS * d * F32)
    added = dsf.added(
        generator=in_proj + a_diag,
        metadata=int(named_added.get("metadata") or 1920),
        state=state,
    )
    removed = _as_int_breakdown(
        named["bytes_removed"], "catalog_weights", "state", "other"
    )
    return {
        "d_state": d,
        "named_d_state": NAMED_DSTATE,
        "in_proj_bytes": in_proj,
        "a_diag_bytes": a_diag,
        "bytes_added": added,
        "bytes_removed": removed,
        "net_bytes": int(removed["total"]) - int(added["total"]),
        "source": PREDECESSOR_STATE,
        "id": "learned_recurrence",
        "note": (
            "in_proj (weight_codes) scales linearly with d_state from the "
            "recorded d_state=16 candidate. A_diag and the recurrent state "
            "are billed as added activation (state) and added weight_codes "
            "(A_diag lives in generator)."
        ),
    }


def conditional_rule_bytes(n_regimes: int) -> dict[str, Any]:
    """One stored (q,k,v,decay,beta) prototype per regime, billed for ALL regimes.

    The oracle router is free in cited ms (broadcast_aux rate is 0.000) and
    cannot exist at runtime. Rules are weight_codes.
    """
    k = int(n_regimes)
    if k <= 0:
        raise OracleRefuse(f"REFUSED: n_regimes {k} is not positive")
    geo = dsf.geometry()
    per_layer = (
        int(geo["q_rows"]) + int(geo["k_rows"]) + int(geo["v_rows"])
        + int(geo["value_heads"]) * 2
    ) * F32
    one_rule = per_layer * int(dsf.N_DN_LAYERS)
    added = dsf.added(generator=one_rule * k, metadata=int(dsf.N_DN_LAYERS * HEADER_PER_LAYER))
    ba = int(dsf.q4_stored(int(geo["ba_rows"]), dsf.HIDDEN, n_layers=dsf.N_DN_LAYERS))
    removed = dsf.removed(catalog_weights=int(dsf.QKVZ_ACTIVE_TARGET) + ba)
    return {
        "n_regimes": k,
        "bytes_per_rule": one_rule,
        "router_bytes": 0,
        "router_stream": "broadcast_aux",
        "router_cited_ms": 0.0,
        "bytes_added": added,
        "bytes_removed": removed,
        "net_bytes": int(removed["total"]) - int(added["total"]),
        "note": (
            "Bills ALL K rules. The oracle router is assigned for free "
            "(broadcast_aux bills at 0.000 ms/GB) and cannot exist at runtime. "
            "Removed qkvz + ba (the maps that emit the coefficients the "
            "prototypes replace)."
        ),
    }


def bound_from_added_removed(
    added: Mapping[str, int],
    removed: Mapping[str, int],
    *,
    added_state_is_activation: bool = True,
    residual: float | None = None,
) -> dict[str, Any]:
    """Split added/removed into streams. State is activation; the rest is weight_codes."""
    rem_state = int(removed.get("state") or 0)
    rem_w = int(removed.get("total") or 0) - rem_state
    add_state = int(added.get("state") or 0) if added_state_is_activation else 0
    add_w = int(added.get("total") or 0) - add_state
    return opportunity_bound(
        removed_weight_codes=rem_w,
        removed_activation=rem_state,
        added_weight_codes=add_w,
        added_activation=add_state,
        residual=residual,
    )


def decide_verdict(
    *,
    reconstruction_clears: bool,
    bound: Mapping[str, Any],
) -> str:
    """Investigation is licensed only if the unfair oracle reconstructs AND pays."""
    if reconstruction_clears and bool(bound.get("clears_residual_gap")):
        return ORACLE_PERMITS_INVESTIGATION
    return ORACLE_NEGATIVE


# ---------------------------------------------------------------------------
# Family oracles.
# ---------------------------------------------------------------------------


def _split_hat(
    y_hat: np.ndarray, lengths: Sequence[int]
) -> list[dict[str, np.ndarray]]:
    out: list[dict[str, np.ndarray]] = []
    start = 0
    for n in lengths:
        out.append(_unstacK_coeffs(y_hat[start : start + int(n)]))
        start += int(n)
    return out


def _worst(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(max(float(r[key]) for r in rows))


def oracle_generated_coefficients(
    train: Sequence[Mapping[str, Any]],
    hold: Sequence[Mapping[str, Any]],
    *,
    ranks: Sequence[int] = GENERATOR_RANKS,
    residual: float | None = None,
) -> dict[str, Any]:
    """Best cheap generator with perfect per-block codes.

    Shared PCA of X is the generator. Per-head / per-output codes are a
    least-squares map on the scored tokens (unfair). Reconstruction of the
    TRANSITION is one-step gated-delta under generated vs true coefficients
    on the true previous S.
    """
    assert_judge_holdout(hold)
    assert_held_out_sequences(hold, train_ids=[s["prompt_id"] for s in train])
    x_hold, y_hold, hold_len, hold_bundles = _concat_xy(hold)
    x_train, _y_train, _train_len, _train_bundles = _concat_xy(train)
    gap = float(residual if residual is not None else residual_gap_ms())
    n_hold = int(x_hold.shape[0])
    interp_rank = int(math.ceil(n_hold / Q4_GROUP) * Q4_GROUP)
    rank_list = sorted(set(int(r) for r in ranks) | {NAMED_GENERATOR_RANK, interp_rank})

    def _fit(tag: str, x_fit: np.ndarray) -> list[dict[str, Any]]:
        max_r = min(max(rank_list), int(min(x_fit.shape)))
        mean, basis, _svals = _pca_basis(x_fit, max_r)
        rows: list[dict[str, Any]] = []
        y_c = y_hold.astype(np.float64) - y_hold.astype(np.float64).mean(axis=0)
        x_c = x_hold.astype(np.float64) - mean
        for r in rank_list:
            rr = min(int(r), int(basis.shape[0]))
            p = basis[:rr].T  # (dim, r)
            z = x_c @ p
            sol, *_ = np.linalg.lstsq(z, y_c, rcond=None)
            yhat_c = z @ sol
            yhat = yhat_c + y_hold.astype(np.float64).mean(axis=0)
            coeff_rel = relative_l2(yhat, y_hold)
            hats = _split_hat(yhat.astype(np.float32), hold_len)
            trans = [
                transition_errors(true_b, hat_b)
                for true_b, hat_b in zip(hold_bundles, hats)
            ]
            model = generator_byte_model(rr if rr == int(r) else int(r))
            bound = bound_from_added_removed(
                model["bytes_added"], model["bytes_removed"], residual=gap
            )
            one_step_max = _worst(trans, "one_step_state_relative_l2_max")
            rows.append(
                {
                    "fit_on": tag,
                    "rank": int(r),
                    "rank_used": rr,
                    "coefficient_relative_l2": coeff_rel,
                    "one_step_state_relative_l2_max": one_step_max,
                    "one_step_state_relative_l2_mean": float(
                        np.mean([t["one_step_state_relative_l2_mean"] for t in trans])
                    ),
                    "clears_transition_bar": bool(one_step_max <= TRANSITION_BAR)
                    and bool(coeff_rel <= TRANSITION_BAR),
                    "per_prompt": [
                        {
                            "prompt_id": hold[i]["prompt_id"],
                            "n_tokens": int(hold_len[i]),
                            **{k: trans[i][k] for k in (
                                "one_step_state_relative_l2_max",
                                "one_step_state_relative_l2_mean",
                                "clears_transition_bar",
                                "rolled_by_horizon",
                            )},
                        }
                        for i in range(len(hold))
                    ],
                    "byte_model": {
                        k: model[k]
                        for k in (
                            "rank",
                            "packing",
                            "t1_bytes",
                            "t2_bytes",
                            "bytes_added",
                            "bytes_removed",
                            "net_bytes",
                            "scope",
                        )
                    },
                    "oracle_bound": bound,
                }
            )
        return rows

    hold_fit = _fit("hold_interpolating", x_hold)
    train_fit = _fit("train_generator_hold_codes", x_train)
    named = next(r for r in hold_fit if int(r["rank"]) == NAMED_GENERATOR_RANK)
    interp = next(r for r in hold_fit if int(r["rank"]) == interp_rank)
    reconstruction_clears = bool(interp["clears_transition_bar"])
    bound = interp["oracle_bound"]
    verdict = decide_verdict(
        reconstruction_clears=reconstruction_clears, bound=bound
    )
    return {
        "family": "generated_coefficients",
        "verdict": verdict,
        "capability": UNMEASURED,
        "unfairness": (
            "Shared generator is PCA of residual-stream X. Codes are a "
            "least-squares map fitted on the scored hold tokens (perfect "
            "per-block codes, no runtime constraint). The interpolating "
            "operating point uses a generator fitted on the hold X itself."
        ),
        "scored_on": {
            "split": "hold",
            "held_out_unit": "prompt_id",
            "prompt_ids": [s["prompt_id"] for s in hold],
            "n_tokens": [int(s["n_tokens"]) for s in hold],
            "layer": JUDGE_LAYER,
        },
        "fit_on_train_prompt_ids": [s["prompt_id"] for s in train],
        "reconstruction": {
            "of": "transition coefficients (q,k,v,z,decay,beta) and one-step gated-delta",
            "bar": TRANSITION_BAR,
            "named_rank": NAMED_GENERATOR_RANK,
            "named_rank_hold_interpolating": {
                "coefficient_relative_l2": named["coefficient_relative_l2"],
                "one_step_state_relative_l2_max": named["one_step_state_relative_l2_max"],
                "clears_transition_bar": named["clears_transition_bar"],
            },
            "interpolating_rank": interp_rank,
            "interpolating": {
                "coefficient_relative_l2": interp["coefficient_relative_l2"],
                "one_step_state_relative_l2_max": interp["one_step_state_relative_l2_max"],
                "clears_transition_bar": interp["clears_transition_bar"],
            },
            "clears_bar": reconstruction_clears,
        },
        "operating_points": hold_fit,
        "train_generator_hold_codes": train_fit,
        "oracle_bound": bound,
        "named_rank_bound": named["oracle_bound"],
        "why": (
            "Even a hold-interpolating generator whose rank equals the scored "
            f"token count (r={interp_rank}) only removes qkvz weight_codes "
            f"({bound['cited_ms_removed']['weight_codes']} ms cited) and does "
            f"not clear the {bound['residual_gap_ms_cited']} ms residual. "
            f"At the claimed rank {NAMED_GENERATOR_RANK} the transition is "
            f"not reconstructed "
            f"(coeff relative L2 {named['coefficient_relative_l2']:.3f}, "
            f"one-step state {named['one_step_state_relative_l2_max']:.3f}, "
            f"bar {TRANSITION_BAR}). A fair runtime generator is strictly weaker."
        ),
    }


def oracle_learned_recurrence(
    train: Sequence[Mapping[str, Any]],
    hold: Sequence[Mapping[str, Any]],
    *,
    d_states: Sequence[int] = LEARNED_DSTATES,
    residual: float | None = None,
) -> dict[str, Any]:
    """Compact recurrence fitted directly on the recorded S trajectory.

    No parameter-budget discipline. Per-head PCA of vec(S) gives the
    dimension a store would need; a linear SSM in those coordinates is
    the unfair dynamics. The budget actually needed is the d_state at
    which the STORE hits the bar, billed through the recorded candidate
    scaled to that d_state.
    """
    assert_judge_holdout(hold)
    assert_held_out_sequences(hold, train_ids=[s["prompt_id"] for s in train])
    gap = float(residual if residual is not None else residual_gap_ms())
    per_prompt: list[dict[str, Any]] = []
    needed_store: list[int] = []
    for seq in hold:
        bundle = coefficients_of(seq)
        states, _hs = roll_state(bundle)
        t, n_heads, dim, _ = states.shape
        head_rows: list[dict[str, Any]] = []
        u_parts = np.concatenate(
            [
                bundle["k"].reshape(t, -1),
                bundle["v"].reshape(t, -1),
                bundle["decay"],
                bundle["beta"],
            ],
            axis=1,
        ).astype(np.float64)
        for h in range(n_heads):
            m = states[:, h].reshape(t, -1).astype(np.float64)
            mean = m.mean(axis=0)
            mc = m - mean
            _u, svals, vt = np.linalg.svd(mc, full_matrices=False)
            energy = _energy_ranks(svals)
            needed_store.append(int(energy["rank_at_store_bar"]))
            dyn_rows: list[dict[str, Any]] = []
            for d in d_states:
                r = min(int(d), int(vt.shape[0]), max(t - 1, 1))
                p = vt[:r]
                z = mc @ p.T
                z_prev, z_next = z[:-1], z[1:]
                u = u_parts[1:]
                phi = np.concatenate([z_prev, u], axis=1)
                sol, *_ = np.linalg.lstsq(phi, z_next, rcond=None)
                pred = phi @ sol
                store_rel = relative_l2((z @ p) + mean, m)
                one_step_z = relative_l2(pred, z_next)
                shat = (pred @ p) + mean
                one_step_s = relative_l2(shat, m[1:])
                zhat = np.zeros_like(z)
                zhat[0] = z[0]
                for i in range(1, t):
                    zhat[i] = np.concatenate([zhat[i - 1], u_parts[i]]) @ sol
                unroll_s = relative_l2((zhat @ p) + mean, m)
                dyn_rows.append(
                    {
                        "d_state": int(d),
                        "rank_used": r,
                        "store_relative_l2": store_rel,
                        "one_step_z_relative_l2": one_step_z,
                        "one_step_state_relative_l2": one_step_s,
                        "unroll_state_relative_l2": unroll_s,
                        "clears_transition_bar": bool(
                            one_step_s <= TRANSITION_BAR and unroll_s <= TRANSITION_BAR
                        ),
                    }
                )
            head_rows.append(
                {
                    "head": h,
                    "energy": energy,
                    "dynamics": dyn_rows,
                }
            )
        per_prompt.append(
            {
                "prompt_id": seq["prompt_id"],
                "n_tokens": t,
                "split": "hold",
                "heads": head_rows,
                "rank_at_store_bar_max": int(
                    max(int(h["energy"]["rank_at_store_bar"]) for h in head_rows)
                ),
                "rank_at_0.99_max": int(
                    max(int(h["energy"]["rank_at_0.99"]) for h in head_rows)
                ),
            }
        )

    d_needed = int(max(needed_store)) if needed_store else int(max(d_states))
    named_dyn = [
        row
        for p in per_prompt
        for head in p["heads"]
        for row in head["dynamics"]
        if int(row["d_state"]) == NAMED_DSTATE
    ]
    largest = max(int(d) for d in d_states)
    largest_dyn = [
        row
        for p in per_prompt
        for head in p["heads"]
        for row in head["dynamics"]
        if int(row["d_state"]) == largest
    ]
    named_clear = bool(named_dyn) and all(r["clears_transition_bar"] for r in named_dyn)
    # A store of rank d_needed is not a recurrence. Dynamics at every
    # swept d_state are the authority for "same useful transition":
    # every head of every hold prompt must clear.
    reconstruction_clears = bool(largest_dyn) and all(
        r["clears_transition_bar"] for r in largest_dyn
    )
    needed_model = learned_byte_model(d_needed)
    named_model = learned_byte_model(NAMED_DSTATE)
    needed_bound = bound_from_added_removed(
        needed_model["bytes_added"], needed_model["bytes_removed"], residual=gap
    )
    named_bound = bound_from_added_removed(
        named_model["bytes_added"], named_model["bytes_removed"], residual=gap
    )
    bound = needed_bound
    verdict = decide_verdict(
        reconstruction_clears=reconstruction_clears, bound=bound
    )
    one_step_named = (
        float(max(r["one_step_state_relative_l2"] for r in named_dyn))
        if named_dyn
        else None
    )
    unroll_named = (
        float(max(r["unroll_state_relative_l2"] for r in named_dyn))
        if named_dyn
        else None
    )
    return {
        "family": "learned_recurrence",
        "verdict": verdict,
        "capability": UNMEASURED,
        "unfairness": (
            "PCA basis of vec(S) is fitted on the scored hold trajectory "
            "(sequence-specific, cannot exist as a runtime basis without "
            "storing it). Linear A,B are least-squares on that same "
            "trajectory. No parameter-budget cap was applied."
        ),
        "scored_on": {
            "split": "hold",
            "held_out_unit": "prompt_id",
            "prompt_ids": [s["prompt_id"] for s in hold],
            "n_tokens": [int(s["n_tokens"]) for s in hold],
            "layer": JUDGE_LAYER,
        },
        "fit_on_train_prompt_ids": [s["prompt_id"] for s in train],
        "budget_actually_needed": {
            "d_state_for_store_bar": d_needed,
            "store_bar": TRANSITION_BAR,
            "named_d_state": NAMED_DSTATE,
            "note": (
                "d_state_for_store_bar is a LOWER BOUND on the dimension of a "
                "store of this trajectory (relative L2 <= 0.01). A recurrence "
                "that has to GENERATE the trajectory cannot be smaller. The "
                "linear SSM at every swept d_state fails the transition bar."
            ),
        },
        "reconstruction": {
            "of": "recorded S trajectory under a linear SSM in the PCA subspace",
            "bar": TRANSITION_BAR,
            "named_d_state": NAMED_DSTATE,
            "named_one_step_state_relative_l2_max": one_step_named,
            "named_unroll_state_relative_l2_max": unroll_named,
            "named_clears_transition_bar": named_clear,
            "any_d_state_clears": reconstruction_clears,
            "clears_bar": reconstruction_clears,
        },
        "per_prompt": [
            {
                "prompt_id": p["prompt_id"],
                "n_tokens": p["n_tokens"],
                "rank_at_store_bar_max": p["rank_at_store_bar_max"],
                "rank_at_0.99_max": p["rank_at_0.99_max"],
                "head0_energy": p["heads"][0]["energy"] if p["heads"] else None,
                "head0_dynamics": p["heads"][0]["dynamics"] if p["heads"] else None,
            }
            for p in per_prompt
        ],
        "oracle_bound": bound,
        "named_d_state_bound": named_bound,
        "byte_model_at_needed_d_state": {
            k: needed_model[k]
            for k in (
                "d_state",
                "in_proj_bytes",
                "a_diag_bytes",
                "bytes_added",
                "bytes_removed",
                "net_bytes",
            )
        },
        "why": (
            f"The recorded S trajectory on hold fills rank "
            f"(store-bar d_state {d_needed} vs named {NAMED_DSTATE}). A linear "
            f"SSM at d_state={NAMED_DSTATE} has one-step state relative L2 "
            f"{one_step_named} and unroll {unroll_named} against bar "
            f"{TRANSITION_BAR}. Scaling in_proj to d_state={d_needed} "
            f"bills {needed_bound['cited_ms_added']['total']} ms cited added "
            f"against {needed_bound['cited_ms_removed']['total']} ms cited "
            f"removed; the bound "
            f"{needed_bound['opportunity_bound_ms']} ms "
            f"{'clears' if needed_bound['clears_residual_gap'] else 'does not clear'} "
            f"the {needed_bound['residual_gap_ms_cited']} ms residual. "
            "A fair recurrence is strictly weaker than this oracle."
        ),
    }


def _kmeans(
    x: np.ndarray, k: int, *, iters: int = 8, seed: int = RNG_SEED
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])
    k = int(min(max(int(k), 1), n))
    rng = np.random.default_rng(int(seed))
    centers = np.array(x[rng.choice(n, size=k, replace=False)], dtype=np.float32, copy=True)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(int(iters)):
        labels = np.empty(n, dtype=np.int32)
        for i in range(0, n, 32):
            sl = x[i : i + 32]
            dist = ((sl[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels[i : i + sl.shape[0]] = dist.argmin(1).astype(np.int32)
        for j in range(k):
            m = labels == j
            if bool(m.any()):
                centers[j] = x[m].mean(axis=0)
            else:
                centers[j] = x[int(rng.integers(0, n))]
    return labels, centers


def oracle_conditional_recurrence(
    train: Sequence[Mapping[str, Any]],
    hold: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = CONDITIONAL_KS,
    residual: float | None = None,
) -> dict[str, Any]:
    """Perfect regime assignment, one rule per regime, bill ALL rules.

    Tokens are clustered on (k, v, decay, beta). Each cluster's centroid is
    the rule. The oracle router assigns the true cluster (cannot exist at
    runtime). q is left as the true readout (additional unfairness). If
    this cannot reconstruct the transition inside a byte budget that pays,
    the school dies without anyone building a router.
    """
    assert_judge_holdout(hold)
    assert_held_out_sequences(hold, train_ids=[s["prompt_id"] for s in train])
    gap = float(residual if residual is not None else residual_gap_ms())
    points: list[dict[str, Any]] = []
    for k in ks:
        per_prompt: list[dict[str, Any]] = []
        for seq in hold:
            bundle = coefficients_of(seq)
            t = int(bundle["q"].shape[0])
            feat = np.concatenate(
                [
                    bundle["k"].reshape(t, -1),
                    bundle["v"].reshape(t, -1),
                    bundle["decay"],
                    bundle["beta"],
                ],
                axis=1,
            )
            labels, centers = _kmeans(feat, int(k), seed=RNG_SEED)
            kdim = JUDGE_HEADS * JUDGE_DIM
            hat = {
                "q": bundle["q"],
                "z": bundle["z"],
                "k": np.empty_like(bundle["k"]),
                "v": np.empty_like(bundle["v"]),
                "decay": np.empty_like(bundle["decay"]),
                "beta": np.empty_like(bundle["beta"]),
            }
            for i in range(t):
                f = centers[int(labels[i])]
                hat["k"][i] = f[:kdim].reshape(JUDGE_HEADS, JUDGE_DIM)
                hat["v"][i] = f[kdim : 2 * kdim].reshape(JUDGE_HEADS, JUDGE_DIM)
                hat["decay"][i] = f[2 * kdim : 2 * kdim + JUDGE_HEADS]
                hat["beta"][i] = f[2 * kdim + JUDGE_HEADS : 2 * kdim + 2 * JUDGE_HEADS]
            err = transition_errors(bundle, hat)
            per_prompt.append(
                {
                    "prompt_id": seq["prompt_id"],
                    "n_tokens": t,
                    "n_regimes_used": int(len(set(int(x) for x in labels.tolist()))),
                    "one_step_state_relative_l2_max": err["one_step_state_relative_l2_max"],
                    "one_step_state_relative_l2_mean": err["one_step_state_relative_l2_mean"],
                    "clears_transition_bar": err["clears_transition_bar"],
                    "rolled_by_horizon": err["rolled_by_horizon"],
                }
            )
        one_step_max = _worst(per_prompt, "one_step_state_relative_l2_max")
        model = conditional_rule_bytes(int(k))
        bound = bound_from_added_removed(
            model["bytes_added"], model["bytes_removed"], residual=gap
        )
        points.append(
            {
                "n_regimes": int(k),
                "one_step_state_relative_l2_max": one_step_max,
                "clears_transition_bar": bool(one_step_max <= TRANSITION_BAR),
                "per_prompt": per_prompt,
                "byte_model": {
                    k2: model[k2]
                    for k2 in (
                        "n_regimes",
                        "bytes_per_rule",
                        "router_bytes",
                        "router_cited_ms",
                        "bytes_added",
                        "bytes_removed",
                        "net_bytes",
                    )
                },
                "oracle_bound": bound,
            }
        )

    best_recon = min(points, key=lambda r: float(r["one_step_state_relative_l2_max"]))
    reconstruction_clears = bool(best_recon["clears_transition_bar"])
    # Economics of the cheapest K that clears, else of the best reconstruction.
    paying = [
        p for p in points if p["clears_transition_bar"] and p["oracle_bound"]["clears_residual_gap"]
    ]
    headline = paying[0] if paying else best_recon
    bound = headline["oracle_bound"]
    verdict = decide_verdict(
        reconstruction_clears=reconstruction_clears, bound=bound
    )
    return {
        "family": "conditional_recurrence",
        "verdict": verdict,
        "capability": UNMEASURED,
        "unfairness": (
            "k-means on the scored hold (k,v,decay,beta). Perfect assignment "
            "to the true cluster (oracle router, cannot exist at runtime, "
            "billed at broadcast_aux = 0.000 ms/GB). True q is kept as the "
            "readout. ALL K rules are billed."
        ),
        "scored_on": {
            "split": "hold",
            "held_out_unit": "prompt_id",
            "prompt_ids": [s["prompt_id"] for s in hold],
            "n_tokens": [int(s["n_tokens"]) for s in hold],
            "layer": JUDGE_LAYER,
        },
        "fit_on_train_prompt_ids": [s["prompt_id"] for s in train],
        "reconstruction": {
            "of": "one-step gated-delta under centroid (k,v,decay,beta) per regime",
            "bar": TRANSITION_BAR,
            "best_n_regimes": int(best_recon["n_regimes"]),
            "best_one_step_state_relative_l2_max": best_recon["one_step_state_relative_l2_max"],
            "clears_bar": reconstruction_clears,
        },
        "operating_points": points,
        "oracle_bound": bound,
        "why": (
            f"Perfect routing into {best_recon['n_regimes']} regimes still "
            f"leaves one-step state relative L2 "
            f"{best_recon['one_step_state_relative_l2_max']:.3f} against bar "
            f"{TRANSITION_BAR}. Billing all {headline['n_regimes']} rules is "
            f"{bound['opportunity_bound_ms']} ms cited "
            f"({bound['cited_ms_removed']['total']} removed − "
            f"{bound['cited_ms_added']['total']} added) against a "
            f"{bound['residual_gap_ms_cited']} ms residual. A real router is "
            "strictly weaker than this oracle, and the named beta-skip "
            "candidate removes zero catalog bytes."
        ),
    }


def evaluate_family(
    family: str,
    *,
    payload: Path | None = None,
    hold: Sequence[Mapping[str, Any]] | None = None,
    train: Sequence[Mapping[str, Any]] | None = None,
    residual: float | None = None,
) -> dict[str, Any]:
    """Score one family. Missing input RAISES; the family is not skipped."""
    if family not in REQUIRED_FAMILIES:
        raise OracleRefuse(
            f"REFUSED: unknown family {family!r}; known {list(REQUIRED_FAMILIES)}"
        )
    try:
        hold_seq = list(hold) if hold is not None else load_hold_sequences(payload)
        train_seq = list(train) if train is not None else load_train_sequences(payload)
    except CorpusUnavailable:
        raise
    except FileNotFoundError as exc:
        raise CorpusUnavailable([getattr(exc, "filename", payload or "?")]) from exc
    if not hold_seq:
        raise FamilyInputMissing(
            f"REFUSED: family {family} has no hold sequences; "
            "missing input must not become a skipped family"
        )
    if family == "generated_coefficients":
        return oracle_generated_coefficients(train_seq, hold_seq, residual=residual)
    if family == "learned_recurrence":
        return oracle_learned_recurrence(train_seq, hold_seq, residual=residual)
    return oracle_conditional_recurrence(train_seq, hold_seq, residual=residual)


def blocked_family(family: str, *, missing_paths: Sequence[str], detail: str) -> dict[str, Any]:
    empty_added = dsf.added()
    empty_removed = dsf.removed()
    bound = bound_from_added_removed(empty_added, empty_removed)
    bound["opportunity_bound_ms"] = None
    bound["clears_residual_gap"] = False
    bound["blocked"] = True
    bound["missing_paths"] = list(missing_paths)
    return {
        "family": family,
        "verdict": BLOCKED,
        "capability": UNMEASURED,
        "missing_paths": list(missing_paths),
        "detail": detail,
        "oracle_bound": bound,
        "reconstruction": {
            "clears_bar": False,
            "bar": TRANSITION_BAR,
            "of": None,
        },
        "scored_on": {
            "split": "hold",
            "held_out_unit": "prompt_id",
            "prompt_ids": list(JUDGE_HOLD_PROMPT_IDS),
            "blocked": True,
        },
        "why": (
            f"BLOCKED: corpus not readable at {list(missing_paths)}. "
            "Refusing to synthesise X (NNS-001)."
        ),
    }


def require_all_families(doc: Mapping[str, Any]) -> None:
    families = doc.get("families")
    if not isinstance(families, Mapping):
        raise FamilyAbsentRefuse(
            "REFUSED: families mapping is absent; an absent family reads as a judged one"
        )
    missing = [f for f in REQUIRED_FAMILIES if f not in families]
    if missing:
        raise FamilyAbsentRefuse(
            f"REFUSED: families {missing} are absent; an absent family reads as a judged one"
        )
    for name in REQUIRED_FAMILIES:
        row = families[name]
        if not isinstance(row, Mapping):
            raise FamilyAbsentRefuse(f"REFUSED: family {name} is not a mapping")
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise FamilyAbsentRefuse(
                f"REFUSED: family {name} has verdict {verdict!r}, not one of {list(VERDICTS)}"
            )
        if "oracle_bound" not in row:
            raise FamilyAbsentRefuse(
                f"REFUSED: family {name} has no oracle_bound; a bound-less family reads as a pass"
            )


def _input_block(hold: Sequence[Mapping[str, Any]], train: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "kind": JUDGE_KIND,
        "layer": JUDGE_LAYER,
        "n_heads": JUDGE_HEADS,
        "dim": JUDGE_DIM,
        "n_sequences": len(hold),
        "n_tokens": [int(s["n_tokens"]) for s in hold],
        "prompt_ids": [s["prompt_id"] for s in hold],
        "train_prompt_ids_used_for_generator": [s["prompt_id"] for s in train],
        "report_as": "held_out",
        "held_out_unit": "prompt_id",
        "coefficients": (
            "seeded STATIC map of X, shader L2 on q/k; not W_qkvz. "
            "Same map as DELTANET_MULTISTEP."
        ),
        "trained_qkv_trajectory": UNMEASURED,
        "x_is_mixer_input": False,
        "x_is_post_attn_norm": True,
        "same_corpus_as": PREDECESSOR_MULTISTEP,
        "payload": str(resolve_payload_dir()),
    }


def build(*, payload: Path | None = None) -> dict[str, Any]:
    gap = residual_gap_ms()
    price = sme.price()
    prior = {
        "not_neutral": True,
        "already_judged": {
            "smaller_state": {
                "verdict": "MEASURED_NEGATIVE",
                "shape": "ONSET_THEN_PLATEAU",
                "source": PREDECESSOR_MULTISTEP,
            },
            "structured_transitions": {
                "verdict": "MEASURED_NEGATIVE",
                "shape": "PLATEAU",
                "source": PREDECESSOR_MULTISTEP,
            },
        },
        "fused_update_and_consume": "NOT_A_TRAJECTORY_QUESTION",
        "expect": (
            "Two of six are already MEASURED_NEGATIVE. The prior on these "
            "three is not neutral. A well-supported ORACLE_NEGATIVE is the "
            "deliverable: it closes S020 §16 and stops the campaign returning."
        ),
    }
    try:
        hold = load_hold_sequences(payload)
        train = load_train_sequences(payload)
        families = {
            "generated_coefficients": oracle_generated_coefficients(
                train, hold, residual=gap
            ),
            "learned_recurrence": oracle_learned_recurrence(
                train, hold, residual=gap
            ),
            "conditional_recurrence": oracle_conditional_recurrence(
                train, hold, residual=gap
            ),
        }
        input_block = _input_block(hold, train)
        corpus_error = None
    except CorpusUnavailable as exc:
        families = {
            name: blocked_family(
                name, missing_paths=exc.missing_paths, detail=str(exc)
            )
            for name in REQUIRED_FAMILIES
        }
        input_block = {
            "kind": JUDGE_KIND,
            "layer": JUDGE_LAYER,
            "prompt_ids": list(JUDGE_HOLD_PROMPT_IDS),
            "report_as": "held_out",
            "blocked": True,
            "missing_paths": list(exc.missing_paths),
        }
        corpus_error = str(exc)

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "purpose": (
            "Oracle-first bounds for generated_coefficients, learned_recurrence "
            "and conditional_recurrence. If an unfair oracle cannot make the "
            "economics work, do not build the real version."
        ),
        "input": input_block,
        "corpus_error": corpus_error,
        "predecessor": {
            "multistep": PREDECESSOR_MULTISTEP,
            "state_function": PREDECESSOR_STATE,
            "economics": PREDECESSOR_ECON,
            "calibration": CALIBRATION_REL,
        },
        "prior": prior,
        "bar": {
            "transition_relative_l2": TRANSITION_BAR,
            "residual_gap_ms_cited": gap,
            "residual_source": PREDECESSOR_ECON,
            "opportunity_bound_kind": BOUND_KIND,
            "capability": UNMEASURED,
            "rates_used": stream_rates(),
        },
        "price_context": {
            "token_ms_cited": price["token_ms"],
            "gap_to_71_raw_ms_cited": price["gap_to_71_raw_ms"],
            "gap_to_71_residual_ms_cited": price[
                "gap_to_71_residual_after_everything_on_record_ms"
            ],
            "upper_bound_ms_cited": price["upper_bound_ms"],
            "halved_ms_cited": price["halved_ms"],
            "basis": price["basis"],
            "note": (
                "Cited from DELTANET_STATE_MACHINE_ECONOMICS, not re-measured. "
                "The residual is the gap a NEW lever has to close."
            ),
        },
        "families": families,
        "answers": {
            "closable_as_oracle_negative": all(
                families[f]["verdict"] == ORACLE_NEGATIVE for f in REQUIRED_FAMILIES
            ),
            "any_permits_investigation": any(
                families[f]["verdict"] == ORACLE_PERMITS_INVESTIGATION
                for f in REQUIRED_FAMILIES
            ),
            "any_blocked": any(families[f]["verdict"] == BLOCKED for f in REQUIRED_FAMILIES),
            "verdicts": {f: families[f]["verdict"] for f in REQUIRED_FAMILIES},
        },
        "what_this_does_not_prove": [
            "Capability. Every family's capability is UNMEASURED.",
            "A generate-identity gate on the sealed residual stream.",
            "That X equals the sealed mixer input (capture_diverse2 is post_attn_norm).",
            "That the seeded STATIC coefficient map is W_qkvz.",
            "A measured speedup. Every millisecond is cited and pro-rated.",
            "A hardware number. gpu_authority is false.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    require_all_families(doc)
    return _py(doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT, doc, RECORDED_BY))
        return 0
    print(json.dumps({f: doc["families"][f]["verdict"] for f in REQUIRED_FAMILIES}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
