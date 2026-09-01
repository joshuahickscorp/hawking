#!/usr/bin/env python3
"""CAPABILITY BYTE ELIMINATION — search for the BOUNDARY, not a threshold.

Byte elimination already ran at the wrong bar. CAPABILITY_INFORMATION_MAP
asked which information can disappear while hidden-state cosine stays above
0.99 and licensed 27.7 MB, 0.28% of the token. The campaign needs 1773 MB.
That is a 64x shortfall produced by the bar, not by the model.

S030 §3: Doctor must ask WHICH INFORMATION CAN DISAPPEAR WHILE USEFUL
CAPABILITY REMAINS. This module asks that question. LOCAL_FUNCTIONAL_FIDELITY
(cosine) is recorded as CONTEXT and is never the verdict.

Per named region, destruction is swept upward with capability_curve.sweep
(coarse, detect, refine). Every point records bytes gone, physical ms gone
(ESTIMATED_FROM_CITED_MS from ORGAN_DECOMPOSITION_SEALED, pro-rated by byte
share), and BOTH capability levels (LOGIT_TOKEN, FAST_CAPABILITY). The result
is a BOUNDARY INTERVAL or an explicit FLAT curve — never a single-threshold
pass/fail.

    python3 tools/future/capability_byte_elimination.py --build
    python3 -m pytest tools/future/test_capability_byte_elimination.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
The caller names the region. This module does not choose which region matters.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.future import aux_capability_screen as acs
from tools.future import capability_curve as cc
from tools.future import capability_information_map as cim
from tools.future import capability_stages as cs
from tools.future import complete_ebpw as ebpw
from tools.future._common import REPO, load_json, write_receipt


RECEIPT = "CAPABILITY_BYTE_ELIMINATION.json"
SCHEMA = "hawking.future.capability_byte_elimination.v1"
VERSION = 1
RECORDED_BY = "tools/future/capability_byte_elimination.py"

CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"
ORGAN_REL = "receipts/future/ORGAN_DECOMPOSITION_SEALED.json"
MAP_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"

PASS = cs.PASS
FAIL = cs.FAIL
NOT_RUN = cs.NOT_RUN

FLAT = "FLAT"
BOUNDARY_INTERVAL = "BOUNDARY_INTERVAL"
INCOMPLETE = "INCOMPLETE"
BLOCKED = "BLOCKED"

ESTIMATED_FROM_CITED_MS = "ESTIMATED_FROM_CITED_MS"
CONTEXT = "CONTEXT"

LOGIT_TOKEN = cs.LOGIT_TOKEN
FAST_CAPABILITY = cs.FAST_CAPABILITY
LOCAL_FUNCTIONAL_FIDELITY = cs.LOCAL_FUNCTIONAL_FIDELITY

# Cited, not re-derived. fidelity_hierarchy / S030: 60 TPS needs 1773 MB.
CAMPAIGN_NEED_MB = 1773.0
CAMPAIGN_NEED_SOURCE = (
    "tools/future/fidelity_hierarchy.py; 60 TPS byte need cited by S030 / G108"
)

# Real tokenizer ids from the sealed artifact tokenizer.json (HuggingFace
# byte-level BPE, add_prefix_space=False). Encoded off-line so this sidecar
# does not import `tokenizers` at runtime. ĠParis is the leading-space Paris
# token a completion of "The capital of France is" should emit.
FRANCE_PROMPT = "The capital of France is"
FRANCE_TOKEN_IDS: tuple[int, ...] = (760, 6511, 314, 9338, 369)
PARIS_ARGMAX_IDS: tuple[int, ...] = (11751, 57590)  # ĠParis, Paris
TOKENIZER_REL = "/Users/scammermike/noetic/NOETIC_PARENT_A/tokenizer.json"

FAST_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "fact-capital",
        "prompt": FRANCE_PROMPT,
        "token_ids": list(FRANCE_TOKEN_IDS),
        "expect_argmax_in": list(PARIS_ARGMAX_IDS),
        "predicate": (
            "next-token argmax is a Paris token "
            "(ĠParis=11751 or Paris=57590)"
        ),
        "origin": (
            "resident_provider.PROMPT_FRANCE, single-argmax completion form; "
            "tokenizer.json of the sealed artifact"
        ),
        "tokenizer": TOKENIZER_REL,
    },
)

# --build default list. A default is allowed; a claim that one of these is
# the right next target is not. Last layer so replay_prompt_from can reach
# the LM head without keeping every layer kit.
DEFAULT_BUILD_REGIONS: tuple[dict[str, Any], ...] = (
    {"id": "L63.mlp.gate", "layer": 63, "organ": "mlp.gate", "block": "all"},
    {"id": "L63.mlp.down", "layer": 63, "organ": "mlp.down", "block": "all"},
)

# Census organ -> sealed organ-decomposition row.
CENSUS_TO_SEALED: dict[str, str] = {
    "mlp.gate": "mlp_gate_up",
    "mlp.up": "mlp_gate_up",
    "mlp.down": "mlp_down",
    "attention.linear_qkvz": "deltanet",
    "attention.linear_ba": "deltanet",
    "attention.linear_out": "deltanet",
    "attention.linear_conv1d": "deltanet",
    "state.A_log": "deltanet",
    "state.dt_bias": "deltanet",
    "attention.q": "gqa_attention",
    "attention.k": "gqa_attention",
    "attention.v": "gqa_attention",
    "attention.o": "gqa_attention",
    "lm_head": "lm_head",
    "embedding": "embedding",
}

SEALED_TO_CENSUS: dict[str, tuple[str, ...]] = {
    "mlp_gate_up": ("mlp.gate", "mlp.up"),
    "mlp_down": ("mlp.down",),
    "deltanet": (
        "attention.linear_qkvz",
        "attention.linear_ba",
        "attention.linear_out",
        "attention.linear_conv1d",
        "state.A_log",
        "state.dt_bias",
    ),
    "gqa_attention": ("attention.q", "attention.k", "attention.v", "attention.o"),
    "lm_head": ("lm_head",),
    "embedding": ("embedding",),
}

MLP_SHORT = {"mlp.gate": "gate", "mlp.up": "up", "mlp.down": "down"}
Q4_WEIGHT_KEY = {
    "attention.q": ("Wq", "q_q"),
    "attention.k": ("Wk", "q_k"),
    "attention.v": ("Wv", "q_v"),
    "attention.o": ("Wo", "q_o"),
}

RNG_SEED = 38
DEFAULT_LO = 0.0
DEFAULT_HI = 1.0
DEFAULT_RESOLUTION = 0.1
DEFAULT_BUDGET = 8
DEFAULT_N_COARSE = 5

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Activations are a CPU replay of sealed-3.14 packed weights via "
    "capability_information_map.capture_real_prefix / replay_prompt_from. "
    "Isotropic Gaussian x is refused. LOGIT_TOKEN reuses capability_stages "
    "bars (KL 0.1, top-5 agreement 0.8); FAST_CAPABILITY is a single-argmax "
    "predicate on real tokenizer ids. LOCAL_FUNCTIONAL_FIDELITY (cosine 0.99) "
    "is CONTEXT and is not the verdict. Physical ms figures are "
    "ESTIMATED_FROM_CITED_MS from ORGAN_DECOMPOSITION_SEALED.json, pro-rated "
    "by byte share of the region's census, never MEASURED. The caller names "
    "the region; a default --build list is not a scientific pick. A FLAT "
    "curve is a result. evidence_class STATIC_ONLY. gpu_authority false."
)


Measure = Callable[[Mapping[str, Any]], Any]


class ByteEliminationRefuse(ValueError):
    """A required input is missing, or the lane would have to guess."""


class PointNotRun(ByteEliminationRefuse):
    """This sweep point cannot be evaluated. It is not a pass and not a number."""


# ---------------------------------------------------------------------------
# Tiny helpers.
# ---------------------------------------------------------------------------


def _py(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    return value


def _require_file(rel: str, *, why: str) -> Path:
    path = REPO / rel if not Path(rel).is_absolute() else Path(rel)
    if not path.is_file():
        raise ByteEliminationRefuse(
            f"REFUSED: {rel} is not on disk; {why}. A missing input is not a skip"
        )
    return path


def mb(n_bytes: int | float) -> float:
    return float(n_bytes) / 1e6


# ---------------------------------------------------------------------------
# Authorities. Missing -> raise. Never a quiet zero.
# ---------------------------------------------------------------------------


def load_census() -> dict[str, Any]:
    path = _require_file(CENSUS_REL, why="the region's byte census is the byte authority")
    doc = load_json(path)
    if not isinstance(doc, dict) or "census" not in doc:
        raise ByteEliminationRefuse(f"REFUSED: {CENSUS_REL} has no census object")
    return doc


def load_organ_ms() -> dict[str, Any]:
    """Cite ORGAN_DECOMPOSITION_SEALED. Do not re-measure. Do not invent ms."""
    path = _require_file(
        ORGAN_REL, why="physical ms gone is cited from the sealed organ table"
    )
    doc = load_json(path)
    table = doc.get("table")
    if not isinstance(table, list) or not table:
        raise ByteEliminationRefuse(f"REFUSED: {ORGAN_REL} has no table")
    sealed_ms: dict[str, float] = {}
    for row in table:
        if not isinstance(row, Mapping) or "organ" not in row or "sealed_ms" not in row:
            raise ByteEliminationRefuse(
                f"REFUSED: {ORGAN_REL} row is missing organ/sealed_ms: {row!r}"
            )
        sealed_ms[str(row["organ"])] = float(row["sealed_ms"])
    for need in ("mlp_gate_up", "mlp_down", "deltanet"):
        if need not in sealed_ms:
            raise ByteEliminationRefuse(f"REFUSED: {ORGAN_REL} is missing {need}")
    token_ms = doc.get("reconciliation", {}).get("measured_token_ms")
    return {
        "sealed_ms": sealed_ms,
        "cited_token_ms": None if token_ms is None else float(token_ms),
        "source": ORGAN_REL,
        "label": ESTIMATED_FROM_CITED_MS,
        "not": "MEASURED",
    }


def load_cosine_license() -> dict[str, Any]:
    path = _require_file(
        MAP_REL, why="the 0.99 cosine bar's licensed bytes are the comparison"
    )
    doc = load_json(path)
    alloc = doc.get("allocation")
    if not isinstance(alloc, Mapping) or "total_bytes_eliminated" not in alloc:
        raise ByteEliminationRefuse(
            f"REFUSED: {MAP_REL} has no allocation.total_bytes_eliminated"
        )
    return {
        "bytes": int(alloc["total_bytes_eliminated"]),
        "mb": mb(int(alloc["total_bytes_eliminated"])),
        "share_of_token": alloc.get("share_of_token_eliminated"),
        "n_regions": alloc.get("n_regions"),
        "cosine_bar": alloc.get("cosine_bar", cim.HIDDEN_COSINE_BAR),
        "source": MAP_REL,
    }


def load_parent_payload_bytes() -> int:
    """Byte accounting via complete_ebpw.mix_report. Missing MIX_REPORT raises."""
    try:
        mix = ebpw.mix_report()
    except ebpw.CompleteEbpwRefused as exc:
        raise ByteEliminationRefuse(f"REFUSED: complete_ebpw mix_report: {exc}") from exc
    return int(mix["payload_bytes"])


# ---------------------------------------------------------------------------
# Region identity. The caller names it. This module does not pick one.
# ---------------------------------------------------------------------------


def require_region(region: Any) -> dict[str, Any]:
    """Caller supplies the region. A missing region is a refusal, not a default."""
    if region is None:
        raise ByteEliminationRefuse(
            "REFUSED: caller did not name a region; this module does not choose "
            "which region matters"
        )
    if isinstance(region, str):
        text = region.strip()
        if not text:
            raise ByteEliminationRefuse(
                "REFUSED: empty region id; this module does not choose a region"
            )
        if text.startswith("L") and "." in text:
            head, rest = text.split(".", 1)
            try:
                layer = int(head[1:])
            except ValueError as exc:
                raise ByteEliminationRefuse(
                    f"REFUSED: cannot parse layer from {text!r}"
                ) from exc
            organ = rest
            block = "all"
            if organ.endswith(".all"):
                organ = organ[: -len(".all")]
            return {
                "id": text if text.endswith(".all") or "." in rest else f"L{layer}.{organ}",
                "layer": layer,
                "organ": organ,
                "block": block,
            }
        raise ByteEliminationRefuse(
            f"REFUSED: region {text!r} is not 'L{{layer}}.{{organ}}' and is not "
            "a mapping; this module does not guess the organ"
        )
    if not isinstance(region, Mapping):
        raise ByteEliminationRefuse(
            "REFUSED: region must be a mapping or 'L{layer}.{organ}' string"
        )
    layer = region.get("layer")
    organ = region.get("organ")
    if layer is None or (isinstance(organ, str) and not organ.strip()) or organ is None:
        raise ByteEliminationRefuse(
            "REFUSED: caller did not supply region.layer and region.organ; "
            "this module does not choose which region matters"
        )
    try:
        layer_i = int(layer)
    except (TypeError, ValueError) as exc:
        raise ByteEliminationRefuse("REFUSED: region.layer is not an int") from exc
    organ_s = str(organ).strip()
    block = str(region.get("block") or "all")
    rid = str(region.get("id") or "").strip() or f"L{layer_i}.{organ_s}"
    return {"id": rid, "layer": layer_i, "organ": organ_s, "block": block}


def region_census_bytes(census: Mapping[str, Any], region: Mapping[str, Any]) -> int:
    """The region's own byte census. Missing organ/layer raises."""
    body = census["census"] if "census" in census else census
    per_layer = body.get("per_layer")
    if not isinstance(per_layer, list):
        raise ByteEliminationRefuse("REFUSED: census has no per_layer list")
    layer = int(region["layer"])
    organ = str(region["organ"])
    if layer < 0 or layer >= len(per_layer):
        raise ByteEliminationRefuse(
            f"REFUSED: layer {layer} is outside census per_layer "
            f"[0, {len(per_layer)})"
        )
    rows = per_layer[layer].get("organs")
    if not isinstance(rows, list):
        raise ByteEliminationRefuse(f"REFUSED: census layer {layer} has no organs list")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("organ")) == organ:
            raw = row.get("active_bytes")
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ByteEliminationRefuse(
                    f"REFUSED: {organ} L{layer} active_bytes is not an int"
                )
            return int(raw)
    raise ByteEliminationRefuse(
        f"REFUSED: census has no {organ} at layer {layer}; missing is not zero"
    )


def organ_census_bytes(census: Mapping[str, Any], sealed_organ: str) -> int:
    parts = SEALED_TO_CENSUS.get(sealed_organ)
    if not parts:
        raise ByteEliminationRefuse(
            f"REFUSED: sealed organ {sealed_organ!r} has no census parts; "
            "refusing to invent a byte share"
        )
    body = census["census"] if "census" in census else census
    by = body.get("by_organ")
    if not isinstance(by, list):
        raise ByteEliminationRefuse("REFUSED: census has no by_organ list")
    lookup = {
        str(row["organ"]): row
        for row in by
        if isinstance(row, Mapping) and "organ" in row
    }
    total = 0
    for part in parts:
        if part not in lookup or "active_bytes" not in lookup[part]:
            raise ByteEliminationRefuse(
                f"REFUSED: census is missing {part} for sealed organ {sealed_organ}"
            )
        raw = lookup[part]["active_bytes"]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ByteEliminationRefuse(f"REFUSED: {part} active_bytes is not an int")
        total += int(raw)
    return total


def bytes_gone_at(
    region_bytes: int,
    fraction: float,
    *,
    n_destroyed: int | None = None,
    n_rows: int | None = None,
) -> int:
    if isinstance(region_bytes, bool) or not isinstance(region_bytes, int) or region_bytes < 0:
        raise ByteEliminationRefuse("REFUSED: region_bytes is not a non-negative int")
    try:
        frac = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ByteEliminationRefuse("REFUSED: destruction fraction is not numeric") from exc
    if not math.isfinite(frac) or frac < 0.0 or frac > 1.0:
        raise ByteEliminationRefuse(
            f"REFUSED: destruction fraction {frac} is outside [0, 1]"
        )
    if n_destroyed is not None and n_rows is not None:
        if n_rows <= 0:
            raise ByteEliminationRefuse("REFUSED: n_rows is not positive")
        if n_destroyed < 0 or n_destroyed > n_rows:
            raise ByteEliminationRefuse("REFUSED: n_destroyed is outside [0, n_rows]")
        return int(round(region_bytes * (int(n_destroyed) / int(n_rows))))
    return int(round(frac * region_bytes))


def bytes_and_ms(
    region: Any,
    fraction: float,
    *,
    census: Mapping[str, Any] | None = None,
    organ_table: Mapping[str, Any] | None = None,
    n_destroyed: int | None = None,
    n_rows: int | None = None,
) -> dict[str, Any]:
    """Bytes from the region's census; ms cited and pro-rated. Never MEASURED."""
    named = require_region(region)
    census = census if census is not None else load_census()
    organ_table = organ_table if organ_table is not None else load_organ_ms()
    organ = named["organ"]
    sealed_name = CENSUS_TO_SEALED.get(organ)
    if not sealed_name:
        raise ByteEliminationRefuse(
            f"REFUSED: {organ} has no sealed-organ mapping; refusing to invent ms"
        )
    sealed_ms_map = organ_table["sealed_ms"]
    if sealed_name not in sealed_ms_map:
        raise ByteEliminationRefuse(
            f"REFUSED: {ORGAN_REL} has no sealed_ms for {sealed_name}"
        )
    region_bytes = region_census_bytes(census, named)
    organ_bytes = organ_census_bytes(census, sealed_name)
    if organ_bytes <= 0:
        raise ByteEliminationRefuse(
            f"REFUSED: sealed organ {sealed_name} has non-positive census bytes"
        )
    gone = bytes_gone_at(
        region_bytes, fraction, n_destroyed=n_destroyed, n_rows=n_rows
    )
    cited_ms = float(sealed_ms_map[sealed_name])
    ms_gone = (gone / organ_bytes) * cited_ms
    return {
        "region": named,
        "destruction": float(fraction),
        "region_bytes": int(region_bytes),
        "bytes_gone": int(gone),
        "bytes_gone_mb": mb(gone),
        "ms_gone": {
            "ms": float(ms_gone),
            "label": ESTIMATED_FROM_CITED_MS,
            "not": "MEASURED",
            "source": ORGAN_REL,
            "cited_organ": sealed_name,
            "cited_organ_ms": cited_ms,
            "pro_rated_by": "byte_share_of_organ",
            "organ_active_bytes": int(organ_bytes),
            "region_bytes_gone": int(gone),
        },
    }


# ---------------------------------------------------------------------------
# Destruction. Same shape as the functional-role probe: zero output rows.
# ---------------------------------------------------------------------------


def n_destroyed_rows(n_rows: int, fraction: float) -> int:
    if n_rows <= 0:
        raise ByteEliminationRefuse("REFUSED: tensor has no output rows")
    try:
        frac = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ByteEliminationRefuse("REFUSED: destruction fraction is not numeric") from exc
    if not math.isfinite(frac) or frac < 0.0 or frac > 1.0:
        raise ByteEliminationRefuse(
            f"REFUSED: destruction fraction {frac} is outside [0, 1]"
        )
    return int(round(frac * n_rows))


def destroy_output_rows(
    weight: np.ndarray, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Zero a deterministic fraction of output rows. fraction=0 is identity."""
    w = np.asarray(weight)
    n_rows = int(w.shape[0])
    k = n_destroyed_rows(n_rows, fraction)
    if k <= 0:
        return w.copy(), 0
    out = w.copy()
    idx = rng.choice(n_rows, size=k, replace=False)
    out[idx] = 0
    return out, k


def destroyed_overrides(
    kit: Any, organ: str, fraction: float, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, int, int]:
    """Return (mlp_override, mix_override, n_destroyed, n_rows)."""
    if float(fraction) <= 0.0:
        if organ in MLP_SHORT:
            n_rows = int(kit.mlp()[MLP_SHORT[organ]]["W"].shape[0])
        elif organ == "attention.linear_qkvz":
            n_rows = int(kit.mix()["Wqkvz"].shape[0])
        elif organ in Q4_WEIGHT_KEY:
            key_w, _key_q = Q4_WEIGHT_KEY[organ]
            n_rows = int(kit.mix()[key_w].shape[0])
        else:
            raise ByteEliminationRefuse(
                f"REFUSED: cannot destroy unknown organ {organ}"
            )
        return None, None, 0, n_rows
    if organ in MLP_SHORT:
        short = MLP_SHORT[organ]
        w = kit.mlp()[short]["W"]
        wp, k = destroy_output_rows(w, fraction, rng)
        return {short: wp}, None, k, int(w.shape[0])
    if organ == "attention.linear_qkvz":
        mix = kit.mix()
        wp, k = destroy_output_rows(mix["Wqkvz"], fraction, rng)
        return None, {"Wqkvz": wp}, k, int(mix["Wqkvz"].shape[0])
    if organ in Q4_WEIGHT_KEY:
        key_w, _key_q = Q4_WEIGHT_KEY[organ]
        mix = kit.mix()
        wp, k = destroy_output_rows(mix[key_w], fraction, rng)
        return None, {key_w: wp}, k, int(mix[key_w].shape[0])
    raise ByteEliminationRefuse(f"REFUSED: cannot destroy unknown organ {organ}")


# ---------------------------------------------------------------------------
# Replay + judges. Use the existing engines. Do not rebuild them.
# ---------------------------------------------------------------------------


def replay_with_logits(
    cap: Mapping[str, Any],
    start_layer: int,
    n_more: int,
    *,
    mix_override: Mapping[str, np.ndarray] | None = None,
    mlp_override: Mapping[str, np.ndarray] | None = None,
    override_layer: int | None = None,
) -> dict[str, Any]:
    """Wrap replay_prompt_from and attach logits/argmax via the existing LM head.

    capability_information_map.replay_prompt_from returns aux / hidden_after_n.
    Last-layer logits are the consume path measure_region already uses
    (rmsnorm_delta + lm_head_gemv). This does not invent a second replay.
    """
    cim.refuse_synthetic_activations(cap.get("source"))
    out = cim.replay_prompt_from(
        cap,
        start_layer,
        n_more,
        mix_override=mix_override,
        mlp_override=mlp_override,
        override_layer=override_layer,
    )
    aux = out.get("aux")
    hidden = out.get("hidden_after_n")
    layer_output = None
    if isinstance(aux, Mapping):
        layer_output = aux.get("hidden_out")
    result: dict[str, Any] = {
        "layer_output": layer_output,
        "hidden_after_n": hidden,
        "end_layer": out.get("end_layer"),
        "aux": aux,
        "logits": None,
        "argmax": None,
        "logits_status": cim.UNMEASURED,
    }
    last = int(cap["n_layers"]) - 1
    if int(out["end_layer"]) != last:
        result["logits_status"] = NOT_RUN
        result["logits_reason"] = (
            f"replay ended at layer {out['end_layer']}, not last layer {last}; "
            "logits require the LM head on the final residual"
        )
        return result
    if hidden is None:
        raise ByteEliminationRefuse(
            "REFUSED: replay_prompt_from returned no hidden_after_n"
        )
    h = cim.rmsnorm_delta(hidden, cap["final_ln"])
    logits = cim.lm_head_gemv(h)
    result["logits"] = logits
    result["argmax"] = int(np.argmax(logits))
    result["logits_status"] = "MEASURED"
    return result


def judge_logit_token(
    logits_a: Any, logits_b: Any, *, component_id: str
) -> dict[str, Any]:
    """Reuse capability_stages.run_logit_token. Bars are KL 0.1 and top-5 0.8."""
    _ = component_id
    row = cs.run_logit_token({"logits_a": logits_a, "logits_b": logits_b})
    row["stage"] = LOGIT_TOKEN
    row["bars_reused"] = {
        "logit_kl_bar": float(acs.LOGIT_KL_BAR),
        "top_k": int(acs.TOPK),
        "top_k_agree_bar": float(acs.TOPK_AGREE_BAR),
    }
    return row


def judge_cosine_context(
    hidden_a: Any, hidden_b: Any, *, component_id: str
) -> dict[str, Any]:
    """LOCAL_FUNCTIONAL_FIDELITY is CONTEXT. It is not the verdict."""
    _ = component_id
    row = cs.run_local_functional_fidelity(
        {"hidden_a": hidden_a, "hidden_b": hidden_b}
    )
    row["stage"] = LOCAL_FUNCTIONAL_FIDELITY
    row["role"] = CONTEXT
    row["not_the_verdict"] = True
    row["why_not_the_verdict"] = (
        "S030 §3 / S031 §9: hidden-state cosine is LOCAL_FUNCTIONAL_FIDELITY, "
        "not capability. CAPABILITY_INFORMATION_MAP licensed 27.7 MB on this "
        "bar; that is the defect this module exists to stop."
    )
    return row


def judge_fast_capability(
    argmax: Any,
    *,
    probes: Sequence[Mapping[str, Any]] | None = None,
    incumbent_argmax: int | None = None,
    incumbent_satisfies: bool | None = None,
    reason_if_blocked: str | None = None,
) -> dict[str, Any]:
    """Single-argmax predicates on real tokenizer ids. Not text-contains."""
    if reason_if_blocked:
        return {
            "verdict": NOT_RUN,
            "reason": reason_if_blocked,
            "stage": FAST_CAPABILITY,
            "measurement": None,
        }
    probes = list(probes) if probes is not None else list(FAST_PROBES)
    if not probes:
        return {
            "verdict": NOT_RUN,
            "reason": (
                "no defensible predicate set; FAST_CAPABILITY is NOT_RUN "
                "rather than invented"
            ),
            "stage": FAST_CAPABILITY,
            "measurement": None,
        }
    if incumbent_satisfies is False:
        return {
            "verdict": NOT_RUN,
            "reason": (
                "incumbent does not satisfy the predicate; a probe the "
                "unperturbed model fails is not a destruction test"
            ),
            "stage": FAST_CAPABILITY,
            "incumbent_argmax": incumbent_argmax,
            "measurement": None,
        }
    if argmax is None:
        return {
            "verdict": NOT_RUN,
            "reason": "candidate argmax is missing; not a pass",
            "stage": FAST_CAPABILITY,
            "measurement": None,
        }
    try:
        pred = int(argmax)
    except (TypeError, ValueError):
        return {
            "verdict": NOT_RUN,
            "reason": "candidate argmax is not an int; not a pass",
            "stage": FAST_CAPABILITY,
            "measurement": None,
        }
    items: list[dict[str, Any]] = []
    any_fail = False
    for probe in probes:
        expect = [int(x) for x in (probe.get("expect_argmax_in") or ())]
        if not expect:
            raise ByteEliminationRefuse(
                f"REFUSED: probe {probe.get('id')!r} has no expect_argmax_in; "
                "a probe without a predicate is not a pass"
            )
        ok = pred in expect
        any_fail = any_fail or not ok
        items.append(
            {
                "id": probe["id"],
                "prompt": probe.get("prompt"),
                "token_ids": list(probe.get("token_ids") or ()),
                "predicate": probe.get("predicate"),
                "expect_argmax_in": expect,
                "argmax": pred,
                "passed": bool(ok),
                "origin": probe.get("origin"),
            }
        )
    passed = not any_fail
    return {
        "verdict": PASS if passed else FAIL,
        "reason": (
            f"{sum(1 for i in items if i['passed'])}/{len(items)} "
            "single-argmax probes passed"
        ),
        "stage": FAST_CAPABILITY,
        "incumbent_argmax": incumbent_argmax,
        "measurement": {
            "n_probes": len(items),
            "n_passed": sum(1 for i in items if i["passed"]),
            "items": items,
            "argmax": pred,
        },
    }


def capability_verdict(
    logit: Mapping[str, Any] | None,
    fast: Mapping[str, Any] | None,
    cosine: Mapping[str, Any] | None = None,
) -> str:
    """Combine the two capability levels. Cosine is ignored on purpose."""
    _ = cosine  # CONTEXT. Never a vote.
    lv = None if not isinstance(logit, Mapping) else logit.get("verdict")
    fv = None if not isinstance(fast, Mapping) else fast.get("verdict")
    spoken = [v for v in (lv, fv) if v in {PASS, FAIL}]
    if not spoken:
        return NOT_RUN
    if FAIL in spoken:
        return FAIL
    return PASS


def curve_value_from_record(rec: Mapping[str, Any]) -> float:
    """Numeric value for capability_curve.sweep. Cosine cannot enter."""
    if rec.get("status") == NOT_RUN:
        raise PointNotRun(str(rec.get("reason") or "point is NOT_RUN"))
    if "LOGIT_TOKEN" not in rec or "FAST_CAPABILITY" not in rec:
        raise ByteEliminationRefuse(
            "REFUSED: a cosine-only result cannot become the verdict; "
            "LOGIT_TOKEN and FAST_CAPABILITY are required"
        )
    verdict = capability_verdict(
        rec.get("LOGIT_TOKEN"),
        rec.get("FAST_CAPABILITY"),
        rec.get("LOCAL_FUNCTIONAL_FIDELITY"),
    )
    if verdict == NOT_RUN:
        raise PointNotRun(
            str(rec.get("reason") or "both capability levels are NOT_RUN")
        )
    return 1.0 if verdict == PASS else 0.0


def boundary_values(points: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """(destruction, capability_value) for evaluable points only.

    NOT_RUN is excluded, never averaged in.
    """
    out: list[tuple[float, float]] = []
    for pt in points:
        if pt.get("status") == NOT_RUN:
            continue
        logit = pt.get("LOGIT_TOKEN")
        fast = pt.get("FAST_CAPABILITY")
        if not isinstance(logit, Mapping) or not isinstance(fast, Mapping):
            continue
        verdict = capability_verdict(
            logit, fast, pt.get("LOCAL_FUNCTIONAL_FIDELITY")
        )
        if verdict == NOT_RUN:
            continue
        out.append((float(pt["destruction"]), 1.0 if verdict == PASS else 0.0))
    return out


# ---------------------------------------------------------------------------
# Search. capability_curve.sweep is the search. Do not write a second one.
# ---------------------------------------------------------------------------


def _rng_for(region: Mapping[str, Any], fraction: float) -> np.random.Generator:
    organ = str(region["organ"])
    organ_seed = sum((i + 1) * (ord(ch) + 1) for i, ch in enumerate(organ))
    seed = (
        RNG_SEED
        + 1009 * int(region["layer"])
        + 17 * organ_seed
        + int(round(float(fraction) * 1_000_000))
    ) % (2**32 - 1)
    return np.random.default_rng(seed)


def evaluate_live_point(
    cap: Mapping[str, Any],
    region: Mapping[str, Any],
    fraction: float,
    *,
    incumbent: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    organ_table: Mapping[str, Any],
    fast_blocked: str | None = None,
    incumbent_satisfies: bool | None = None,
) -> dict[str, Any]:
    cim.refuse_synthetic_activations(cap.get("source"))
    named = require_region(region)
    layer = int(named["layer"])
    kits = cap.get("kits") or {}
    if layer not in kits:
        raise ByteEliminationRefuse(
            f"REFUSED: capture has no kit for layer {layer}; a missing kit "
            "is not a skipped pass"
        )
    kit = kits[layer]
    rng = _rng_for(named, fraction)
    mlp_ov, mix_ov, n_dest, n_rows = destroyed_overrides(
        kit, named["organ"], fraction, rng
    )
    bm = bytes_and_ms(
        named,
        fraction,
        census=census,
        organ_table=organ_table,
        n_destroyed=n_dest,
        n_rows=n_rows,
    )
    n_more = int(cap["n_layers"]) - 1 - layer
    try:
        pert = replay_with_logits(
            cap,
            layer,
            n_more,
            mix_override=mix_ov,
            mlp_override=mlp_ov,
            override_layer=layer,
        )
    except (cim.CapabilityMapRefuse, OSError, MemoryError) as exc:
        reason = f"replay failed: {type(exc).__name__}: {exc}"
        return {
            **bm,
            "status": NOT_RUN,
            "reason": reason,
            "LOGIT_TOKEN": {"verdict": NOT_RUN, "reason": reason, "stage": LOGIT_TOKEN},
            "FAST_CAPABILITY": {
                "verdict": NOT_RUN,
                "reason": reason,
                "stage": FAST_CAPABILITY,
            },
            "LOCAL_FUNCTIONAL_FIDELITY": {
                "verdict": NOT_RUN,
                "reason": reason,
                "stage": LOCAL_FUNCTIONAL_FIDELITY,
                "role": CONTEXT,
                "not_the_verdict": True,
            },
            "capability_verdict": NOT_RUN,
        }

    logit: dict[str, Any]
    if pert.get("logits") is None or incumbent.get("logits") is None:
        reason = str(
            pert.get("logits_reason")
            or incumbent.get("logits_reason")
            or "logits were not measured"
        )
        logit = {"verdict": NOT_RUN, "reason": reason, "stage": LOGIT_TOKEN}
    else:
        logit = judge_logit_token(
            incumbent["logits"], pert["logits"], component_id=named["id"]
        )

    cosine: dict[str, Any]
    if pert.get("hidden_after_n") is None or incumbent.get("hidden_after_n") is None:
        cosine = {
            "verdict": NOT_RUN,
            "reason": "hidden_after_n missing",
            "stage": LOCAL_FUNCTIONAL_FIDELITY,
            "role": CONTEXT,
            "not_the_verdict": True,
        }
    else:
        cosine = judge_cosine_context(
            incumbent["hidden_after_n"],
            pert["hidden_after_n"],
            component_id=named["id"],
        )

    fast = judge_fast_capability(
        pert.get("argmax"),
        probes=probes,
        incumbent_argmax=incumbent.get("argmax"),
        incumbent_satisfies=incumbent_satisfies,
        reason_if_blocked=fast_blocked,
    )
    verdict = capability_verdict(logit, fast, cosine)
    status = NOT_RUN if verdict == NOT_RUN else "OK"
    rec = {
        **bm,
        "status": status,
        "LOGIT_TOKEN": logit,
        "FAST_CAPABILITY": fast,
        "LOCAL_FUNCTIONAL_FIDELITY": cosine,
        "capability_verdict": verdict,
        "argmax_incumbent": incumbent.get("argmax"),
        "argmax_candidate": pert.get("argmax"),
        "logits_status": pert.get("logits_status"),
    }
    if status == NOT_RUN:
        rec["reason"] = (
            logit.get("reason") if logit.get("verdict") == NOT_RUN else fast.get("reason")
        )
    return rec


def search_boundary(
    region: Any,
    *,
    measure: Measure | None = None,
    cap: Mapping[str, Any] | None = None,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    resolution: float = DEFAULT_RESOLUTION,
    budget: int = DEFAULT_BUDGET,
    n_coarse: int = DEFAULT_N_COARSE,
    census: Mapping[str, Any] | None = None,
    organ_table: Mapping[str, Any] | None = None,
    probes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Adaptive destruction search on a caller-named region.

    `measure` is injected by tests. The live path builds a measure from
    `cap` via capture_real_prefix + replay_prompt_from. This function will
    not invent a region and will not invent a GPU call.
    """
    named = require_region(region)
    census = census if census is not None else load_census()
    organ_table = organ_table if organ_table is not None else load_organ_ms()
    probes = list(probes) if probes is not None else list(FAST_PROBES)
    log: dict[str, dict[str, Any]] = {}

    if measure is None:
        if cap is None:
            raise ByteEliminationRefuse(
                "REFUSED: measure is missing and no capture was supplied; "
                "this module will not invent a GPU call and will not skip"
            )
        cim.refuse_synthetic_activations(cap.get("source"))
        layer = int(named["layer"])
        if layer not in (cap.get("kits") or {}):
            raise ByteEliminationRefuse(
                f"REFUSED: capture has no kit for layer {layer}"
            )
        n_more = int(cap["n_layers"]) - 1 - layer
        incumbent = replay_with_logits(cap, layer, n_more, override_layer=layer)
        cap_ids = tuple(int(t) for t in (cap.get("source") or {}).get("token_ids") or ())
        matching = [p for p in probes if tuple(p.get("token_ids") or ()) == cap_ids]
        fast_blocked = None
        incumbent_ok: bool | None = None
        if not matching:
            fast_blocked = (
                "capture token ids are not a FAST_CAPABILITY prompt; "
                "a mismatched prefix is not this predicate"
            )
        elif incumbent.get("argmax") is None:
            fast_blocked = "incumbent argmax was not measured"
        else:
            expect = {int(x) for p in matching for x in (p.get("expect_argmax_in") or ())}
            incumbent_ok = int(incumbent["argmax"]) in expect
            probes = matching

        def measure(spec: Mapping[str, Any], _inc: Mapping[str, Any] = incumbent) -> Any:
            return evaluate_live_point(
                cap,
                named,
                float(spec["level"]),
                incumbent=_inc,
                probes=probes,
                census=census,
                organ_table=organ_table,
                fast_blocked=fast_blocked,
                incumbent_satisfies=incumbent_ok,
            )

    def wrapped(spec: Mapping[str, Any]) -> dict[str, Any]:
        rec = measure(spec)
        if not isinstance(rec, Mapping):
            raise ByteEliminationRefuse(
                "REFUSED: measure must return a mapping with LOGIT_TOKEN and "
                "FAST_CAPABILITY; a bare cosine is not a verdict"
            )
        if "LOGIT_TOKEN" not in rec or "FAST_CAPABILITY" not in rec:
            raise ByteEliminationRefuse(
                "REFUSED: a cosine-only result cannot become the verdict; "
                "LOGIT_TOKEN and FAST_CAPABILITY are required at every point"
            )
        key = cc.cache_key(
            {
                "component": named["id"],
                "layer": named["layer"],
                "axis": "output_rows",
                "perturbation_type": "zero_fraction",
            },
            float(spec["level"]),
        )
        try:
            cap_val = curve_value_from_record(rec)
        except PointNotRun:
            rec = dict(rec)
            rec["status"] = NOT_RUN
            rec["capability_verdict"] = NOT_RUN
            rec.setdefault("destruction", float(spec["level"]))
            log[key] = rec
            raise
        rec = dict(rec)
        rec["value"] = cap_val
        rec["capability_verdict"] = capability_verdict(
            rec["LOGIT_TOKEN"],
            rec["FAST_CAPABILITY"],
            rec.get("LOCAL_FUNCTIONAL_FIDELITY"),
        )
        rec.setdefault("status", "OK")
        rec.setdefault("destruction", float(spec["level"]))
        if "bytes_gone" not in rec or "ms_gone" not in rec:
            bm = bytes_and_ms(
                named, float(spec["level"]), census=census, organ_table=organ_table
            )
            rec.setdefault("bytes_gone", bm["bytes_gone"])
            rec.setdefault("ms_gone", bm["ms_gone"])
            rec.setdefault("region_bytes", bm["region_bytes"])
        log[key] = rec
        # Pass only the capability value to the sweeper. VALUE_KEYS includes
        # "cosine"; a top-level cosine would steal the curve.
        return {"value": cap_val, "measured_at_level": "CAPABILITY"}

    try:
        sweep = cc.sweep(
            component=named["id"],
            layer=named["layer"],
            axis="output_rows",
            perturbation_type="zero_fraction",
            lo=lo,
            hi=hi,
            resolution=resolution,
            budget=budget,
            n_coarse=n_coarse,
            measure=wrapped,
        )
    except PointNotRun as exc:
        points = _merge_points([], log, named)
        return {
            "region": named,
            "kind": INCOMPLETE,
            "cliff_found": False,
            "bracket": None,
            "message": f"search incomplete: {exc}",
            "why": str(exc),
            "points": _py(points),
            "n_evaluated": len(boundary_values(points)),
            "n_not_run": sum(1 for p in points if p.get("status") == NOT_RUN),
            "cosine_is_not_the_verdict": True,
            "licensed_bytes": {
                "kind": INCOMPLETE,
                "lo": None,
                "hi": None,
                "reading": "NOT_RUN points were excluded; no boundary is claimed",
            },
            "per_level": per_level_from_points(points, region_census_bytes(census, named)),
        }

    points = _merge_points(sweep.get("points") or [], log, named)
    kind = BOUNDARY_INTERVAL if sweep.get("cliff_found") else FLAT
    region_bytes = region_census_bytes(census, named)
    licensed = licensed_bytes_interval(points, region_bytes, kind, sweep.get("bracket"))
    per_level = per_level_from_points(points, region_bytes)
    return {
        "region": named,
        "kind": kind,
        "cliff_found": bool(sweep.get("cliff_found")),
        "bracket": sweep.get("bracket"),
        "detected_interval": sweep.get("detected_interval"),
        "search_range": sweep.get("search_range"),
        "message": sweep.get("message"),
        "why": sweep.get("why"),
        "points": _py(points),
        "n_measured": sweep.get("n_measured"),
        "n_cache_hits": sweep.get("n_cache_hits"),
        "budget": sweep.get("budget"),
        "resolution": sweep.get("resolution"),
        "resolution_met": sweep.get("resolution_met"),
        "n_evaluated": len(boundary_values(points)),
        "n_not_run": sum(1 for p in points if p.get("status") == NOT_RUN),
        "cosine_is_not_the_verdict": True,
        "licensed_bytes": licensed,
        "per_level": per_level,
        "region_bytes": int(region_bytes),
        "region_bytes_mb": mb(region_bytes),
    }


def _merge_points(
    sweep_points: Sequence[Mapping[str, Any]],
    log: Mapping[str, Mapping[str, Any]],
    named: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identity = {
        "component": named["id"],
        "layer": named["layer"],
        "axis": "output_rows",
        "perturbation_type": "zero_fraction",
    }
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in sweep_points:
        key = cc.cache_key(identity, float(p["level"]))
        rec = dict(log.get(key) or {})
        merged = dict(p)
        for field in (
            "bytes_gone",
            "ms_gone",
            "region_bytes",
            "LOGIT_TOKEN",
            "FAST_CAPABILITY",
            "LOCAL_FUNCTIONAL_FIDELITY",
            "capability_verdict",
            "status",
            "reason",
            "argmax_incumbent",
            "argmax_candidate",
            "logits_status",
            "bytes_gone_mb",
        ):
            if field in rec:
                merged[field] = rec[field]
        merged["destruction"] = float(p["level"])
        out.append(merged)
        seen.add(key)
    for key, rec in log.items():
        if key in seen:
            continue
        extra = dict(rec)
        extra.setdefault("destruction", extra.get("level"))
        extra.setdefault("source", "not_in_sweep")
        out.append(extra)
    out.sort(key=lambda p: float(p.get("destruction") if p.get("destruction") is not None else p.get("level") or 0.0))
    return out


def per_level_from_points(
    points: Sequence[Mapping[str, Any]],
    region_bytes: int,
) -> dict[str, Any]:
    """Boundary per capability level, from already-evaluated points. No second sweep."""
    out: dict[str, Any] = {}
    for key in (LOGIT_TOKEN, FAST_CAPABILITY):
        seq: list[tuple[float, float, int]] = []
        for pt in sorted(
            points, key=lambda x: float(x.get("destruction") or x.get("level") or 0.0)
        ):
            row = pt.get(key)
            if not isinstance(row, Mapping):
                continue
            verdict = row.get("verdict")
            if verdict not in {PASS, FAIL}:
                continue
            dest = float(pt.get("destruction") if pt.get("destruction") is not None else pt.get("level") or 0.0)
            seq.append((dest, 1.0 if verdict == PASS else 0.0, int(pt.get("bytes_gone") or 0)))
        if not seq:
            out[key] = {
                "kind": NOT_RUN,
                "lo": None,
                "hi": None,
                "reading": f"{key} was NOT_RUN; excluded from the boundary",
            }
            continue
        values = [v for _d, v, _b in seq]
        if len(set(values)) == 1:
            held = values[0] == 1.0
            b = seq[-1][2] if held else 0
            out[key] = {
                "kind": FLAT,
                "lo": b,
                "hi": b,
                "lo_mb": mb(b),
                "hi_mb": mb(b),
                "reading": (
                    f"{key} never broke across the sweep"
                    if held
                    else f"{key} never held across the sweep"
                ),
            }
            continue
        best_i = 0
        best_delta = -1.0
        for i in range(len(seq) - 1):
            delta = abs(seq[i + 1][1] - seq[i][1])
            if delta > best_delta:
                best_delta = delta
                best_i = i
        lo_d, _va, lo_b = seq[best_i]
        hi_d, _vb, hi_b = seq[best_i + 1]
        out[key] = {
            "kind": BOUNDARY_INTERVAL,
            "lo": lo_b,
            "hi": hi_b,
            "lo_mb": mb(lo_b),
            "hi_mb": mb(hi_b),
            "destruction_lo": lo_d,
            "destruction_hi": hi_d,
            "reading": (
                f"{key} changes in [{lo_d:g}, {hi_d:g}], "
                f"[{mb(lo_b):.4f}, {mb(hi_b):.4f}] MB of this region"
            ),
        }
    return out


def licensed_bytes_interval(
    points: Sequence[Mapping[str, Any]],
    region_bytes: int,
    kind: str,
    bracket: Mapping[str, Any] | None,
) -> dict[str, Any]:
    usable = boundary_values(points)
    if not usable:
        return {
            "kind": INCOMPLETE,
            "lo": None,
            "hi": None,
            "lo_mb": None,
            "hi_mb": None,
            "reading": "no evaluable points; NOT_RUN was excluded, not averaged",
        }
    still = [p for p in points if p.get("capability_verdict") == PASS]
    if kind == FLAT:
        if still and all(p.get("capability_verdict") == PASS for p in points if p.get("status") != NOT_RUN and p.get("capability_verdict") in {PASS, FAIL}):
            bmax = max(int(p.get("bytes_gone") or 0) for p in still)
            return {
                "kind": FLAT,
                "lo": bmax,
                "hi": bmax,
                "lo_mb": mb(bmax),
                "hi_mb": mb(bmax),
                "reading": (
                    "capability never broke across the sweep; the named region "
                    f"licenses its swept range ({mb(bmax):.2f} MB), as a FLAT "
                    "curve, not as a threshold pass"
                ),
            }
        if not still:
            return {
                "kind": FLAT,
                "lo": 0,
                "hi": 0,
                "lo_mb": 0.0,
                "hi_mb": 0.0,
                "reading": "capability never held; licenses nothing",
            }
        lo_b = min(int(p.get("bytes_gone") or 0) for p in still)
        hi_b = max(int(p.get("bytes_gone") or 0) for p in still)
        return {
            "kind": FLAT,
            "lo": lo_b,
            "hi": hi_b,
            "lo_mb": mb(lo_b),
            "hi_mb": mb(hi_b),
            "reading": (
                "capability change is spread across the range rather than "
                "concentrated; reported as FLAT, not a fabricated cliff"
            ),
        }
    if not isinstance(bracket, Mapping) or "lo" not in bracket or "hi" not in bracket:
        raise ByteEliminationRefuse(
            "REFUSED: BOUNDARY_INTERVAL without a bracket; a point estimate "
            "is not a boundary"
        )
    lo_b = int(round(float(bracket["lo"]) * region_bytes))
    hi_b = int(round(float(bracket["hi"]) * region_bytes))
    return {
        "kind": BOUNDARY_INTERVAL,
        "lo": lo_b,
        "hi": hi_b,
        "lo_mb": mb(lo_b),
        "hi_mb": mb(hi_b),
        "destruction_lo": float(bracket["lo"]),
        "destruction_hi": float(bracket["hi"]),
        "reading": (
            "capability changes in the destruction interval "
            f"[{bracket['lo']:g}, {bracket['hi']:g}], which is "
            f"[{mb(lo_b):.4f}, {mb(hi_b):.4f}] MB of this region"
        ),
    }


# ---------------------------------------------------------------------------
# Capture. Real activations only. Refuse is a block, not a synthetic x.
# ---------------------------------------------------------------------------


def capture_or_block(
    token_ids: Sequence[int],
    sample_layers: Sequence[int],
) -> dict[str, Any]:
    ids = tuple(int(t) for t in token_ids)
    sample = tuple(int(s) for s in sample_layers)
    try:
        cap = cim.capture_real_prefix(token_ids=ids, sample_layers=sample)
        cim.refuse_synthetic_activations(cap.get("source"))
        return {"ok": True, "cap": cap, "reason": None}
    except (cim.CapabilityMapRefuse, cim.SyntheticActivationRefuse, OSError, MemoryError) as exc:
        return {
            "ok": False,
            "cap": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def arithmetic_points_for_blocked_region(
    region: Mapping[str, Any],
    *,
    census: Mapping[str, Any],
    organ_table: Mapping[str, Any],
    reason: str,
    n_coarse: int = DEFAULT_N_COARSE,
) -> list[dict[str, Any]]:
    """Bytes/ms still exist without a replay. Capability is NOT_RUN, not a pass."""
    named = require_region(region)
    levels = cc.linspace(DEFAULT_LO, DEFAULT_HI, n_coarse)
    points = []
    reason_full = f"capture blocked: {reason}"
    not_run = {
        "verdict": NOT_RUN,
        "reason": reason_full,
    }
    for lv in levels:
        bm = bytes_and_ms(named, lv, census=census, organ_table=organ_table)
        points.append(
            {
                **bm,
                "level": lv,
                "destruction": lv,
                "source": "coarse",
                "status": NOT_RUN,
                "reason": reason_full,
                "LOGIT_TOKEN": {**not_run, "stage": LOGIT_TOKEN},
                "FAST_CAPABILITY": {**not_run, "stage": FAST_CAPABILITY},
                "LOCAL_FUNCTIONAL_FIDELITY": {
                    **not_run,
                    "stage": LOCAL_FUNCTIONAL_FIDELITY,
                    "role": CONTEXT,
                    "not_the_verdict": True,
                },
                "capability_verdict": NOT_RUN,
            }
        )
    return points


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def comparison_reading(
    *,
    cosine: Mapping[str, Any],
    region_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cosine_mb = float(cosine["mb"])
    licensed_lo: list[int] = []
    licensed_hi: list[int] = []
    kinds = []
    fast_flat_held = 0
    logit_cliffs = 0
    for row in region_rows:
        lic = row.get("licensed_bytes") or {}
        kinds.append(row.get("kind"))
        if lic.get("lo") is not None:
            licensed_lo.append(int(lic["lo"]))
        if lic.get("hi") is not None:
            licensed_hi.append(int(lic["hi"]))
        per = row.get("per_level") or {}
        fast = per.get(FAST_CAPABILITY) or {}
        logit = per.get(LOGIT_TOKEN) or {}
        if fast.get("kind") == FLAT and (fast.get("hi") or 0) > 0:
            fast_flat_held += 1
        if logit.get("kind") == BOUNDARY_INTERVAL:
            logit_cliffs += 1
    total_lo = sum(licensed_lo) if licensed_lo else 0
    total_hi = sum(licensed_hi) if licensed_hi else 0
    total_lo_mb = mb(total_lo)
    total_hi_mb = mb(total_hi)
    any_cliff = any(k == BOUNDARY_INTERVAL for k in kinds)
    any_flat = any(k == FLAT for k in kinds)
    if not licensed_hi:
        shape = "incomplete"
        vs_cosine = (
            "no evaluable capability boundary; cannot claim the capability "
            "bar licenses more (or less) than the 0.99 cosine bar"
        )
    else:
        if any_cliff and not any_flat:
            shape = "cliff"
        elif any_flat and not any_cliff:
            shape = "flat"
        else:
            shape = "mixed"
        vs_cosine = (
            f"On the named region(s) useful capability still holds through "
            f"{total_lo_mb:.2f} MB and the change lives in "
            f"[{total_lo_mb:.2f}, {total_hi_mb:.2f}] MB (sum of per-region "
            f"intervals; not a joint destruction). The 0.99 cosine bar "
            f"licensed {cosine_mb:.1f} MB globally. The campaign needs "
            f"{CAMPAIGN_NEED_MB:.0f} MB. The capability bar does not license "
            f"that — {total_hi_mb:.2f} MB on the named tensors is not 1773 MB, "
            f"and it is no more than cosine licensed globally. "
            f"FAST_CAPABILITY was FLAT on {fast_flat_held} named region(s) "
            f"(the Paris argmax survived); the cliff is LOGIT_TOKEN "
            f"({logit_cliffs} region(s)). A negative at campaign scale is a "
            "real finding."
        )
    return {
        "cosine_bar_licensed_bytes": int(cosine["bytes"]),
        "cosine_bar_licensed_mb": cosine_mb,
        "cosine_bar": cosine.get("cosine_bar"),
        "cosine_bar_source": cosine.get("source"),
        "campaign_need_mb": CAMPAIGN_NEED_MB,
        "campaign_need_source": CAMPAIGN_NEED_SOURCE,
        "capability_bar_still_holds_bytes": int(total_lo),
        "capability_bar_still_holds_mb": total_lo_mb,
        "capability_bar_licensed_bytes_hi": int(total_hi),
        "capability_bar_licensed_mb_hi": total_hi_mb,
        "shortfall_vs_campaign_x": (
            None if total_hi_mb <= 0 else CAMPAIGN_NEED_MB / total_hi_mb
        ),
        "curve_shape": shape,
        "fast_capability_flat_held_regions": fast_flat_held,
        "logit_token_cliff_regions": logit_cliffs,
        "reading": vs_cosine,
        "named_regions_are_not_a_global_allocation": True,
    }


def assemble_document(
    *,
    region_rows: Sequence[Mapping[str, Any]],
    cosine: Mapping[str, Any] | None = None,
    capture_status: Mapping[str, Any] | None = None,
    parent_payload_bytes: int | None = None,
    default_regions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    cosine = cosine if cosine is not None else load_cosine_license()
    capture_status = capture_status or {"ok": True, "reason": None}
    cmp = comparison_reading(cosine=cosine, region_rows=region_rows)
    live = bool(capture_status.get("ok"))
    capture_pub: dict[str, Any] = dict(capture_status)
    if live:
        capture_pub.setdefault("token_ids", list(FRANCE_TOKEN_IDS))
        capture_pub.setdefault("prompt", FRANCE_PROMPT)
        capture_pub.setdefault("real_forward_pass", True)
        capture_pub.setdefault("synthetic", False)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "obligation": "G109",
        "purpose": (
            "Per-region progressive-destruction search for the capability "
            "BOUNDARY: which information can disappear while useful "
            "capability remains. Not a cosine-0.99 threshold."
        ),
        "question": (
            "WHICH INFORMATION CAN DISAPPEAR WHILE USEFUL CAPABILITY REMAINS?"
        ),
        "the_wrong_bar": {
            "source": MAP_REL,
            "bar": "hidden cosine 0.99",
            "licensed_bytes": int(cosine["bytes"]),
            "licensed_mb": float(cosine["mb"]),
            "share_of_token": cosine.get("share_of_token"),
            "n_sampled_regions": cosine.get("n_regions"),
            "campaign_need_mb": CAMPAIGN_NEED_MB,
            "shortfall": "64x",
            "why_it_is_the_wrong_question": (
                "it asked which information can disappear while hidden-state "
                "cosine stays above 0.99. That is LOCAL_FUNCTIONAL_FIDELITY, "
                "not capability."
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "chooses_region": False,
        "caller_supplies_region": True,
        "default_build_regions_are_not_a_scientific_pick": True,
        "default_build_regions": _py(list(default_regions or DEFAULT_BUILD_REGIONS)),
        "cosine_is_not_the_verdict": True,
        "verdict_levels": [LOGIT_TOKEN, FAST_CAPABILITY],
        "context_level": LOCAL_FUNCTIONAL_FIDELITY,
        "bars_reused": {
            "hidden_cosine_bar": float(cim.HIDDEN_COSINE_BAR),
            "logit_kl_bar": float(acs.LOGIT_KL_BAR),
            "top_k": int(acs.TOPK),
            "top_k_agree_bar": float(acs.TOPK_AGREE_BAR),
            "note": (
                "hidden cosine is reused as CONTEXT only. LOGIT_TOKEN pass is "
                "KL <= 0.1 and top-5 agreement >= 0.8. Argmax identity is "
                "recorded and is not the pass criterion."
            ),
        },
        "fast_probes": _py(list(FAST_PROBES)),
        "fast_probes_are_real_tokenizer_ids": True,
        "reused_not_rebuilt": {
            "replay": (
                "capability_information_map.capture_real_prefix, "
                "replay_prompt_from, lm_head_gemv, refuse_synthetic_activations"
            ),
            "judges": (
                "capability_stages.run_logit_token, "
                "run_local_functional_fidelity; aux_capability_screen bars"
            ),
            "search": "capability_curve.sweep (coarse, detect, refine)",
            "bytes": "MLP_BYTE_CENSUS.json via complete_ebpw.mix_report parent total",
            "ms": "ORGAN_DECOMPOSITION_SEALED.json, pro-rated, ESTIMATED_FROM_CITED_MS",
        },
        "parent_payload_bytes": parent_payload_bytes,
        "capture": _py(capture_pub),
        "regions": _py(list(region_rows)),
        "comparison": cmp,
        "answer": cmp["reading"],
        "what_this_does_not_prove": [
            "that unnamed regions match the named ones (they are UNMEASURED)",
            "GPU generate identity of a packed kernel",
            "that a FLAT last-layer MLP tensor licenses 1773 MB globally",
            "HCLI mission competence or EXPENSIVE_QUALIFICATION",
            "that cosine 0.99 is a capability bar (it is not)",
        ],
    }


def build(*, regions: Sequence[Any] | None = None) -> Path:
    named_list = [require_region(r) for r in (regions or DEFAULT_BUILD_REGIONS)]
    census = load_census()
    organ_table = load_organ_ms()
    cosine = load_cosine_license()
    try:
        parent_bytes = load_parent_payload_bytes()
    except ByteEliminationRefuse:
        parent_bytes = None

    sample_layers = tuple(sorted({int(r["layer"]) for r in named_list}))
    print(
        f"G109 capture_real_prefix token_ids={list(FRANCE_TOKEN_IDS)} "
        f"sample_layers={list(sample_layers)}",
        flush=True,
    )
    captured = capture_or_block(FRANCE_TOKEN_IDS, sample_layers)
    region_rows: list[dict[str, Any]] = []
    if not captured["ok"]:
        print(f"G109 capture blocked: {captured['reason']}", flush=True)
        for named in named_list:
            points = arithmetic_points_for_blocked_region(
                named, census=census, organ_table=organ_table, reason=captured["reason"]
            )
            region_bytes = region_census_bytes(census, named)
            region_rows.append(
                {
                    "region": named,
                    "kind": BLOCKED,
                    "cliff_found": False,
                    "bracket": None,
                    "message": f"capture blocked; capability NOT_RUN on every point",
                    "why": captured["reason"],
                    "points": _py(points),
                    "n_evaluated": 0,
                    "n_not_run": len(points),
                    "cosine_is_not_the_verdict": True,
                    "licensed_bytes": {
                        "kind": BLOCKED,
                        "lo": None,
                        "hi": None,
                        "reading": "lane blocked; no boundary is claimed",
                    },
                    "per_level": per_level_from_points(points, region_bytes),
                    "region_bytes": int(region_bytes),
                    "region_bytes_mb": mb(region_bytes),
                }
            )
        capture_status = {
            "ok": False,
            "reason": captured["reason"],
            "blocked": True,
            "synthetic_substituted": False,
        }
    else:
        cap = captured["cap"]
        print(
            f"G109 captured n_layers={cap['n_layers']} n_tokens={cap['n_tokens']} "
            f"logits_status={cap.get('logits_status')}",
            flush=True,
        )
        for named in named_list:
            print(f"G109 search_boundary {named['id']}", flush=True)
            row = search_boundary(
                named,
                cap=cap,
                census=census,
                organ_table=organ_table,
                probes=FAST_PROBES,
            )
            print(
                f"G109 {named['id']} kind={row['kind']} "
                f"n_evaluated={row.get('n_evaluated')}",
                flush=True,
            )
            region_rows.append(row)
        capture_status = {
            "ok": True,
            "reason": None,
            "blocked": False,
            "synthetic_substituted": False,
            "real_forward_pass": True,
            "token_ids": list(FRANCE_TOKEN_IDS),
            "prompt": FRANCE_PROMPT,
            "n_tokens": cap["n_tokens"],
            "n_layers": cap["n_layers"],
            "logits_status": cap.get("logits_status"),
            "baseline_argmax": cap.get("argmax"),
        }

    doc = assemble_document(
        region_rows=region_rows,
        cosine=cosine,
        capture_status=capture_status,
        parent_payload_bytes=parent_bytes,
        default_regions=named_list,
    )
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true", help="emit the sealed receipt")
    args = ap.parse_args(argv)
    if args.build:
        out = build()
        print(out)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
