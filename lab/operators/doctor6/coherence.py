"""Coherence pre-pack gate: necessary-condition composition screen.

Empirical ground truth (this campaign, 4-point sweep):
  c ≈ 0.987  coherent text
  c ≈ 0.925  incoherent
  c ≈ 0.834  incoherent
  c ≈ 0.685  incoherent (source-gated-v2)

Bound: for L residual-stream layers, if each layer's residual addition agrees
with the parent at cosine c_l, a *necessary* condition for end-to-end directional
agreement above 0.5 is

    product_l c_l  >=  0.5

With uniform c this is c^L >= 0.5, i.e. c >= 0.5^(1/L). For L=48:
    0.5^(1/48) ≈ 0.9857.

Residual connections mean the real map is not purely multiplicative (the residual
stream carries identity), so c^L is a NECESSARY-condition screen: high recall for
failure, not a sufficient guarantee of coherence. Correct asymmetry for a gate
that saves ~77 minutes of repack+generate: never pass something that will fail;
may occasionally fail something that would have worked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

N_LAYERS = 48
COMPOSITION_FLOOR = 0.5
LAYER_TARGET_COS = COMPOSITION_FLOOR ** (1.0 / N_LAYERS)  # ≈ 0.9857

# Layer bands used throughout the Q30 campaign (12 layers each).
BANDS: tuple[tuple[int, int], ...] = ((0, 11), (12, 23), (24, 35), (36, 47))


@dataclass(frozen=True)
class CoherenceVerdict:
    pass_gate: bool
    predicted_composition: float
    composition_floor: float
    layer_target: float
    n_layers: int
    aggregation: str
    per_layer_cos: list[float]
    per_band_mean: dict[str, float]
    per_band_deficit: dict[str, float]
    mean_organ_cos: float
    geometric_mean_organ_cos: float
    uniform_proxy_composition: float  # mean_organ_cos ** n_layers
    refusal_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "hawking.doctor6.coherence_screen.v1",
            "pass_gate": self.pass_gate,
            "predicted_composition": self.predicted_composition,
            "composition_floor": self.composition_floor,
            "layer_target": self.layer_target,
            "n_layers": self.n_layers,
            "aggregation": self.aggregation,
            "per_layer_cos": self.per_layer_cos,
            "per_band_mean": self.per_band_mean,
            "per_band_deficit": self.per_band_deficit,
            "mean_organ_cos": self.mean_organ_cos,
            "geometric_mean_organ_cos": self.geometric_mean_organ_cos,
            "uniform_proxy_composition": self.uniform_proxy_composition,
            "refusal_reason": self.refusal_reason,
            "notes": list(self.notes),
            "posture": (
                "necessary_condition_high_recall: residual connections mean "
                "composition is not literally multiplicative; refuse if product "
                f"of layer cosines < {self.composition_floor}; false negatives "
                "(refuse something that would work) are preferred to false "
                "positives (pack something incoherent)."
            ),
        }


def _safe_mean(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _safe_gmean(vals: Sequence[float]) -> float:
    clean = [max(1e-12, min(1.0, float(v))) for v in vals if v is not None and math.isfinite(float(v))]
    if not clean:
        return float("nan")
    # geometric mean via logs for stability
    return float(math.exp(sum(math.log(v) for v in clean) / len(clean)))


def _band_name(layer: int) -> str:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= layer <= hi:
            return f"band_{i}"
    return "band_other"


def _extract_layer_cosines(
    organs: Sequence[Mapping[str, Any]] | Sequence[float],
    *,
    n_layers: int,
) -> tuple[list[list[float]], list[float], str]:
    """Return (per_layer_lists, flat_cosines, aggregation_description)."""

    if not organs:
        return [[] for _ in range(n_layers)], [], "empty"

    # Flat list of floats: treat as a homogeneous population; each layer gets the mean.
    if isinstance(organs[0], (int, float)):
        flat = [float(c) for c in organs]  # type: ignore[arg-type]
        mean_c = _safe_mean(flat)
        per_layer = [[mean_c] for _ in range(n_layers)]
        return (
            per_layer,
            flat,
            (
                "homogeneous_mean_fill: input was a flat cosine list; each of the "
                f"{n_layers} layers is assigned the arithmetic mean of the list, "
                "then composition = product over layers. Equivalent to mean^n when "
                "the population is treated as layer-uniform (the acceptance ground truth)."
            ),
        )

    # Structured organ records: expect keys cosine/output_cosine/c and optional layer.
    flat: list[float] = []
    per_layer: list[list[float]] = [[] for _ in range(n_layers)]
    for row in organs:  # type: ignore[assignment]
        if not isinstance(row, Mapping):
            continue
        c = row.get("output_cosine", row.get("cosine", row.get("c", row.get("mean_cos"))))
        if c is None:
            continue
        c_f = float(c)
        flat.append(c_f)
        layer = row.get("layer", row.get("L"))
        if layer is None:
            # No layer tag: deferred to fill below.
            continue
        li = int(layer)
        if 0 <= li < n_layers:
            per_layer[li].append(c_f)

    observed = [i for i, xs in enumerate(per_layer) if xs]
    if not observed:
        # No layer tags at all: homogeneous fill from organ mean.
        mean_c = _safe_mean(flat)
        per_layer = [[mean_c] for _ in range(n_layers)]
        return (
            per_layer,
            flat,
            (
                "homogeneous_mean_fill: organ records lacked layer tags; each layer "
                f"gets arithmetic mean of organ cosines; composition = mean^{n_layers}."
            ),
        )

    # Fill unobserved layers with the arithmetic mean of *observed layer means*
    # (not the organ mean — preserves band structure when the sample is stratified).
    layer_means_obs = [_safe_mean(per_layer[i]) for i in observed]
    fill = _safe_mean(layer_means_obs)
    for i in range(n_layers):
        if not per_layer[i]:
            per_layer[i] = [fill]
    return (
        per_layer,
        flat,
        (
            "layer_wise_arithmetic_mean: for each layer, c_l = mean of organ "
            "cosines observed in that layer (gate/up/down and experts in the "
            "sample). Unobserved layers filled with the mean of observed layer "
            "means. predicted_composition = product_l c_l. "
            "Why mean within layer: a MoE residual addition is a routing-weighted "
            "sum of expert outputs; with unknown route mass on a holdout organ "
            "sample the arithmetic mean is an unbiased proxy. Why product across "
            "layers: under a pure residual-addition composition (no residual "
            "identity help) directional agreement multiplies; residual identity "
            "only makes the bound conservative (necessary, not sufficient)."
        ),
    )


def predict_composition(
    organs: Sequence[Mapping[str, Any]] | Sequence[float],
    *,
    n_layers: int = N_LAYERS,
    composition_floor: float = COMPOSITION_FLOOR,
) -> CoherenceVerdict:
    """Predict end-to-end composition cosine and decide the pre-pack gate."""

    per_layer_lists, flat, agg = _extract_layer_cosines(organs, n_layers=n_layers)
    per_layer_cos = [_safe_mean(xs) for xs in per_layer_lists]
    # Clamp into (0, 1] for the product; a non-positive cosine is already fatal.
    clamped = [max(1e-12, min(1.0, c)) if math.isfinite(c) else 1e-12 for c in per_layer_cos]
    log_sum = sum(math.log(c) for c in clamped)
    predicted = float(math.exp(log_sum))

    mean_organ = _safe_mean(flat) if flat else _safe_mean(per_layer_cos)
    gmean_organ = _safe_gmean(flat) if flat else _safe_gmean(per_layer_cos)
    uniform_proxy = float(mean_organ ** n_layers) if math.isfinite(mean_organ) else 0.0

    per_band_mean: dict[str, float] = {}
    per_band_deficit: dict[str, float] = {}
    layer_target = composition_floor ** (1.0 / n_layers)
    for bi, (lo, hi) in enumerate(BANDS):
        name = f"band_{bi}"
        band_vals = [per_layer_cos[i] for i in range(lo, min(hi + 1, n_layers))]
        m = _safe_mean(band_vals)
        per_band_mean[name] = m
        per_band_deficit[name] = float(layer_target - m) if math.isfinite(m) else float("nan")

    pass_gate = bool(predicted >= composition_floor)
    refusal = None
    if not pass_gate:
        worst_band = max(per_band_deficit, key=lambda k: per_band_deficit[k])
        refusal = (
            f"predicted_composition={predicted:.6e} < floor={composition_floor} "
            f"(layer_target={layer_target:.6f}, mean_organ={mean_organ:.4f}, "
            f"worst_band={worst_band} deficit={per_band_deficit[worst_band]:+.4f}); "
            "REFUSE pack — necessary-condition screen"
        )

    notes = [
        f"layer_target = floor^(1/n_layers) = {composition_floor}^{1}/{n_layers} = {layer_target:.6f}",
        f"uniform_proxy (mean^n) = {uniform_proxy:.6e}",
        "posture: false-negative preferred (may refuse a survivor; must not pass a failure)",
    ]
    return CoherenceVerdict(
        pass_gate=pass_gate,
        predicted_composition=predicted,
        composition_floor=composition_floor,
        layer_target=layer_target,
        n_layers=n_layers,
        aggregation=agg,
        per_layer_cos=per_layer_cos,
        per_band_mean=per_band_mean,
        per_band_deficit=per_band_deficit,
        mean_organ_cos=float(mean_organ) if math.isfinite(mean_organ) else float("nan"),
        geometric_mean_organ_cos=float(gmean_organ) if math.isfinite(gmean_organ) else float("nan"),
        uniform_proxy_composition=uniform_proxy,
        refusal_reason=refusal,
        notes=notes,
    )


def screen_organ_cosines(
    cosines: Iterable[float],
    *,
    n_layers: int = N_LAYERS,
    composition_floor: float = COMPOSITION_FLOOR,
) -> CoherenceVerdict:
    """Convenience: screen a flat iterable of organ cosines (e.g. mean 0.685 / 0.987)."""
    return predict_composition(list(cosines), n_layers=n_layers, composition_floor=composition_floor)
