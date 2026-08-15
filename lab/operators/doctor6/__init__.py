"""Doctor v6 — objective-driven, per-organ adaptive, predictive.

Three verbs:

  prescribe  diagnose + compose cheapest-sufficient chains + predict + refuse/emit
  treat      execute a prescription, re-measure, seal under the ceiling
  verify     c^48 coherence screen (+ optional generation probe)

v5 was a registry of rungs a human composed. v6 is an optimizer with an
objective that composes itself, adapts per organ, and refuses to emit
something that cannot work under the necessary-condition coherence screen.
"""
from __future__ import annotations

from lab.operators.doctor6.coherence import (
    COMPOSITION_FLOOR,
    LAYER_TARGET_COS,
    N_LAYERS,
    CoherenceVerdict,
    predict_composition,
    screen_organ_cosines,
)
from lab.operators.doctor6.prescribe import prescribe
from lab.operators.doctor6.treat import treat
from lab.operators.doctor6.verify import verify

__all__ = [
    "COMPOSITION_FLOOR",
    "LAYER_TARGET_COS",
    "N_LAYERS",
    "CoherenceVerdict",
    "predict_composition",
    "screen_organ_cosines",
    "prescribe",
    "treat",
    "verify",
]
