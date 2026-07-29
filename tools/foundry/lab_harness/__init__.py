"""Single laboratory harness authority for bench, training, strand, foundry.

One declarative experiment spec, one runner, one measurement recorder,
one receipt writer, one report renderer. Domain trees supply specs and
step bodies; they do not re-implement the harness.
"""
from __future__ import annotations

from .measure import MeasurementRecorder
from .receipt import ReceiptWriter
from .report import ReportRenderer
from .runner import Runner
from .spec import ExperimentSpec, load_spec, validate_spec

SCHEMA = "hawking.lab.harness.v1"
HARNESS_VERSION = "1.0.0"

__all__ = [
    "SCHEMA",
    "HARNESS_VERSION",
    "ExperimentSpec",
    "load_spec",
    "validate_spec",
    "Runner",
    "MeasurementRecorder",
    "ReceiptWriter",
    "ReportRenderer",
]
