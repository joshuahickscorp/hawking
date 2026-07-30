"""Hawking laboratory — Core C experiment engine + science operators.

Process lifecycle for campaigns and the single operator registry. Science
bodies live under lab/operators/; tools/condense retains only the S5 client
and the launchd-anchored source-fetch CLI.
"""
from __future__ import annotations

__version__ = "1.1.0"
AUTHORITY = "lab"
SCHEMA = "hawking.lab.core.v1"

from lab.runtime import ExperimentRuntime, RunResult, run_experiment
from lab.receipts import Receipt, ReceiptAuthority, seal, verify
from lab.spec import ExperimentSpec, SpecError, load_spec, validate_spec

__all__ = [
    "AUTHORITY",
    "ExperimentRuntime",
    "ExperimentSpec",
    "Receipt",
    "ReceiptAuthority",
    "RunResult",
    "SCHEMA",
    "SpecError",
    "load_spec",
    "run_experiment",
    "seal",
    "validate_spec",
    "verify",
]
