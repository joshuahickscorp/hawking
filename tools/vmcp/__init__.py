"""Hawking-side VMCP organs that run without the foreign visionmcp package.

The compact E.14 model-facing surface is still the nine acts
(see/hold/open/know/make/check/fix/keep/prove) on
``tools.future.vmcp.compact_surface``. This package is the implementation
those verbs call for local file classification, tool receipts, the
behavior-lab fixture matrix, and an attempted PTY capture.

Do not add a verb per organ.
"""
from __future__ import annotations

from tools.vmcp.behavior_lab import run_matrix
from tools.vmcp.file_eye import classify_bytes, observe
from tools.vmcp.pty_eye import capture, probe
from tools.vmcp.tool_doctor import profile, report

__all__ = [
    "observe",
    "classify_bytes",
    "capture",
    "probe",
    "profile",
    "report",
    "run_matrix",
]
