"""Focused regression for the P5 perception registry drift summary line.

Exercises the production behavior added in this tranche:
``hawking.perception.tools.registry_drift_summary_line`` renders the same
observable drift verdict as ``registry_drift_summary`` as one stable line.
"""
from __future__ import annotations

from hawking.perception import tools


def test_registry_drift_summary_line_renders_ok_verdict() -> None:
    summary = tools.registry_drift_summary()
    line = tools.registry_drift_summary_line()

    assert summary["status"] == "ok"
    assert line == (
        f"registry=ok "
        f"observed={summary['observed_digest']} "
        f"tools={','.join(summary['tools'])}"
    )
    assert line.startswith("registry=ok ")
    assert summary["observed_digest"] in line
    for name in summary["tools"]:
        assert name in line


def test_registry_drift_summary_line_tracks_live_surface() -> None:
    line = tools.registry_drift_summary_line()
    assert tools.registry_digest() in line
    assert tools.TOOL_REGISTRY_DIGEST in line