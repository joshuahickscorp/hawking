"""Tool doctor: real local invocation, E.4 receipt, network/danger refused."""
from __future__ import annotations

import sys

from tools.vmcp.tool_doctor import profile, report


def test_profile_echo_is_a_real_invocation():
    rec = profile(["/bin/echo", "doctor-real"])
    assert rec["looked"] is True
    assert rec["empty_success"] is False
    assert rec["exit_code"] == 0
    assert "doctor-real" in rec["stdout"]
    receipt = rec["tool_receipt"]
    assert receipt["schema"] == "hawking.vmcp.tool_receipt.v1"
    assert receipt["status"] == "ok"
    assert receipt["elapsed_ms"] is not None
    assert receipt["network_used"] is False
    assert rec["execution"] == "REAL"


def test_profile_python_dash_c():
    rec = profile([sys.executable, "-c", "print('ok-py')"])
    assert rec["ok"] is True
    assert "ok-py" in rec["stdout"]


def test_profile_missing_tool_is_absent_not_empty_success():
    rec = profile(["vmcp-no-such-tool-abcxyz"])
    assert rec["available"] is False
    assert "TOOL_ABSENT" in rec["limitations"]
    assert rec["empty_success"] is False
    assert rec["looked"] is True


def test_profile_refuses_network_and_dangerous_before_exec():
    curl = profile(["curl", "http://example.invalid/"])
    assert curl["looked"] is False
    assert any("NETWORK_TOOL_REFUSED" in x for x in curl["limitations"])
    rm = profile(["rm", "-rf", "/"])
    assert rm["looked"] is False
    assert any("DANGEROUS_COMMAND" in x for x in rm["limitations"])


def test_report_lists_e3_classes_without_visionmcp():
    doc = report()
    names = [c["class"] for c in doc["classes"]]
    assert "file classifier" in names
    assert "process inspection" in names
    assert doc["network_used"] is False
    connected = [c for c in doc["classes"] if c["disposition"] == "CONNECTED"]
    assert len(connected) >= 5
    assert doc["empty_success"] is False
