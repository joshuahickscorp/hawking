"""PTY eye: real session if openpty works; otherwise an explicit blocker."""
from __future__ import annotations

from hcli.agentos.vmcp.pty_eye import capture, probe


def test_probe_does_not_claim_a_pipe_is_a_pty():
    probed = probe()
    assert probed.get("empty_success") in (None, False)
    if probed.get("ok"):
        assert probed["used_real_pty"] is True
        assert probed["isatty"] is True
    else:
        assert probed["used_real_pty"] is False
        assert probed["blocker"]
        assert "pipe" in probed["blocker"].lower() or "EPERM" in probed["blocker"] or "openpty" in probed["blocker"]


def test_capture_without_argv_is_not_empty_success():
    rec = capture()
    assert rec["empty_success"] is False
    assert rec["looked"] is False
    assert "COMMAND_REQUIRED" in rec["limitations"]
    assert rec.get("results") is None


def test_capture_echo_is_real_or_honestly_blocked():
    rec = capture(argv=["/bin/echo", "pty-eye-real"])
    assert rec["empty_success"] is False
    if rec.get("used_real_pty"):
        assert rec["status"] == "CONNECTED"
        assert rec["ok"] is True
        assert rec["isatty"] is True
        assert "pty-eye-real" in (rec.get("text") or "")
        assert rec["tool_receipt"]["schema"] == "hawking.vmcp.tool_receipt.v1"
    else:
        assert rec["status"] == "PARKED"
        assert rec["looked"] is True
        assert "PTY_OPEN_DENIED" in rec["limitations"]
        assert rec.get("results") is None
        assert rec.get("items") is None
        assert rec["wake"]["required_kind"] == "call"
        assert rec["wake"]["required_symbol"] == "hcli.agentos.vmcp.pty_eye.capture"


def test_capture_refuses_network_tool():
    rec = capture(argv=["curl", "http://example.invalid/"])
    assert rec["looked"] is False
    assert any("NETWORK_TOOL_REFUSED" in x for x in rec["limitations"])
