"""The gate must actually refuse, and its refusal must not be bypassable quietly."""
import pathlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integration_gate as g
from _common import REPO
from tools.future import status_causality as sc


def test_it_finds_the_test_module_for_a_source_module():
    want = g.required_tests(["tools/future/path_to_71.py"])
    assert "tools/future/test_path_to_71.py" in want


def test_a_test_module_maps_to_itself_not_to_test_test():
    want = g.required_tests(["tools/future/test_path_to_71.py"])
    assert want == ["tools/future/test_path_to_71.py"]


def test_land_refuses_when_the_suite_is_red(tmp_path):
    """The exact incident this module exists for."""
    red = REPO / "tools" / "future" / "test__gate_probe_red.py"
    red.write_text("def test_red():\n    assert 1.0027 == 1.0027008\n")
    src = REPO / "tools" / "future" / "_gate_probe_red.py"
    src.write_text("VALUE = 1.0027008\n")
    msg = tmp_path / "m.txt"
    msg.write_text("should never land\n")
    try:
        try:
            g.land(["tools/future/_gate_probe_red.py"], str(msg))
        except g.GateRed as exc:
            assert "REFUSED" in str(exc)
            return
        raise AssertionError("the gate landed a red suite")
    finally:
        red.unlink(missing_ok=True)
        src.unlink(missing_ok=True)


def test_malformed_receipt_is_red(tmp_path):
    bad = REPO / "receipts" / "future" / "_GATE_PROBE_BAD.json"
    bad.write_text("{not json")
    try:
        r = g.receipts_parse(["receipts/future/_GATE_PROBE_BAD.json"])
        assert not r["green"]
        assert r["malformed"][0]["path"].endswith("_GATE_PROBE_BAD.json")
    finally:
        bad.unlink(missing_ok=True)


def test_the_escape_hatch_is_loud_not_quiet():
    d = g.build()
    assert d["escape_hatch"]["not_quiet"] is True
    assert "KNOWN_RED_CHECKPOINT" in d["escape_hatch"]["behaviour"]


def test_the_incident_that_caused_this_is_named():
    d = g.build()
    assert "db4dacede" in d["why_it_is_a_door_not_a_rule"]


def test_cli_check_exits_nonzero_on_red():
    red = REPO / "tools" / "future" / "test__gate_probe_red2.py"
    red.write_text("def test_red():\n    assert False\n")
    src = REPO / "tools" / "future" / "_gate_probe_red2.py"
    src.write_text("X = 1\n")
    try:
        p = subprocess.run(
            [sys.executable, "tools/future/integration_gate.py", "--check",
             "tools/future/_gate_probe_red2.py"],
            cwd=REPO, capture_output=True, text=True, timeout=300)
        assert p.returncode != 0, p.stdout
    finally:
        red.unlink(missing_ok=True)
        src.unlink(missing_ok=True)


def test_check_records_the_five_causality_fields():
    """A coverage number no test defends will drift back to zero."""
    r = g.check([])
    assert g.records_five_fields(r)
    src = pathlib.Path(g.__file__).read_text()
    assert "sc.emit(" in src
    assert r["probe_performed"]
    assert r["direct_observation"]
    assert r["direct_observation"] != r["verdict"]
    assert r["direct_observation"] != r["interpretation"]
    assert "required_tests" in r["probe_performed"] or "json.loads" in r["probe_performed"]
    assert r["verdict"] == "GREEN"
    assert r["green"] is True
    assert r["causality_verdict"] in {sc.SUPPORTED, sc.OVERREACHING, sc.UNTESTED}


def test_unsupplied_observation_records_untested_not_a_restatement():
    result = {"green": False, "verdict": "RED"}
    rec = g.record_check_causality(
        result, probe_performed="", direct_observation=""
    )
    assert rec["verdict"] == sc.UNTESTED
    assert rec["direct_observation"] in ("", None)
    assert rec["direct_observation"] != "RED"
    assert "RED" not in str(rec["direct_observation"] or "")
    assert result["verdict"] == "RED"
    assert result["green"] is False
    assert rec["interpretation"] != rec["direct_observation"]


def test_overreaching_does_not_override_green_or_red(monkeypatch):
    def overreach(status, **kwargs):
        return {
            "probe_performed": kwargs.get("probe_performed") or "p",
            "direct_observation": kwargs.get("direct_observation") or "o",
            "interpretation": kwargs.get("interpretation") or status,
            "confidence": {
                "level": "LOW",
                "about": "a",
                "would_raise": "b",
                "would_lower": "c",
            },
            "alternatives": [
                {
                    "hypothetical": "h",
                    "consistent_with_observation": True,
                    "consistent_with_claim": False,
                }
            ],
            "verdict": sc.OVERREACHING,
            "falsifier": "f",
            "probe_kind": sc.PROBE_MEASURED_FLAGS,
            "claim_kind": sc.CLAIM_OBJECT_ABSENCE,
        }

    monkeypatch.setattr(g.sc, "emit", overreach)
    green = g.check([])
    assert green["green"] is True
    assert green["verdict"] == "GREEN"
    assert green["causality_verdict"] == sc.OVERREACHING

    red = g.receipts_parse(["receipts/future/INTEGRATION_GATE.json"])
    # receipts_parse itself is not the door; check() is. A malformed receipt
    # still has to stay RED after an OVERREACHING stamp.
    bad = REPO / "receipts" / "future" / "_GATE_PROBE_CAUSALITY_BAD.json"
    bad.write_text("{not json")
    try:
        stamped = g.check(["receipts/future/_GATE_PROBE_CAUSALITY_BAD.json"])
        assert stamped["green"] is False
        assert stamped["verdict"] == "RED"
        assert stamped["causality_verdict"] == sc.OVERREACHING
    finally:
        bad.unlink(missing_ok=True)
