"""G016 role_suitability: vector not a blend; null is not zero."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "future"))

import role_suitability as rs  # noqa: E402


RECEIPT = ROOT / "receipts" / "future" / "G016_ROLE_SUITABILITY.json"

# Concrete values from the cited receipts, not from this file's memory.
SEALED_SO_FAIL = 0.33969465648854963
SEALED_RECOVERY = 0.007272727272727273
G007_PPT = 23.5
G007_PREFILL_SHARE = 0.9308
G007_PREFILL_TPS = 27.24


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        "G016_ROLE_SUITABILITY.json missing; "
        "run python3.12 tools/future/role_suitability.py --selftest"
    )
    return json.loads(RECEIPT.read_text())


def _live(doc: dict) -> dict:
    cands = doc["candidates"]
    assert cands, "receipt has no candidates"
    live = next(
        (c for c in cands if c.get("role") == "hcli_resident"),
        cands[0],
    )
    return live


def test_null_vs_zero_on_synthetics():
    """A failing candidate is measured; a silent candidate is null, not 0."""
    neg = rs.negative_control()
    assert neg["passed"], neg["failed_checks"]

    failed = neg["all_failed"]
    none = neg["no_evidence"]
    part = neg["partial"]

    for name in rs.HCLI_RESIDENT_REQUIREMENTS:
        cell = failed["requirements"][name]
        assert cell["value"] is not None, name
    assert failed["requirements"]["repair_behaviour"]["value"] == 0.0
    assert failed["comparable_value"] == float(len(rs.HCLI_RESIDENT_REQUIREMENTS))

    for name in rs.HCLI_RESIDENT_REQUIREMENTS:
        cell = none["requirements"][name]
        assert cell["value"] is None, (name, cell)
        assert cell["value"] is not 0
        assert cell["value"] != 0.0
        assert cell["cause"]
    assert none["comparable_value"] is None
    assert none["n_measured_requirements"] == 0

    assert failed["comparable_value"] != none["comparable_value"]
    assert part["covered"] == ["instruction_following", "repair_behaviour"]
    assert part["n_measured_requirements"] == 2
    assert part["comparable_value"] == 2.0
    assert part["comparable_value"] not in (
        None, 0.0, float(len(rs.HCLI_RESIDENT_REQUIREMENTS)),
    )


def test_receipt_null_vs_zero_and_partial_live_coverage():
    doc = _receipt()
    live = _live(doc)
    req = live["requirements"]

    tool = req["tool_reliability"]
    assert tool["value"] is None
    assert tool["value"] is not 0 and tool["value"] != 0.0
    assert tool["cause"]
    assert "observations" in tool["cause"]
    assert tool["source_receipt"] == "receipts/future/G016_RELIABILITY_AXIS.json"
    assert "tool_failure_rate.value" in tool["source_field"]

    so = req["instruction_following"]
    assert so["value"] == SEALED_SO_FAIL
    assert so["source_receipt"] == "receipts/future/G016_RELIABILITY_AXIS.json"
    assert "structured_action_failure_rate.value" in so["source_field"]

    repair = req["repair_behaviour"]
    assert repair["value"] == SEALED_RECOVERY
    assert repair["source_receipt"] == "receipts/future/G016_RELIABILITY_AXIS.json"
    assert "recovery_rate.value" in repair["source_field"]

    econ = req["context_economy"]["value"]
    assert econ["prompt_tokens_per_generated_token"] == G007_PPT
    assert econ["prefill_share_of_model_wall"] == G007_PREFILL_SHARE
    assert req["context_economy"]["source_receipt"] == (
        "receipts/future/G007_ROUND_PREFILL_ECONOMICS.json"
    )

    lat = req["latency"]["value"]
    assert lat["prefill_tok_per_s"] == G007_PREFILL_TPS
    assert lat["per_call_wall_s"]
    assert lat["per_call_wall_s_min"] == min(lat["per_call_wall_s"])
    assert lat["per_call_wall_s_max"] == max(lat["per_call_wall_s"])
    assert req["latency"]["source_receipt"] == (
        "receipts/future/G007_ROUND_PREFILL_ECONOMICS.json"
    )

    sel = req["scientific_target_selection"]["value"]
    assert isinstance(sel["self_selected_target"], int)
    assert isinstance(sel["selected_AND_measured"], int)
    assert req["scientific_target_selection"]["source_receipt"] == (
        "tools/future/autonomy_ratio.py"
    )

    assert "tool_reliability" not in live["covered"]
    assert "instruction_following" in live["covered"]
    assert live["n_measured_requirements"] == len(live["covered"])
    assert live["n_measured_requirements"] != 0
    assert live["n_measured_requirements"] != live["n_requirements"]
    assert live["comparable_value"] == float(live["n_measured_requirements"])


def test_receipt_has_no_blended_score():
    doc = _receipt()
    hits = rs.blend_keys_found(doc)
    assert hits == [], hits
    live = _live(doc)
    assert live.get("not_a_blend") is True
    assert "score" not in live
    assert "grade" not in live
    assert "weights" not in live
    axis = doc["pareto_axis"]
    assert axis["comparable_value"] == "n_measured_requirements"
    assert axis["not_a_blend"] is True
    assert live["comparable_value"] == float(live["n_measured_requirements"])


def test_hcli_resident_contract_exists_magnetar_pulsar_do_not():
    doc = _receipt()
    assert "hcli_resident" in doc["roles"]
    names = [r["name"] for r in doc["roles"]["hcli_resident"]["requirements"]]
    assert names == list(rs.HCLI_RESIDENT_REQUIREMENTS)
    absent = {row["role"]: row for row in doc["roles_without_contracts"]}
    assert "magnetar" in absent and "pulsar" in absent
    assert absent["magnetar"]["requirements"] is None
    assert absent["pulsar"]["requirements"] is None
    assert absent["magnetar"]["cause"]
    assert absent["pulsar"]["cause"]


def test_controls_in_receipt_are_distinguishable():
    doc = _receipt()
    ctrl = doc["controls"]["negative_control"]
    assert ctrl["passed"] is True
    failed = ctrl["all_failed"]
    none = ctrl["no_evidence"]
    part = ctrl["partial"]
    assert failed["comparable_value"] == 6.0
    assert failed["repair_behaviour"] == 0.0
    assert failed["self_selected_target"] == 0
    assert none["comparable_value"] is None
    assert none["all_values_null"] is True
    assert none["n_measured_requirements"] == 0
    assert part["covered"] == ["instruction_following", "repair_behaviour"]
    assert part["comparable_value"] == 2.0
    assert failed["comparable_value"] != none["comparable_value"]


def test_mutation_check_recorded_and_restored():
    doc = _receipt()
    mut = doc["controls"]["mutation_check"]
    assert mut["restored"] is True
    for key in ("null_vs_zero", "harvest"):
        row = mut[key]
        assert row["anchor"]
        assert row["before_sha256"]
        assert row["after_sha256"]
        assert row["before_sha256"] != row["after_sha256"]
        assert row["probe_failed_as_required"] is True
        assert "MUTATION_ACTIVE" not in Path(
            ROOT / "tools" / "future" / "role_suitability.py"
        ).read_text().split("def measured_or_null")[1].split("def ")[0]
    harv_fn = (ROOT / "tools" / "future" / "campaign_pareto.py").read_text().split(
        "def harvest_role_suitability"
    )[1].split("def ")[0]
    assert "MUTATION_ACTIVE harvest" not in harv_fn
    assert "LOAD_BEARING_HARVEST" in harv_fn


def test_campaign_pareto_reports_non_null_role_suitability():
    """The frontier must actually ingest G016_ROLE_SUITABILITY, not leave the axis at zero."""
    import campaign_pareto as cp

    doc = _receipt()
    live = _live(doc)
    assert live["comparable_value"] == float(live["n_measured_requirements"])
    assert live["comparable_value"] > 0

    harvested = cp.harvest()
    debt = harvested["measurement_debt"]
    assert "role_suitability" not in debt["unmeasured_for_everyone"], debt["unmeasured_for_everyone"]
    assert debt["n_with_axis"]["role_suitability"] > 0
    scored = harvested["scored"]
    with_role = [p for p in scored if "role_suitability" in p["values"]]
    assert with_role, "harvest produced no comparable role_suitability values"
    hits = [
        p for p in with_role
        if "sealed-3.14" in str(p.get("name", "")) or "sealed-3.14" in str(p.get("id", ""))
    ]
    assert hits, [p["id"] for p in with_role[:8]]
    point = hits[0]
    assert point["values"]["role_suitability"] == float(live["n_measured_requirements"])
    contract = point.get("role_suitability_contract") or {}
    assert contract.get("named_value") == "n_measured_requirements"
    assert contract.get("not_a_blend") is True
    assert "score" not in contract
    assert "weights" not in contract
