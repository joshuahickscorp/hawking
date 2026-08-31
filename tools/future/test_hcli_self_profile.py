"""HCLI self-profile: rank refuses without timings, actionable refuses without a removal, hardware names raise."""
from __future__ import annotations

import json

import pytest

from tools.future import hcli_self_profile as sp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _measured(cost: str, median_ms: float) -> dict:
    return {
        "cost": cost,
        "state": "MEASURED_SELF_DIRTY",
        "self_timing": {
            "evidence_class": "SELF_MEASURED_DIRTY",
            "median_ms": median_ms,
            "n": 3,
            "samples_ms": [median_ms, median_ms, median_ms],
        },
    }


def _unknown(cost: str) -> dict:
    return {"cost": cost, "state": "UNKNOWN", "reason": "test", "self_timing": None}


def _hypo(**over) -> dict:
    base = {
        "cost": "git_query",
        "why_this_cost_exists": "git status refreshes the index on a 43GB tree",
        "removal": {
            "kind": "wait",
            "what_would_remove_it": "pass --no-optional-locks",
        },
        "cheapest_falsifier": "time git --no-optional-locks status; do not run the lock-taking arm",
    }
    base.update(over)
    return base


@pytest.fixture(scope="module")
def doc():
    path = sp.build(repeats=2)
    return json.loads(path.read_text())


def test_build_seals_receipt(doc):
    path = RECEIPTS / sp.RECEIPT
    assert path.is_file()
    assert doc["schema"] == sp.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["self_timing"]["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["self_timing"]["not_protected_absolute"] is True
    assert doc["self_timing"]["numbers_decide_nothing"] is True
    assert doc["self_timing"]["contamination"]["live_campaign_declared"] is True
    assert doc["resident_callable"]["frontier"] == "FT.HCLI_SELF.emit-workunits"
    assert doc["resident_callable"]["orchestration_bound"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]


def test_all_ten_buckets_are_attributed(doc):
    names = [row["cost"] for row in doc["attributed_costs"]]
    assert names == list(sp.COST_BUCKETS)
    for row in doc["attributed_costs"]:
        assert row["state"] in {"MEASURED_SELF_DIRTY", "UNKNOWN"}
        assert row.get("actionable") is True
        assert row["hypothesis"]["status"] == "HYPOTHESIS"
        assert row["hypothesis"]["does_not_decide"] is True


def test_compile_queue_idle_are_unknown_not_invented(doc):
    by = {row["cost"]: row for row in doc["attributed_costs"]}
    for name in ("compile_wait", "queue_wait", "process_idle"):
        assert by[name]["state"] == "UNKNOWN"
        assert by[name]["self_timing"] is None
        assert by[name]["reason"]
    assert by["compile_wait"]["cargo_forbidden"] is True


def test_git_query_is_sidecar_path_not_hcli_path(doc):
    git_row = {row["cost"]: row for row in doc["attributed_costs"]}["git_query"]
    if git_row["state"] == "MEASURED_SELF_DIRTY":
        assert git_row["is_hcli_path"] is False
        assert "--no-optional-locks" in git_row["argv"]
        assert isinstance(git_row["self_timing"]["median_ms"], (int, float))
        assert git_row["self_timing"]["n"] >= sp.MIN_REPEATS
    we = doc["worked_example"]
    assert we["id"] == "H.GIT.no-optional-locks"
    assert we["status"] == "HYPOTHESIS"
    assert we["causal_claim_verified"] is False
    assert we["historical_before"]["not_remeasured"] is True
    assert we["sidecar_path"] == "LANDED"
    assert we["hcli_path"] == "OPEN"
    assert "status" in we["hcli_argv"]
    assert "--no-optional-locks" not in we["hcli_argv"]


def test_rank_is_by_median_or_refused(doc):
    if doc["rank_state"] == "REFUSED":
        assert doc["ranked"] == []
        assert doc["rank_reason"]
        return
    assert doc["rank_state"] == "RANKED_SELF_DIRTY"
    medians = [row["median_ms"] for row in doc["ranked"]]
    assert medians == sorted(medians, reverse=True)
    assert [row["rank"] for row in doc["ranked"]] == list(range(1, len(doc["ranked"]) + 1))
    measured = [
        row["cost"]
        for row in doc["attributed_costs"]
        if row["state"] == "MEASURED_SELF_DIRTY"
    ]
    assert {row["cost"] for row in doc["ranked"]} == set(measured)


def test_no_numeric_hardware_fields(doc):
    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in HARDWARE_FIELDS:
                    assert not isinstance(value, (int, float)), here
                walk(value, here)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(doc, "")


def test_hardware_named_field_raises_on_write():
    probe = RECEIPTS / "_HCLI_SELF_PROFILE_PROBE_SHOULD_NOT_EXIST.json"
    if probe.exists():
        probe.unlink()
    with pytest.raises(HardwareClaimError, match="tps"):
        write_receipt(
            "_HCLI_SELF_PROFILE_PROBE_SHOULD_NOT_EXIST.json",
            {"schema": "probe", "tps": 12.0},
            "test_hcli_self_profile",
        )
    assert not probe.exists()


def test_hardware_named_timing_field_is_illegal():
    for name in sorted(HARDWARE_FIELDS):
        with pytest.raises(HardwareClaimError, match=name):
            sp.assert_timing_field_legal(name)
    sp.assert_timing_field_legal("median_ms")
    sp.assert_timing_field_legal("samples_ms")


def test_cost_without_removal_is_not_actionable():
    with pytest.raises(sp.ActionableRefused, match="removal"):
        sp.as_actionable(
            {
                "cost": "git_query",
                "why_this_cost_exists": "because",
                "cheapest_falsifier": "time it",
            }
        )
    with pytest.raises(sp.ActionableRefused, match="removal.kind"):
        sp.as_actionable(
            _hypo(removal={"kind": "magic", "what_would_remove_it": "wish"})
        )
    with pytest.raises(sp.ActionableRefused, match="why_this_cost_exists"):
        sp.as_actionable(_hypo(why_this_cost_exists=""))
    with pytest.raises(sp.ActionableRefused, match="cheapest_falsifier"):
        sp.as_actionable(_hypo(cheapest_falsifier=""))
    ok = sp.as_actionable(_hypo())
    assert ok["actionable"] is True
    assert ok["status"] == "HYPOTHESIS"


def test_rank_refuses_without_timing_data():
    with pytest.raises(sp.RankRefused, match="arbitrarily"):
        sp.rank_attributed_costs([])
    with pytest.raises(sp.RankRefused, match="arbitrarily"):
        sp.rank_attributed_costs([_unknown("git_query"), _unknown("compile_wait")])
    with pytest.raises(sp.RankRefused, match="arbitrarily"):
        sp.rank_attributed_costs(
            [
                {
                    "cost": "git_query",
                    "state": "MEASURED_SELF_DIRTY",
                    "self_timing": {"median_ms": None},
                }
            ]
        )


def test_rank_orders_by_median_and_does_not_include_unknown():
    ranked = sp.rank_attributed_costs(
        [
            _unknown("compile_wait"),
            _measured("verifier_overhead", 2.0),
            _measured("git_query", 180.0),
            _measured("subprocess_launch", 25.0),
        ]
    )
    assert [row["cost"] for row in ranked] == [
        "git_query",
        "subprocess_launch",
        "verifier_overhead",
    ]
    assert ranked[0]["rank"] == 1
    assert "compile_wait" not in {row["cost"] for row in ranked}


def test_compile_wait_refuses_a_cpu_proxy():
    with pytest.raises(sp.CompileWaitForbidden, match="cargo"):
        sp.record_compile_wait_as_measured(12.3)
    with pytest.raises(sp.CompileWaitForbidden, match="cargo"):
        sp.record_compile_wait_as_measured(0)


def test_bare_git_status_is_refused():
    with pytest.raises(sp.LiveGitStatusForbidden, match="index.lock"):
        sp.refuse_bare_git_status(["git", "status", "--porcelain"])
    with pytest.raises(sp.LiveGitStatusForbidden, match="index.lock"):
        sp.refuse_bare_git_status(["git", "-C", "/tmp", "status", "--short", "--branch"])
    sp.refuse_bare_git_status(["git", "--no-optional-locks", "status", "--porcelain"])
    sp.refuse_bare_git_status(["git", "rev-parse", "HEAD"])


def test_hcli_git_status_invoke_is_refused():
    with pytest.raises(sp.LiveGitStatusForbidden, match="git.status"):
        sp.refuse_hcli_git_status_invoke("git.status")
    with pytest.raises(ValueError, match="does not invoke"):
        sp.refuse_hcli_git_status_invoke("fs.read")


def test_recovered_hcli_git_status_lacks_the_flag():
    rec = sp.recover_hcli_git_status()
    assert rec["argv"][0] == "git" or rec["argv"][0] == "<dyn>"
    assert "status" in rec["argv"]
    assert rec["carries_no_optional_locks"] is False
    sidecar = sp.recover_sidecar_git()
    assert sidecar["carries_no_optional_locks"] is True
    assert sidecar["not_remeasured_without_flag"] is True


def test_catalog_every_bucket_is_actionable_and_covers_all_costs():
    catalog = sp.hypothesis_catalog()
    assert set(catalog) == set(sp.COST_BUCKETS)
    for cost, hypo in catalog.items():
        action = sp.as_actionable(hypo)
        assert action["cost"] == cost
        assert action["removal"]["kind"] in sp.REMOVAL_KINDS


def test_profile_copes_when_hcli_is_unimportable(monkeypatch):
    monkeypatch.setattr(sp, "_hcli_modules", lambda: (None, "ImportError: test"))
    doc = sp.profile(repeats=2)
    by = {row["cost"]: row for row in doc["attributed_costs"]}
    for cost in ("scheduler_decision", "tool_routing", "verifier_overhead"):
        assert by[cost]["state"] == "UNKNOWN"
        assert "hcli not importable" in by[cost]["reason"]
    assert by["git_query"]["state"] in {"MEASURED_SELF_DIRTY", "UNKNOWN"}
    assert by["compile_wait"]["state"] == "UNKNOWN"
    if by["git_query"]["state"] == "MEASURED_SELF_DIRTY":
        ranked = sp.rank_attributed_costs(doc["attributed_costs"])
        assert ranked
    else:
        with pytest.raises(sp.RankRefused):
            sp.rank_attributed_costs(
                [row for row in doc["attributed_costs"] if row["cost"] in {
                    "scheduler_decision", "tool_routing", "verifier_overhead",
                    "compile_wait", "queue_wait", "process_idle",
                }]
            )


def test_verifier_negative_control_rejects_vacuous_when_hcli_imports():
    mods, how = sp._hcli_modules()
    if mods is None:
        rec = _unknown("verifier_overhead")
        rec["reason"] = how
        assert rec["state"] == "UNKNOWN"
        return
    last = sp._time_verifier(mods)
    assert last["n_rejected"] >= 3
    reasons = " ".join(last["rejected_reasons"])
    assert "VACUOUS_COMMAND" in reasons or "EMPTY_COMMAND" in reasons
    ok, why = mods["verifier_pipeline"].command_is_admissible("true")
    assert ok is False
    assert why


def test_scheduler_analog_leaves_gpu_units_unassigned_when_cap_is_zero():
    mods, how = sp._hcli_modules()
    if mods is None:
        rec = sp._unknown("scheduler_decision", how)
        assert rec["state"] == "UNKNOWN"
        return
    last = sp._time_scheduler_decision(mods)
    assert last["n_units"] == 400
    assert last["n_unassigned"] >= last["n_gpu_exclusive_ready"]
    assert last["n_gpu_exclusive_ready"] > 0


def test_measure_refuses_a_single_sample():
    with pytest.raises(ValueError, match="single sample"):
        sp.profile(repeats=1)


def test_selftest_aliases_build():
    a = sp.selftest(repeats=2)
    assert a.name == sp.RECEIPT
    assert a.parent == RECEIPTS
