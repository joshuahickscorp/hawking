"""Negative controls for the resident concurrency doctor.

A doctor nobody has watched refuse is a doctor that will invent a TPS.
These tests drive SLEEPING, every legal verdict including the negative,
hardware-named-field raises, and unscoped-law refusal.
"""
from __future__ import annotations

import json

import pytest

from tools.future import concurrency_doctor as cd
from tools.future import resident_health as rh
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)


def _stub_cap(*, presence: str = "UNDECLARED", **over) -> dict:
    body = {
        "resident_presence": presence,
        "gpu_authority": False,
        "protected_lease": False,
        "quiescence": "UNKNOWN",
        "resident_bytes": None,
        "memory_pressure": "UNKNOWN",
        "swap": {"status": "UNKNOWN", "swap_ins": None},
        "gpu_occupancy_class": "UNKNOWN",
    }
    body.update(over)
    body["not_runnable_reasons"] = cd._not_runnable_reasons(body)
    return body


def _syn(concurrency: int, work: float, *, gpu: str, ceremony: str, **over) -> dict:
    return cd.make_synthetic_observation(
        concurrency=concurrency,
        useful_work_per_wall_second=work,
        gpu_occupancy_class=gpu,
        host_ceremony_class=ceremony,
        **over,
    )


def _walk_hardware(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS:
                assert not isinstance(value, (int, float)) or isinstance(value, bool), here
            _walk_hardware(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk_hardware(value, f"{path}[{i}]")


def test_build_emits_sealed_receipt():
    out = cd.build()
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_CONCURRENCY_DOCTOR.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == cd.SCHEMA
    assert doc["schema"] == "hawking.future.concurrency_doctor.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["is_a_measurement"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["experiment_state"] == "SLEEPING"
    assert doc["verdict"] is None
    assert doc["verdict"] != "CONCURRENCY_HELPS"
    assert doc["law"]["status"] == "NOT_SEALED"
    assert doc["law"]["authoritative"] is False
    assert doc["proofs"]["synthetic"]["all_passed"] is True
    assert set(doc["proofs"]["synthetic"]["verdicts_reached"]) == set(cd.VERDICTS)
    assert doc["resident_callable"]["frontier"] == "FT.TPS.protected-tps"
    assert doc["resident_callable"]["orchestration_bound"] is False
    assert doc["sleeping_workunit"]["classification"] == "SLEEPING"
    assert doc["sleeping_workunit"]["status"] == "blocked"
    assert doc["plan"]["ladder"] == [1, 2, 3, 4]
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    _assert_no_hardware_claims(doc)
    _walk_hardware(doc)


def test_selftest_aliases_build():
    a = cd.selftest()
    b = cd.build()
    assert a.name == b.name == "RESIDENT_CONCURRENCY_DOCTOR.json"


def test_receipt_contains_no_useful_work_magnitude():
    doc = json.loads(cd.build().read_text())

    def walk(node):
        if isinstance(node, dict):
            assert "useful_work_per_wall_second" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)


def test_attempting_to_record_a_hardware_named_field_raises():
    with pytest.raises(HardwareClaimError, match="accepted_tps"):
        cd.refuse_hardware_named_number("accepted_tps", 12.0)
    with pytest.raises(HardwareClaimError, match="token_ns"):
        cd.refuse_hardware_named_number("token_ns", 8000)
    with pytest.raises(HardwareClaimError, match="tps"):
        cd.refuse_hardware_named_number("tps", 1)
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "CONCURRENCY_DOCTOR_MUST_NOT_EXIST.json",
            {"schema": "test", "tps": 26.65},
            "test_concurrency_doctor",
        )
    assert not (RECEIPTS / "CONCURRENCY_DOCTOR_MUST_NOT_EXIST.json").exists()


def test_synthetic_observation_with_hardware_named_field_raises():
    with pytest.raises(HardwareClaimError, match="accepted_tps"):
        cd.make_synthetic_observation(
            concurrency=1,
            useful_work_per_wall_second=10.0,
            gpu_occupancy_class="LOW",
            host_ceremony_class="LOW",
            accepted_tps=20.0,
        )
    with pytest.raises(HardwareClaimError, match="token_ns"):
        cd.observe(
            2,
            synthetic={
                "useful_work_per_wall_second": 10.0,
                "gpu_occupancy_class": "LOW",
                "host_ceremony_class": "LOW",
                "token_ns": 111.0,
            },
        )


def test_no_resident_emits_sleeping_and_refuses_verdict():
    cap = _stub_cap(presence="UNDECLARED")
    decision = cd.decide(cap)
    assert decision["experiment_state"] == "SLEEPING"
    assert decision["verdict"] is None
    assert decision["verdict"] != "CONCURRENCY_HELPS"
    assert "PRESENT" in decision["reason"] or "UNDECLARED" in decision["reason"]
    unit = decision["workunit"]
    assert unit["classification"] == "SLEEPING"
    assert unit["status"] == "blocked"
    assert unit["wakeup_state"] == "SLEEPING"
    assert unit["wake_condition"]["all_of"]
    assert "utilisation treated as available compute" in unit["wake_condition"]["never"]

    obs = cd.observe(1, capability=cap)
    assert obs["status"] == "REFUSED"
    assert obs["verdict"] is None
    assert obs["slots"]["accepted_tps"] is None
    assert obs["slots"]["token_ns"] is None
    with pytest.raises(cd.VerdictRefuse, match="REFUSED"):
        cd.verdict([obs])


def test_absent_resident_also_sleeps():
    sample = rh.make_sample(presence="ABSENT", pid=9, reason="dead")
    cap = cd.host_capability(
        resident_sample=sample,
        metal={"chip": "Apple M3 Ultra", "gpu_present": True, "is_a_measurement": False},
        occupancy={"status": "OK", "device_utilization_pct": 0},
    )
    assert cap["resident_presence"] == "ABSENT"
    assert cap["resident_bytes"] is None
    decision = cd.decide(cap)
    assert decision["experiment_state"] == "SLEEPING"
    assert decision["verdict"] is None


def test_present_resident_without_lease_still_sleeps():
    sample = rh.make_sample(
        presence="PRESENT",
        pid=4242,
        rss_bytes=5 * 1024 ** 3,
        memory_status="OK",
        uma_pressure_name="normal",
        uma_pressure_level=0,
        swap_ins=0,
    )
    cap = cd.host_capability(
        resident_sample=sample,
        metal={"chip": "Apple M3 Ultra", "gpu_present": True, "is_a_measurement": False},
        occupancy={"status": "FAILED", "device_utilization_pct": None},
    )
    assert cap["resident_presence"] == "PRESENT"
    assert cap["gpu_authority"] is False
    assert cap["protected_lease"] is False
    assert cap["runnable_today"] is False
    decision = cd.decide(cap)
    assert decision["experiment_state"] == "SLEEPING"
    assert decision["verdict"] is None
    obs = cd.observe(2, capability=cap)
    assert obs["status"] == "REFUSED"
    assert "lease" in obs["reason"]


def test_concurrency_helps_is_reachable():
    v = cd.verdict(
        [
            _syn(1, 10.0, gpu="MEDIUM", ceremony="LOW"),
            _syn(2, 18.0, gpu="HIGH", ceremony="LOW"),
        ]
    )
    assert v["verdict"] == "CONCURRENCY_HELPS"
    assert v["winner_concurrency"] == 2
    assert v["occupancy_is_not_the_objective"] is True


def test_no_useful_concurrency_headroom_is_reachable():
    v = cd.verdict(
        [
            _syn(1, 10.0, gpu="HIGH", ceremony="LOW"),
            _syn(2, 8.0, gpu="HIGH", ceremony="LOW"),
        ]
    )
    assert v["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"
    assert v["winner_concurrency"] == 1


def test_headroom_is_host_ceremony_is_reachable():
    v = cd.verdict(
        [
            _syn(1, 10.0, gpu="LOW", ceremony="HIGH"),
            _syn(2, 10.0, gpu="LOW", ceremony="HIGH"),
        ]
    )
    assert v["verdict"] == "HEADROOM_IS_HOST_CEREMONY"
    assert "ceremony" in v["why"].lower()


def test_high_occupancy_with_less_useful_work_loses():
    """95% GPU with fewer useful experiments per hour is not CONCURRENCY_HELPS."""
    v = cd.verdict(
        [
            _syn(1, 10.0, gpu="MEDIUM", ceremony="LOW"),
            _syn(2, 7.0, gpu="HIGH", ceremony="LOW"),
        ]
    )
    assert v["verdict"] != "CONCURRENCY_HELPS"
    assert v["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"
    assert v["winner_concurrency"] == 1
    assert v["high_occupancy_with_less_useful_work_loses"] is True


def test_occupancy_up_without_useful_work_up_is_not_helps():
    v = cd.verdict(
        [
            _syn(1, 10.0, gpu="LOW", ceremony="LOW"),
            _syn(2, 10.2, gpu="HIGH", ceremony="LOW"),
        ]
    )
    assert v["deltas"][0]["class"] == "FLAT"
    assert v["verdict"] != "CONCURRENCY_HELPS"
    assert v["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"


def test_unknown_occupancy_and_no_gain_refuses_rather_than_guessing():
    with pytest.raises(cd.VerdictRefuse, match="UNKNOWN"):
        cd.verdict(
            [
                _syn(1, 10.0, gpu="UNKNOWN", ceremony="UNKNOWN"),
                _syn(2, 9.5, gpu="UNKNOWN", ceremony="UNKNOWN"),
            ]
        )


def test_low_gpu_unknown_ceremony_refuses_the_ceremony_guess():
    with pytest.raises(cd.VerdictRefuse, match="HEADROOM_IS_HOST_CEREMONY"):
        cd.verdict(
            [
                _syn(1, 10.0, gpu="LOW", ceremony="UNKNOWN"),
                _syn(2, 10.0, gpu="LOW", ceremony="UNKNOWN"),
            ]
        )


def test_empty_observations_refuse_verdict():
    with pytest.raises(cd.VerdictRefuse, match="CONCURRENCY_HELPS"):
        cd.verdict([])


def test_single_level_is_not_a_comparison():
    with pytest.raises(cd.VerdictRefuse, match="two concurrency levels"):
        cd.verdict([_syn(1, 10.0, gpu="HIGH", ceremony="LOW")])


def test_zero_useful_work_is_refused():
    with pytest.raises(cd.ObservationRefuse, match="missing/zero"):
        cd.make_synthetic_observation(
            concurrency=1,
            useful_work_per_wall_second=0,
            gpu_occupancy_class="HIGH",
            host_ceremony_class="LOW",
        )


def test_plan_names_four_levels_and_the_stop_rule():
    p = cd.plan()
    assert [row["concurrency"] for row in p["levels"]] == [1, 2, 3, 4]
    assert "occupancy" in p["stop_rule"].lower()
    assert p["objective"].startswith("verified useful work")
    assert p["evidence_class_here"] == "STATIC_ONLY"
    assert "CONCURRENCY_HELPS" in p["what_it_refuses"][0] or any(
        "CONCURRENCY_HELPS" in x for x in p["what_it_refuses"]
    )


def test_advance_continues_only_while_informative():
    control = _syn(1, 10.0, gpu="MEDIUM", ceremony="LOW")
    up = _syn(2, 18.0, gpu="MEDIUM", ceremony="LOW")
    flat = _syn(2, 10.0, gpu="HIGH", ceremony="LOW")
    stepped = cd.advance([control, up])
    assert stepped["action"] == "RUN"
    assert stepped["next"] == 3
    stopped = cd.advance([control, flat])
    assert stopped["action"] == "STOP"
    assert stopped["next"] is None
    assert "occupancy" in stopped["why"].lower()
    first = cd.advance([])
    assert first["next"] == 1
    after_control = cd.advance([control])
    assert after_control["next"] == 2
    refused = cd.observe(1, capability=_stub_cap())
    assert cd.advance([refused])["action"] == "STOP"


def test_advance_stops_at_four():
    rows = [
        _syn(n, 10.0 * n, gpu="MEDIUM", ceremony="LOW") for n in (1, 2, 3, 4)
    ]
    done = cd.advance(rows)
    assert done["action"] == "STOP"
    assert done["next"] is None


def test_scope_fields_are_required():
    kwargs = dict(
        verdict_name="NO_USEFUL_CONCURRENCY_HEADROOM",
        statement="scoped",
        machine="Apple M3 Ultra",
        nx="noetic-sealed-3.14",
        runtime="Metal",
        context_regime="WHOLE_MODEL_BODY",
    )
    for field in cd.SCOPE_FIELDS:
        bad = dict(kwargs)
        bad[field] = ""
        with pytest.raises(cd.ScopeRefused, match=field):
            cd.seal_law(**bad)
        bad[field] = "UNKNOWN"
        with pytest.raises(cd.ScopeRefused, match=field):
            cd.seal_law(**bad)


def test_law_without_context_regime_is_refused():
    with pytest.raises(cd.ScopeRefused, match="context_regime"):
        cd.seal_law(
            verdict_name="CONCURRENCY_HELPS",
            statement="x",
            machine="Apple M3 Ultra",
            nx="noetic-sealed-3.14",
            runtime="Metal",
            context_regime="ALL_APPLE_SILICON",
        )


def test_universalisation_to_flash_m5_fpga_cuda_is_refused():
    base = dict(
        verdict_name="CONCURRENCY_HELPS",
        statement="does not transfer",
        machine="Apple M3 Ultra",
        nx="noetic-sealed-3.14",
        runtime="Metal",
        context_regime="WHOLE_MODEL_BODY",
    )
    for name in cd.UNIVERSAL_REFUSALS:
        with pytest.raises(cd.ScopeRefused, match=name):
            cd.seal_law(**base, applies_to=[name])
        with pytest.raises(cd.ScopeRefused, match=name):
            cd.seal_law(**base, transfer_to=[name])
    with pytest.raises(cd.ScopeRefused, match="class"):
        cd.seal_law(**{**base, "machine": "Apple silicon"})


def test_scoped_law_shape_is_not_authoritative():
    law = cd.seal_law(
        verdict_name="HEADROOM_IS_HOST_CEREMONY",
        statement=(
            "on this machine, this NX, this runtime, HOST_CEREMONY regime, "
            "the GPU idles on CPU prep/readback/sync"
        ),
        machine="Apple M3 Ultra",
        nx="noetic-sealed-3.14",
        runtime="Metal",
        context_regime="HOST_CEREMONY",
    )
    assert law["authoritative"] is False
    assert set(law["scope"]) == set(cd.SCOPE_FIELDS)
    assert law["does_not_apply_to"] == list(cd.UNIVERSAL_REFUSALS)


def test_occupancy_class_is_taxonomy_not_compute():
    assert cd.occupancy_class_from_pct(None) == "UNKNOWN"
    assert cd.occupancy_class_from_pct(-1) == "UNKNOWN"
    assert cd.occupancy_class_from_pct(0) == "LOW"
    assert cd.occupancy_class_from_pct(19) == "LOW"
    assert cd.occupancy_class_from_pct(20) == "MEDIUM"
    assert cd.occupancy_class_from_pct(79) == "MEDIUM"
    assert cd.occupancy_class_from_pct(80) == "HIGH"
    assert cd.occupancy_class_from_pct(95) == "HIGH"


def test_metal_state_shape_is_not_a_measurement():
    from tools.future.hardware_doctor import metal_state

    state = metal_state()
    assert state["is_a_measurement"] is False
    cap = cd.host_capability(
        resident_sample=rh.make_sample(presence="UNDECLARED", reason="test"),
        metal=state,
        occupancy={"status": "OK", "device_utilization_pct": 0},
    )
    assert cap["metal"]["is_a_measurement"] is False
    assert cap["occupancy_is_not_available_compute"] is True
    assert cap["gpu_authority"] is False
    _assert_no_hardware_claims(cap)


def test_sleeping_workunit_validates_as_hcli():
    unit = cd.emit_sleeping_workunit(_stub_cap())
    from tools.future.workunit_species import validate_emitted_unit

    validate_emitted_unit(unit)
    assert unit["id"] == cd.WORKUNIT_ID
    assert "GPU" in str(unit["resource_class"]).upper()
    assert unit.get("verdict") != "CONCURRENCY_HELPS"


def test_live_undeclared_sample_is_coped_not_skipped():
    """Absence is a recorded refusal. A skip would hide the only live path."""
    sample = rh.sample()
    assert sample["resident"]["presence"] in {"UNDECLARED", "ABSENT", "PRESENT", "UNKNOWN"}
    obs = cd.observe(1, resident_sample=sample)
    assert obs["status"] == "REFUSED"
    if sample["resident"]["presence"] != "PRESENT":
        assert sample["resident"]["rss_bytes"] is None
        assert "resident" in obs["reason"]


def test_prove_synthetic_reaches_all_three_verdicts():
    proofs = cd.prove_synthetic()
    assert proofs["all_passed"] is True
    assert set(proofs["verdicts_reached"]) == set(cd.VERDICTS)


def test_helps_then_down_keeps_the_useful_work_winner():
    v = cd.verdict(
        [
            _syn(1, 10.0, gpu="MEDIUM", ceremony="LOW"),
            _syn(2, 18.0, gpu="MEDIUM", ceremony="LOW"),
            _syn(3, 12.0, gpu="HIGH", ceremony="LOW"),
        ]
    )
    assert v["verdict"] == "CONCURRENCY_HELPS"
    assert v["winner_concurrency"] == 2
