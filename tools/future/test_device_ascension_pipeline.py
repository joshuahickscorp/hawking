"""Arrival pipeline pins. A guard nobody has watched fail is not a guard."""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from tools.future import device_ascension_pipeline as dap
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


FIXTURE_DEST = {
    "soc": "Apple M4 Pro",
    "arch": "arm64",
    "cpu_cores": 14,
    "perf_cores": 10,
    "efficiency_cores": 4,
    "gpu_cores": 20,
    "memory_bytes": 51539607552,
    "os": "Darwin test",
    "os_product": "UNKNOWN",
    "discovery_class": "STATIC_ONLY",
}


def _dest_id() -> str:
    return dap.machine_id(FIXTURE_DEST)


def _foreign_verified(**overrides: object) -> dict:
    law = {
        "law_id": "LAW-TEST-FOREIGN-VERIFIED",
        "statement": "A MACHINE_LOCAL law verified on machine A.",
        "scope": "MACHINE_LOCAL",
        "evidence_strength": "VERIFIED",
        "evidence_class": "PROTECTED_ABSOLUTE",
        "origin_machine_id": "foreign|machine-A|cpu=1|gpu=1|mem=1",
        "status": "ACTIVE",
        "calibration_axis": "bandwidth",
    }
    law.update(overrides)
    return law


def test_build_emits_sealed_receipt():
    out = dap.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DEVICE_ASCENSION_PIPELINE.json"
    assert doc["schema"] == "hawking.future.device_ascension.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    _assert_no_hardware_claims(doc)


def test_stage_order_is_the_arrival_pipe_not_the_adp_campaign():
    assert dap.STAGES == (
        "discover_hardware",
        "build_machine_genome",
        "derive_capabilities",
        "import_laws_as_hypotheses",
        "calibration_experiments",
        "recompile_physical_graph",
        "protected_measurement",
        "emit_machine_laws",
    )
    # Codex ADP campaign is a different object; named so we do not fork it.
    assert "adp_seal" in dap.CODEX_ADP_STAGES
    assert "fingerprint" in dap.CODEX_ADP_STAGES
    assert "adp_seal" not in dap.STAGES
    result = dap.run_pipeline(hardware=FIXTURE_DEST, foreign_laws=dap.FOREIGN_SEED_LAWS)
    assert [s["name"] for s in result["stages"]] == list(dap.STAGES)
    assert [s["index"] for s in result["stages"]] == list(range(1, 9))


def test_dry_run_marks_stage_7_unavailable_and_emits_no_verified_laws():
    result = dap.run_pipeline(dry_run=True, hardware=FIXTURE_DEST)
    s7 = result["stages"][6]
    assert s7["name"] == "protected_measurement"
    assert s7["status"] == "UNAVAILABLE"
    assert s7["payload"]["gpu_authority"] is False
    assert s7["payload"]["bench_state"] == "UNKNOWN"
    s8 = result["stages"][7]["payload"]
    assert s8["n_verified_performance_laws"] == 0
    for law in s8["own_laws"]:
        assert law["evidence_strength"] != "VERIFIED"
        assert law["evidence_class"] == "STATIC_ONLY"
        assert law["verified_on_dest"] is False


def test_imported_seed_laws_are_all_hypotheses_on_dest():
    result = dap.run_pipeline(hardware=FIXTURE_DEST, foreign_laws=dap.FOREIGN_SEED_LAWS)
    dest = result["dest_machine_id"]
    assert dest == _dest_id()
    imported = result["stages"][3]["payload"]["laws"]
    assert imported
    for law in imported:
        assert law["origin_machine_id"] != dest
        assert law["crosses_machine_boundary"] is True
        assert law["imported"] is True
        assert law["evidence_strength"] == "HYPOTHESIS"
        assert law["evidence_class"] == "STATIC_ONLY"
        assert law["verified_on_dest"] is False
        assert law["scope"] != "GENERIC_VERIFIED"
        assert law["origin_evidence_strength"] in {"VERIFIED", "HYPOTHESIS"}


def test_generic_verified_scope_is_clamped():
    law = _foreign_verified(
        law_id="LAW-TEST-GENERIC",
        scope="GENERIC_VERIFIED",
    )
    got = dap.import_law(law, _dest_id())
    assert got["origin_scope"] == "GENERIC_VERIFIED"
    assert got["scope"] == "MACHINE_LOCAL"
    assert got["evidence_strength"] == "HYPOTHESIS"


def test_model_local_keeps_model_scope_but_loses_verified():
    law = _foreign_verified(
        law_id="LAW-TEST-MODEL",
        scope="MODEL_LOCAL",
        evidence_class="DIAGNOSTIC_RELATIVE",
        calibration_axis="representation",
    )
    got = dap.import_law(law, _dest_id())
    assert got["scope"] == "MODEL_LOCAL"
    assert got["evidence_strength"] == "HYPOTHESIS"
    assert got["evidence_class"] == "STATIC_ONLY"
    assert got["origin_evidence_class"] == "DIAGNOSTIC_RELATIVE"


def test_missing_origin_is_fail_closed_foreign():
    law = _foreign_verified()
    del law["origin_machine_id"]
    got = dap.import_law(law, _dest_id())
    assert got["crosses_machine_boundary"] is True
    assert got["evidence_strength"] == "HYPOTHESIS"
    assert got["origin_machine_id"] == "UNKNOWN_ORIGIN"


def test_one_cheapest_calibration_experiment_per_hypothesis():
    result = dap.run_pipeline(hardware=FIXTURE_DEST, foreign_laws=dap.FOREIGN_SEED_LAWS)
    hyps = result["stages"][3]["payload"]["laws"]
    exps = result["stages"][4]["payload"]["experiments"]
    assert len(exps) == len(hyps)
    by_id = {e["hypothesis_law_id"]: e for e in exps}
    assert by_id["LAW-FOREIGN-H100-BANDWIDTH"]["kind"] == "PROTECTED_BANDWIDTH_TRIAD"
    assert by_id["LAW-FOREIGN-H100-BANDWIDTH"]["status"] == "UNAVAILABLE"
    assert by_id["LAW-FOREIGN-ALREADY-HYPOTHESIS"]["kind"] == "STATIC_SYSCTL_IDENTITY"
    assert by_id["LAW-FOREIGN-ALREADY-HYPOTHESIS"]["status"] == "RUNNABLE"
    assert by_id["LAW-FOREIGN-ALREADY-HYPOTHESIS"]["cost_rank"] == 0
    # Cheapest for a known axis is unique and not the generic fallback when a
    # cheaper exact axis exists.
    assert by_id["LAW-FOREIGN-GENERIC-FUSION"]["kind"] == "PROTECTED_FUSION_AB"
    assert by_id["LAW-FOREIGN-GENERIC-FUSION"]["cost_rank"] < 20


def test_physical_graph_is_plan_only_and_fpga_is_not_a_civilization():
    result = dap.run_pipeline(hardware=FIXTURE_DEST)
    graph = result["stages"][5]["payload"]
    assert graph["qualification"] == "PLAN_ONLY"
    assert graph["not_executed"] is True
    assert graph["device_placement"]["selected"] != "fpga"
    caps = result["stages"][2]["payload"]
    assert caps["fpga"]["civilization"] is False
    assert caps["fpga"]["backend_status"] == "FUTURE"
    assert "not its own civilization" in caps["fpga"]["note"]


def test_selftest_aliases_build():
    assert dap.selftest is dap.build


# --------------------------------------------------------------------------- negative controls: the refusal must fire


def test_preserve_verified_raises_on_every_import_surface():
    """Watch the refusal fire. A flag nobody has seen reject is not a guard."""
    law = _foreign_verified()
    dest = _dest_id()
    with pytest.raises(dap.VerifiedImportRefused, match="preserve_verified"):
        dap.apply_downgrade_rule(law, dest, preserve_verified=True)
    with pytest.raises(dap.VerifiedImportRefused, match="preserve_verified"):
        dap.import_law(law, dest, preserve_verified=True)
    with pytest.raises(dap.VerifiedImportRefused, match="preserve_verified"):
        dap.import_laws([law], dest, preserve_verified=True)
    with pytest.raises(dap.VerifiedImportRefused, match="preserve_verified"):
        dap.import_law_catalog(dest, laws=[law], preserve_verified=True)


def test_every_public_import_path_strips_verified_and_protected_absolute():
    dest = _dest_id()
    law = _foreign_verified(scope="GENERIC_VERIFIED")
    surfaces = dap.import_surfaces()
    assert set(surfaces) == {
        "apply_downgrade_rule",
        "import_law",
        "import_laws",
        "import_law_catalog",
    }
    got = {
        "apply_downgrade_rule": surfaces["apply_downgrade_rule"](law, dest),
        "import_law": surfaces["import_law"](law, dest),
        "import_laws": surfaces["import_laws"]([law], dest)[0],
        "import_law_catalog": surfaces["import_law_catalog"](dest, laws=[law])[0],
    }
    for name, imported in got.items():
        assert imported["evidence_strength"] == "HYPOTHESIS", name
        assert imported["evidence_class"] == "STATIC_ONLY", name
        assert imported["scope"] == "MACHINE_LOCAL", name
        assert imported["verified_on_dest"] is False, name
        assert imported["crosses_machine_boundary"] is True, name
        assert imported["origin_evidence_strength"] == "VERIFIED", name
        assert imported["origin_evidence_class"] == "PROTECTED_ABSOLUTE", name


def test_pipeline_import_stage_is_also_an_import_path():
    law = _foreign_verified(law_id="LAW-PLANTED-VERIFIED")
    result = dap.run_pipeline(hardware=FIXTURE_DEST, foreign_laws=[law])
    imported = result["stages"][3]["payload"]["laws"]
    assert len(imported) == 1
    assert imported[0]["evidence_strength"] == "HYPOTHESIS"
    assert imported[0]["evidence_class"] == "STATIC_ONLY"


def test_assert_no_foreign_verified_actually_refuses_a_planted_record():
    dest = _dest_id()
    planted = _foreign_verified()
    # The guard must fail when a VERIFIED foreign law is planted, not pass
    # because nobody fed it a bad record.
    with pytest.raises(dap.ArrivalInvariantError, match="VERIFIED"):
        dap.assert_no_foreign_verified([planted], dest)
    with pytest.raises(dap.ArrivalInvariantError, match="PROTECTED_ABSOLUTE"):
        dap.assert_no_foreign_verified(
            [{**planted, "evidence_strength": "HYPOTHESIS"}], dest
        )
    with pytest.raises(dap.ArrivalInvariantError, match="GENERIC_VERIFIED"):
        dap.assert_no_foreign_verified(
            [{
                **planted,
                "evidence_strength": "HYPOTHESIS",
                "evidence_class": "STATIC_ONLY",
                "scope": "GENERIC_VERIFIED",
            }],
            dest,
        )
    # Legal imported form must pass, so the guard can also succeed.
    dap.assert_no_foreign_verified([dap.import_law(planted, dest)], dest)


def test_empty_dest_is_refused_rather_than_guessed():
    with pytest.raises(dap.VerifiedImportRefused, match="dest_machine_id is empty"):
        dap.import_law(_foreign_verified(), "")


def test_unttyped_law_is_refused_not_imported_as_verified():
    with pytest.raises(dap.VerifiedImportRefused, match="missing required"):
        dap.import_law({"statement": "no id"}, _dest_id())


def test_receipt_cannot_carry_a_hardware_number():
    result = dap.run_pipeline(hardware=FIXTURE_DEST)
    _assert_no_hardware_claims(result)
    planted = {"tps": 12.0, "nested": {"bandwidth_gbps": 1}}
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims(planted)
    for field in HARDWARE_FIELDS:
        with pytest.raises(HardwareClaimError):
            _assert_no_hardware_claims({field: 1})


def test_odyssey_ii_field_names_are_present_and_sibling_store_is_not_imported():
    for field in ("law_id", "scope", "evidence_strength"):
        assert field in dap.LAW_FIELDS
    # Sibling lane must not be imported; the module's own globals are the vocab.
    assert "odyssey2_law_store" not in dir(dap)
    # NOT sys.modules: that is a process global, so a sibling lane's own test
    # importing the store makes this fail for reasons unrelated to this module.
    # The real property is that THIS module never imports it -- check the source.
    # Parse imports rather than grepping text: the module legitimately NAMES the
    # sibling in its recovery notes to record that it deliberately did not import it.
    tree = ast.parse(pathlib.Path(dap.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("odyssey2_law_store" in m for m in imported), sorted(imported)
