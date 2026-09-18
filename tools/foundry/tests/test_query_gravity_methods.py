import hashlib
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import json

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "foundry"))
sys.path.insert(0, str(ROOT / "tools"))

import query_gravity_methods as registry
from nr_container import seal_payload_sha256_v1
from tools.odyssey import arch_recognizer


def test_registry_is_queryable_and_source_bound():
    document = registry.load_registry()
    assert document["schema"] == "hawking.foundry.gravity_method_registry.v2"
    passports = document["organ_passports"]
    assert passports["schema"] == registry.PASSPORT_CATALOG_SCHEMA
    assert {entry["id"] for entry in passports["entries"]} >= {
        "flash.routed-moe.v1",
        "flash.ple.hashed-ngram.v1",
        "flash.decode-state-kv.v1",
        "flash.tokenization-sampling.v1",
    }
    assert {entry["id"] for entry in document["methods"]} >= {
        "M-stateful-repeated-accepted-decode",
        "M-teacher-route-union-compact-experts",
        "M-device-only-immutable-weight-release",
        "M-device-state-handoff",
        "M-source-ple-context-response-tangent-screen",
        "M-source-ple-additive-hash-factor-screen",
        "M-source-ple-native-context-state-bank-formula",
        "M-source-expert-lawful-nonlinear-atom-sharing-screen",
        "M-flash-router-source-output-fixed-int4",
        "M-flash-hyperconnection-symbolic-closure-screen",
    }
    assert all(entry["evidence"] and entry["implementation_owner"] for entry in document["methods"])

    accounting = next(entry for entry in document["methods"] if entry["id"] == "M-closed-nr-rate-accounting")
    adapter = accounting["automation"]["adapters"][0]
    assert adapter["runner"] == {
        "kind": "flash_complete_nr.v1",
        "path": "tools/flash_complete_nr.py",
        "timeout_seconds": 120,
    }
    assert adapter["verifier"] == {
        "kind": "nr_container_validate.v1",
        "path": "tools/nr_container.py",
        "symbol": "validate",
    }


def test_registry_has_one_machine_readable_map_of_living_owners():
    owners = registry.load_registry()["canonical_owners"]
    assert set(owners) >= {
        "discover_identify",
        "methods_representations",
        "account_verify",
        "execute_physical",
        "learn_odyssey",
    }

    def paths(value):
        if isinstance(value, str) and "/" in value and not value.startswith("A new "):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from paths(child)

    assert all((ROOT / path).exists() for path in paths(owners))


def test_live_execution_frontier_refuses_e13_until_its_sealed_controls_exist(tmp_path):
    checked = registry.validate_execution_frontier()
    assert checked["valid"] is True
    assert checked["e13_state"] == (
        "WITHHELD_BROADER_WIKIMEDIA_CONTROL_UNDERPOWERED__DESIGN_REVIEW_REQUIRED"
    )
    assert checked["e13_first_underpowered_stratum"] == "rare-latin-short"
    assert checked["e13_first_underpowered_eligible_rows"] == 330
    assert checked["e13_first_underpowered_required_rows"] == 512
    assert checked["e13_han_short_structural_rows"] == 5426
    assert checked["e06_state"] == "SEALED_FIXED_ROW_NORM_CONDITIONAL_INFORMATION_PREFILTER_NEGATIVE"
    assert checked["e02_state"] == (
        "SEALED_FIXED_TOP8_PAIR_REPLICATION_NEGATIVE__ORIGINAL_PAIR_BOUNDED_SIGNAL_ONLY"
    )
    assert checked["e02_original_calibration_pair_reproduced"] is True
    assert checked["e02_independent_replication_pair_count"] == 3
    assert checked["e02_independent_survivor_count"] == 2
    assert checked["e02_replicated"] is False
    assert checked["e06_conditional_saving_percent"] == pytest.approx(3.1584898630777993)
    assert checked["e06_conditional_net_saving_bits_after_auxiliary_cost"] == -74645.5625
    assert checked["e06_candidate_for_separate_review"] is False
    assert checked["e13_tensor_row_count"] == 248320
    assert checked["e13_tokenizer_addressable_count"] == 248077
    assert checked["e13_lexical_eligible_count"] == 248044
    assert checked["e13_tensor_only_tail_count"] == 243
    assert checked["e13_row_extraction_performed"] is False
    assert checked["e13_tensor_payload_bytes_read"] == 0

    stale = json.loads((ROOT / "tools/foundry/GRAVITY_METHOD_REGISTRY.json").read_text())
    stale["execution_frontier"]["withheld_without_native_bank"][0]["state"] = (
        "READY_CPU_STATIC_PREFILTER"
    )
    stale_path = tmp_path / "stale-registry.json"
    stale_path.write_text(json.dumps(stale))
    try:
        registry.validate_execution_frontier(stale_path)
    except ValueError as exc:
        assert "registry state" in str(exc)
    else:
        raise AssertionError("stale E13 readiness state was accepted")

    stale = json.loads((ROOT / "tools/foundry/GRAVITY_METHOD_REGISTRY.json").read_text())
    e06 = next(row for row in stale["execution_frontier"]["sealed_now"] if row["id"] == "E06")
    e06["state"] = "READY_CPU_STATIC_PREFILTER"
    stale_path = tmp_path / "stale-e06-registry.json"
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="E06 must remain sealed"):
        registry.validate_execution_frontier(stale_path)

    stale = json.loads((ROOT / "tools/foundry/GRAVITY_METHOD_REGISTRY.json").read_text())
    e02 = next(row for row in stale["execution_frontier"]["sealed_now"] if row["id"] == "E02")
    e02["state"] = "SOURCE_OUTPUT_SURVIVOR_REQUIRES_REPLICATION"
    stale_path = tmp_path / "stale-e02-registry.json"
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="E02 fixed top-eight replication"):
        registry.validate_execution_frontier(stale_path)


def test_moe_query_requires_all_architecture_tags():
    ids = {entry["id"] for entry in registry.select_methods({"moe", "routed_experts"})}
    assert "M-teacher-route-union-compact-experts" in ids
    assert "M-device-only-immutable-weight-release" not in ids


def test_metal_decoder_query_finds_residency_method():
    ids = {
        entry["id"]
        for entry in registry.select_methods({"metal", "autoregressive_decoder"})
    }
    assert ids == {
        "M-stateful-repeated-accepted-decode",
        "M-device-only-immutable-weight-release",
    }


def test_ple_function_screen_selects_only_after_semantic_controls_exist():
    compatible_tags = {
        "hashed_ngram",
        "sharded_lookup",
        "hyperconnection",
        "stateful_decoder",
        "source_ple_output_control",
        "reference_formula_parity",
    }
    compatible = {entry["id"] for entry in registry.select_methods(compatible_tags)}
    assert "M-source-ple-output-pq-sparse-repair-screen" in compatible
    assert "M-source-ple-address-generator-screen" in compatible
    incompatible = {
        entry["id"]
        for entry in registry.select_methods(compatible_tags - {"reference_formula_parity"})
    }
    assert "M-source-ple-output-pq-sparse-repair-screen" not in incompatible
    assert "M-source-ple-address-generator-screen" not in incompatible


def _ple_address_generator_observation(**overrides):
    observation = {
        "schema": registry.OBSERVATION_SCHEMA,
        "artifact_id": "FLASH_QWEN4_EXP_PLE",
        "objective_family": "representation_discriminator",
        "architecture_tags": [
            "flash_qwen4_exp",
            "hashed_ngram",
            "sharded_lookup",
            "hyperconnection",
            "stateful_decoder",
            "source_ple_output_control",
            "reference_formula_parity",
        ],
        "features": [
            "canonical_combined_lookup_row_address",
            "source_bound_lookup_access_trace",
            "sealed_pre_PLE_hidden_state",
            "reference_checked_PLE_formula_control",
            "source_independent_generator_candidate",
            "candidate_address_only_generator",
        ],
        "candidate_levers": ["ple_address_only_generator"],
    }
    observation.update(overrides)
    return observation


def test_ple_generator_automatic_assessment_selects_one_adapter_then_refuses_scarred_retry():
    selected = [
        decision["method"]
        for decision in registry.assess_methods(_ple_address_generator_observation())
        if decision["applicable"] and (decision["method"].get("automation") or {}).get("adapters")
    ]
    assert [method["id"] for method in selected] == ["M-source-ple-address-generator-screen"]
    refused = registry.assess_methods(
        _ple_address_generator_observation(
            features=_ple_address_generator_observation()["features"]
            + ["rerunning_smooth_address_only_generators_after_a_scoped_negative"]
        )
    )
    decision = next(
        row for row in refused if row["method"]["id"] == "M-source-ple-address-generator-screen"
    )
    assert decision["applicable"] is False
    assert decision["reasons"] == [
        "excluded features present: rerunning_smooth_address_only_generators_after_a_scoped_negative"
    ]


def _ple_additive_hash_observation(**overrides):
    observation = _ple_address_generator_observation(
        objective_family="function_generated_representation_discriminator",
        features=[
            "source_bound_lookup_access_trace",
            "sealed_pre_PLE_hidden_state",
            "reference_checked_PLE_formula_control",
            "source_independent_generator_candidate",
            "candidate_additive_hash_factor_generator",
        ],
        candidate_levers=["ple_additive_hash_factor_generator"],
    )
    observation.update(overrides)
    return observation


def test_ple_additive_hash_automatic_assessment_selects_once_then_refuses_scarred_retry():
    selected = [
        decision["method"]
        for decision in registry.assess_methods(_ple_additive_hash_observation())
        if decision["applicable"] and (decision["method"].get("automation") or {}).get("adapters")
    ]
    assert [method["id"] for method in selected] == [
        "M-source-ple-additive-hash-factor-screen"
    ]
    refused = registry.assess_methods(
        _ple_additive_hash_observation(
            features=_ple_additive_hash_observation()["features"]
            + ["rerunning_fixed_additive_hash_factor_after_a_scoped_negative"]
        )
    )
    decision = next(
        row for row in refused if row["method"]["id"] == "M-source-ple-additive-hash-factor-screen"
    )
    assert decision["applicable"] is False
    assert decision["reasons"] == [
        "excluded features present: rerunning_fixed_additive_hash_factor_after_a_scoped_negative"
    ]


def _ple_shard_local_pq_observation(**overrides):
    observation = {
        "schema": registry.OBSERVATION_SCHEMA,
        "artifact_id": "FLASH_QWEN4_EXP_PLE",
        "objective_family": "representation_discriminator",
        "architecture_tags": [
            "flash_qwen4_exp",
            "hashed_ngram",
            "sharded_lookup",
            "hyperconnection",
            "stateful_decoder",
            "source_ple_output_control",
            "reference_formula_parity",
        ],
        "features": [
            "source_bound_lookup_access_trace",
            "sealed_pre_ple_hidden_state",
            "reference_checked_PLE_formula_control",
            "globally_billed_lookup_table_geometry",
            "physical_numeric_PLE_shard_layout",
            "candidate_shard_local_pq",
        ],
        "candidate_levers": ["ple_shard_local_pq"],
    }
    observation.update(overrides)
    return observation


def test_ple_shard_local_pq_automatic_assessment_selects_once_then_refuses_scarred_retry():
    selected = [
        decision["method"]
        for decision in registry.assess_methods(_ple_shard_local_pq_observation())
        if decision["applicable"] and (decision["method"].get("automation") or {}).get("adapters")
    ]
    assert [method["id"] for method in selected] == [
        "M-source-ple-shard-local-pq-output-screen"
    ]
    refused = registry.assess_methods(
        _ple_shard_local_pq_observation(
            features=_ple_shard_local_pq_observation()["features"]
            + ["rerunning_fixed_shard_local_PQ_after_a_scoped_negative"]
        )
    )
    decision = next(
        row for row in refused if row["method"]["id"] == "M-source-ple-shard-local-pq-output-screen"
    )
    assert decision["applicable"] is False
    assert decision["reasons"] == [
        "excluded features present: rerunning_fixed_shard_local_PQ_after_a_scoped_negative"
    ]


def _ple_additive_pq_observation(**overrides):
    observation = {
        "schema": registry.OBSERVATION_SCHEMA,
        "artifact_id": "FLASH_QWEN4_EXP_PLE",
        "objective_family": "representation_discriminator",
        "architecture_tags": [
            "flash_qwen4_exp",
            "hashed_ngram",
            "sharded_lookup",
            "hyperconnection",
            "stateful_decoder",
            "source_ple_output_control",
            "reference_formula_parity",
        ],
        "features": [
            "source_bound_lookup_access_trace",
            "sealed_pre_ple_hidden_state",
            "reference_checked_PLE_formula_control",
            "globally_billed_lookup_table_geometry",
            "candidate_residual_additive_pq",
        ],
        "candidate_levers": ["ple_greedy_additive_pq"],
    }
    observation.update(overrides)
    return observation


def test_ple_additive_pq_automatic_assessment_selects_once_then_refuses_scarred_retry():
    selected = [
        decision["method"]
        for decision in registry.assess_methods(_ple_additive_pq_observation())
        if decision["applicable"] and (decision["method"].get("automation") or {}).get("adapters")
    ]
    assert [method["id"] for method in selected] == ["M-source-ple-additive-pq-output-screen"]
    refused = registry.assess_methods(
        _ple_additive_pq_observation(
            features=_ple_additive_pq_observation()["features"]
            + ["rerunning_fixed_greedy_additive_PQ_after_a_scoped_negative"]
        )
    )
    decision = next(
        row for row in refused if row["method"]["id"] == "M-source-ple-additive-pq-output-screen"
    )
    assert decision["applicable"] is False
    assert decision["reasons"] == [
        "excluded features present: rerunning_fixed_greedy_additive_PQ_after_a_scoped_negative"
    ]


def _ple_context_response_observation(**overrides):
    observation = {
        "schema": registry.OBSERVATION_SCHEMA,
        "artifact_id": "FLASH_QWEN4_EXP_PLE",
        "objective_family": "function_geometry_discriminator",
        "architecture_tags": [
            "flash_qwen4_exp",
            "hashed_ngram",
            "sharded_lookup",
            "hyperconnection",
            "stateful_decoder",
            "source_ple_output_control",
            "reference_formula_parity",
        ],
        "features": [
            "source_bound_lookup_access_trace",
            "sealed_pre_ple_hidden_state",
            "reference_checked_PLE_formula_control",
            "candidate_address_context_state_function",
            "historical_address_trace_with_verified_native_session_binding",
        ],
        "candidate_levers": ["ple_address_context_state_tangent"],
    }
    observation.update(overrides)
    return observation


def test_ple_context_tangent_automatic_assessment_selects_one_adapter_then_blocks_repetition():
    selected = [
        decision["method"]
        for decision in registry.assess_methods(_ple_context_response_observation())
        if decision["applicable"] and (decision["method"].get("automation") or {}).get("adapters")
    ]
    assert [method["id"] for method in selected] == [
        "M-source-ple-context-response-tangent-screen"
    ]
    refused = registry.assess_methods(
        _ple_context_response_observation(
            features=_ple_context_response_observation()["features"]
            + ["rerunning_zero_state_tangent_without_real_state_bank_followup"]
        )
    )
    decision = next(
        row
        for row in refused
        if row["method"]["id"] == "M-source-ple-context-response-tangent-screen"
    )
    assert decision["applicable"] is False
    assert decision["reasons"] == [
        "excluded features present: rerunning_zero_state_tangent_without_real_state_bank_followup"
    ]


def test_declared_sealed_json_adapter_is_shell_free_and_verifies_its_own_receipt(tmp_path):
    adapter = {
        "id": "test-screen",
        "runner": "tools/odyssey/ple_address_generator_screen.py",
        "argv": ["--output", "{artifact_path}", "--generation", "{generation}"],
        "verifier": "sealed JSON receipt",
        "verification": {"kind": "sealed_json", "schema": "test.v1", "status": "DONE"},
    }
    command = registry._adapter_command(
        adapter, {"generation": "test-generation"}, tmp_path / "artifact.json"
    )
    assert command[0] == sys.executable
    assert command[-4:] == ["--output", str(tmp_path / "artifact.json"), "--generation", "test-generation"]
    artifact = {"schema": "test.v1", "status": "DONE"}
    artifact["seal_sha256"] = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()
    verified, report = registry._verify_sealed_json_artifact(artifact, adapter)
    assert verified is True
    assert report["schema_matches"] and report["status_matches"] and report["seal_matches"]
    assert report["seal_algorithm"] == "python-json-sort-keys-default-separators-sha256.v1"

    compact = {"schema": "test.v1", "status": "DONE"}
    compact["seal_sha256"] = hashlib.sha256(
        json.dumps(compact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    verified, report = registry._verify_sealed_json_artifact(compact, adapter)
    assert verified is True
    assert report["seal_algorithm"] == "python-json-sort-keys-compact-utf8-sha256.v1"


def test_sealed_json_adapter_is_typed_bounded_and_binds_only_its_staging_path(tmp_path):
    adapter = {
        "id": "typed-sealed-screen",
        "architecture_tags": ["fixture"],
        "required_inputs": ["tools/foundry/GRAVITY_METHOD_REGISTRY.json"],
        "runner": "tools/odyssey/hyperconnection_closure_screen.py",
        "argv": ["--out", "{artifact_path}", "--generation", "{generation}"],
        "verifier": "sealed JSON fixture",
        "verification": {"kind": "sealed_json", "schema": "fixture.v1", "status": "DONE"},
    }
    parsed = registry._parse_execution_adapter(adapter, ROOT)
    staging = tmp_path / ".candidate.gravity-fixture.staging"
    command = registry._build_execution_command(
        parsed, {"generation": "fixture-generation"}, staging
    )

    assert parsed.runner_kind == "sealed_json_runner.v1"
    assert parsed.verifier_kind == "sealed_json.v1"
    assert parsed.timeout_seconds == registry.SEALED_JSON_ADAPTER_TIMEOUT_SECONDS
    assert command == [
        sys.executable,
        str(ROOT / "tools/odyssey/hyperconnection_closure_screen.py"),
        "--out", str(staging), "--generation", "fixture-generation",
    ]

    blocked = dict(adapter)
    blocked["argv"] = ["--state-bank", "{state_bank_manifest}", "--out", "{artifact_path}"]
    try:
        registry._parse_execution_adapter(blocked, ROOT)
    except ValueError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("an unbound native-state input placeholder was admitted")


def _accounting_observation(**overrides):
    observation = {
        "schema": registry.OBSERVATION_SCHEMA,
        "artifact_id": "FLASH_QWEN4_EXP",
        "objective_family": "representation_accounting",
        "architecture_tags": ["portable_nr", "flash_qwen4_exp"],
        "features": ["logical_weight_denominator", "declared_persistent_parts"],
        "candidate_levers": ["raw_weight_pq_vq_at_one_bit"],
        "invocation": {"target_ebpw": "0.5", "generation": "test-auto-reuse-v1"},
    }
    observation.update(overrides)
    return observation


def _valid_nr_document(**overrides):
    document = {
        "nr_version": "1.0.0",
        "status": "FIXTURE_OPEN_ACCOUNTING_ONLY",
        "semantic_provenance": {
            "parent_model": "fixture",
            "parent_revision": "fixture-revision",
            "parameter_count": 1,
        },
        "representation": {},
        "kernel_requirements": [],
    }
    document.update(overrides)
    unsigned = dict(document)
    unsigned.pop("seal_sha256", None)
    document["seal_sha256"] = seal_payload_sha256_v1(unsigned)
    return document


def _out_argument(command):
    return Path(command[command.index("--out") + 1])


def _completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _staging_leftovers(artifact):
    return list(artifact.parent.glob(f".{artifact.name}.gravity-*.staging"))


def _write_registry_copy(tmp_path, mutate):
    document = registry.load_registry()
    mutate(document)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def test_public_discovery_keeps_any_model_literal_while_eligibility_is_wildcard():
    discovery = {entry["id"] for entry in registry.select_methods({"portable_nr"})}
    assert "M-closed-nr-rate-accounting" not in discovery

    assessment = next(
        row
        for row in registry.assess_methods(_accounting_observation())
        if row["method"]["id"] == "M-closed-nr-rate-accounting"
    )
    assert assessment["applicable"] is True


def test_automatic_reuse_executes_real_compatible_flash_accounting_into_staging(tmp_path):
    receipt = tmp_path / "reuse.json"
    artifact = tmp_path / "candidate.nr.json"
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "EXECUTED_VERIFIED"
    assert result["schema"] == "hawking.gravity.method_execution.v2"
    assert result["selected_method"] == "M-closed-nr-rate-accounting"
    assert result["cheap_falsifier"]["passed"] is True
    assert len(result["cheap_falsifier"]["input_identities"]) == 2
    assert result["verification"]["kind"] == "nr_container_validate.v1"
    assert result["verification"]["executed"] is True
    assert result["verification"]["valid"] is True
    assert result["verification"]["seal_matches"] is True
    assert result["verification"]["scope"].startswith("NR portability/container")
    assert result["execution_contract"]["registry"]["sha256"]
    assert result["execution_contract"]["runner"]["implementation"]["sha256"]
    assert result["execution_contract"]["verifier"]["implementation"]["sha256"]
    assert result["artifact_destination"]["pre_run_state"]["state"] == "ABSENT"
    assert result["artifact_destination"]["post_promotion_state"]["state"] == "REGULAR"
    assert result["promotion"]["status"] == "PROMOTED_ATOMICALLY"
    assert result["scars"][0]["id"] == "raw_weight_pq_vq_at_one_bit"
    assert "does not establish" in result["claim_boundary"]
    assert artifact.is_file() and receipt.is_file()
    assert not _staging_leftovers(artifact)
    assert json.loads(receipt.read_text()) == result


def test_automatic_reuse_refuses_excluded_feature_without_invocation(tmp_path):
    receipt = tmp_path / "refusal.json"
    artifact = tmp_path / "must-not-exist.nr.json"
    result = registry.reuse_method(
        _accounting_observation(
            features=[
                "logical_weight_denominator",
                "declared_persistent_parts",
                "payload_only_accounting",
            ]
        ),
        receipt_path=receipt,
        artifact_path=artifact,
    )
    assert result["status"] == "REFUSED"
    assert result["invoked"] is False
    assert "no applicable method" in result["refusal"]
    decision = next(
        row for row in result["method_assessment"]
        if row["id"] == "M-closed-nr-rate-accounting"
    )
    assert decision["reasons"] == ["excluded features present: payload_only_accounting"]
    assert receipt.is_file()
    assert not artifact.exists()


def test_symlink_destination_is_refused_without_following_it(tmp_path, monkeypatch):
    artifact = tmp_path / "candidate.nr.json"
    target = tmp_path / "target.nr.json"
    receipt = tmp_path / "symlink.json"
    artifact.symlink_to(target)
    called = []

    def must_not_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("a destination symlink must fail before subprocess invocation")

    monkeypatch.setattr(registry.subprocess, "run", must_not_run)
    result = registry.reuse_method(
        _accounting_observation(),
        receipt_path=receipt,
        artifact_path=artifact,
    )

    assert result["status"] == "REFUSED_OUTPUT_DESTINATION_OCCUPIED_NO_WRITE"
    assert result["artifact_destination"]["pre_run_state"]["state"] == "SYMLINK"
    assert result["receipt_write"]["written"] is False
    assert not target.exists()
    assert not receipt.exists()
    assert not called


@pytest.mark.parametrize("occupied", ["receipt", "artifact", "both"])
def test_occupied_output_destinations_refuse_before_spawn_without_writing(
    tmp_path, monkeypatch, occupied
):
    receipt = tmp_path / "reuse.json"
    artifact = tmp_path / "candidate.nr.json"
    receipt_before = b"historical receipt bytes\n"
    artifact_before = b"historical artifact bytes\n"
    if occupied in {"receipt", "both"}:
        receipt.write_bytes(receipt_before)
    if occupied in {"artifact", "both"}:
        artifact.write_bytes(artifact_before)
    called = []

    def must_not_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("occupied final destinations must fail before subprocess invocation")

    monkeypatch.setattr(registry.subprocess, "run", must_not_run)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "REFUSED_OUTPUT_DESTINATION_OCCUPIED_NO_WRITE"
    assert result["invoked"] is False
    assert result["receipt_write"]["status"] == "NOT_WRITTEN_DESTINATION_PRECHECK_FAILED"
    assert result["receipt_write"]["written"] is False
    assert result["artifact_promotion"]["published"] is False
    assert not called
    if occupied in {"receipt", "both"}:
        assert receipt.read_bytes() == receipt_before
    else:
        assert not receipt.exists()
    if occupied in {"artifact", "both"}:
        assert artifact.read_bytes() == artifact_before
    else:
        assert not artifact.exists()


def test_same_output_destination_refuses_without_writing_or_spawning(tmp_path, monkeypatch):
    shared = tmp_path / "shared.json"
    called = []

    def must_not_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("colliding final destinations must fail before subprocess invocation")

    monkeypatch.setattr(registry.subprocess, "run", must_not_run)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=shared, artifact_path=shared
    )

    assert result["status"] == "REFUSED_OUTPUT_DESTINATION_COLLISION_NO_WRITE"
    assert result["invoked"] is False
    assert result["receipt_write"]["status"] == "NOT_WRITTEN_DESTINATION_COLLISION"
    assert result["receipt_write"]["written"] is False
    assert not shared.exists()
    assert not called


def test_missing_required_input_refuses_before_spawning(tmp_path, monkeypatch):
    def missing_input(document):
        method = next(item for item in document["methods"] if item["id"] == "M-closed-nr-rate-accounting")
        method["automation"]["adapters"][0]["required_inputs"] = [
            "receipts/headless/DOES_NOT_EXIST_FOR_GRAVITY_TEST.json"
        ]

    registry_copy = _write_registry_copy(tmp_path, missing_input)
    called = []

    def must_not_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("missing required input must fail before subprocess invocation")

    monkeypatch.setattr(registry.subprocess, "run", must_not_run)
    result = registry.reuse_method(
        _accounting_observation(),
        receipt_path=tmp_path / "missing-input.json",
        artifact_path=tmp_path / "candidate.nr.json",
        registry_path=registry_copy,
    )

    assert result["status"] == "REFUSED"
    assert result["cheap_falsifier"]["passed"] is False
    assert "DOES_NOT_EXIST_FOR_GRAVITY_TEST" in result["refusal"]
    assert not called


def test_zero_exit_without_fresh_staging_output_leaves_fresh_destination_absent(tmp_path, monkeypatch):
    receipt = tmp_path / "no-write.json"
    artifact = tmp_path / "candidate.nr.json"

    def no_write(command, **_kwargs):
        return _completed(command)

    monkeypatch.setattr(registry.subprocess, "run", no_write)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "OUTPUT_MISSING"
    assert result["artifact_destination"]["pre_run_state"]["state"] == "ABSENT"
    assert not artifact.exists()
    assert result["staging"]["cleanup"]["status"] == "NOT_PRESENT"
    assert not _staging_leftovers(artifact)


def test_verified_staging_output_promotes_fresh_destination_only_after_validation(tmp_path, monkeypatch):
    receipt = tmp_path / "promoted.json"
    artifact = tmp_path / "candidate.nr.json"
    document = _valid_nr_document(status="FIXTURE_FRESH_STAGED_OUTPUT")

    def write_stage(command, **_kwargs):
        stage = _out_argument(command)
        assert stage != artifact
        stage.write_text(json.dumps(document) + "\n")
        return _completed(command, stdout="fixture wrote staging")

    monkeypatch.setattr(registry.subprocess, "run", write_stage)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "EXECUTED_VERIFIED"
    assert result["artifact_destination"]["pre_run_state"]["state"] == "ABSENT"
    assert result["promotion"]["status"] == "PROMOTED_ATOMICALLY"
    assert result["receipt_write"]["status"] == "WRITTEN_ATOMICALLY_NO_REPLACE"
    assert json.loads(artifact.read_text()) == document
    assert result["new_evidence"]["identity"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert json.loads(receipt.read_text()) == result
    assert not _staging_leftovers(artifact)


@pytest.mark.parametrize(
    "payload",
    [
        "{not valid json",
        _valid_nr_document(representation={"kernel": "machine-bound"}),
        {**_valid_nr_document(), "seal_sha256": "0" * 64},
    ],
    ids=["malformed-json", "machine-specific-nr", "bad-seal"],
)
def test_invalid_staged_output_never_publishes(tmp_path, monkeypatch, payload):
    receipt = tmp_path / "invalid.json"
    artifact = tmp_path / "candidate.nr.json"

    def write_invalid_stage(command, **_kwargs):
        stage = _out_argument(command)
        stage.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return _completed(command)

    monkeypatch.setattr(registry.subprocess, "run", write_invalid_stage)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "VERIFICATION_FAILED"
    assert not artifact.exists()
    assert result["staging"]["cleanup"]["status"] == "REMOVED"
    assert not _staging_leftovers(artifact)


def test_nonzero_and_timeout_leave_fresh_destination_absent(tmp_path, monkeypatch):
    artifact = tmp_path / "candidate.nr.json"

    def nonzero(command, **_kwargs):
        return _completed(command, returncode=7, stderr="fixture failure")

    monkeypatch.setattr(registry.subprocess, "run", nonzero)
    failed = registry.reuse_method(
        _accounting_observation(), receipt_path=tmp_path / "nonzero.json", artifact_path=artifact
    )
    assert failed["status"] == "EXECUTION_FAILED"
    assert not artifact.exists()

    def timed_out(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 120, output="fixture output", stderr="fixture timeout")

    monkeypatch.setattr(registry.subprocess, "run", timed_out)
    timeout = registry.reuse_method(
        _accounting_observation(), receipt_path=tmp_path / "timeout.json", artifact_path=artifact
    )
    assert timeout["status"] == "EXECUTION_TIMED_OUT"
    assert not artifact.exists()
    assert not _staging_leftovers(artifact)


@pytest.mark.parametrize(
    ("section", "unadmitted_kind", "message"),
    [
        ("runner", "unadmitted_runner.v1", "runner kind is not supported"),
        ("verifier", "unadmitted_verifier.v1", "verifier kind is not supported"),
    ],
)
def test_registry_adapter_mismatch_refuses_before_spawning(
    tmp_path, monkeypatch, section, unadmitted_kind, message
):
    def mismatch(document):
        method = next(item for item in document["methods"] if item["id"] == "M-closed-nr-rate-accounting")
        method["automation"]["adapters"][0][section]["kind"] = unadmitted_kind

    registry_copy = _write_registry_copy(tmp_path, mismatch)
    called = []

    def must_not_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("unadmitted verifier must fail before subprocess invocation")

    monkeypatch.setattr(registry.subprocess, "run", must_not_run)
    result = registry.reuse_method(
        _accounting_observation(),
        receipt_path=tmp_path / "mismatch.json",
        artifact_path=tmp_path / "candidate.nr.json",
        registry_path=registry_copy,
    )

    assert result["status"] == "REFUSED"
    assert result["invoked"] is False
    assert message in result["refusal"]
    assert not called


def test_destination_changed_during_execution_is_not_clobbered(tmp_path, monkeypatch):
    receipt = tmp_path / "changed.json"
    artifact = tmp_path / "candidate.nr.json"
    document = _valid_nr_document(status="FIXTURE_STAGED_BUT_NOT_PROMOTED")

    def write_stage_and_change_destination(command, **_kwargs):
        _out_argument(command).write_text(json.dumps(document))
        artifact.write_text("concurrent destination\n")
        return _completed(command)

    monkeypatch.setattr(registry.subprocess, "run", write_stage_and_change_destination)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "PROMOTION_REFUSED_DESTINATION_CHANGED_NO_WRITE"
    assert result["invocation"]["outcome"] == "DESTINATION_CHANGED_DURING_EXECUTION_NO_REPLACE"
    assert artifact.read_text() == "concurrent destination\n"
    assert result["receipt_write"]["written"] is False
    assert not receipt.exists()
    assert not _staging_leftovers(artifact)


def test_no_replace_promotion_refuses_a_destination_created_after_last_state_check(
    tmp_path, monkeypatch
):
    receipt = tmp_path / "promotion-race.json"
    artifact = tmp_path / "candidate.nr.json"
    document = _valid_nr_document(status="FIXTURE_LINK_RACE")
    publish = registry._publish_staging_artifact

    def write_stage(command, **_kwargs):
        _out_argument(command).write_text(json.dumps(document))
        return _completed(command)

    def occupy_before_publish(staging_path, destination):
        artifact.write_text("concurrent destination after state check\n")
        return publish(staging_path, destination)

    monkeypatch.setattr(registry.subprocess, "run", write_stage)
    monkeypatch.setattr(registry, "_publish_staging_artifact", occupy_before_publish)
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )

    assert result["status"] == "PROMOTION_REFUSED_DESTINATION_OCCUPIED_NO_WRITE"
    assert result["invocation"]["outcome"] == "ATOMIC_NO_REPLACE_DESTINATION_OCCUPIED"
    assert result["artifact_promotion"]["published"] is False
    assert result["receipt_write"]["written"] is False
    assert artifact.read_text() == "concurrent destination after state check\n"
    assert not receipt.exists()
    assert not _staging_leftovers(artifact)


def test_receipt_publish_refuses_concurrent_destination_without_replacement(tmp_path, monkeypatch):
    receipt = tmp_path / "race.json"
    before = b"concurrent historical receipt\n"
    link = registry.os.link

    def occupy_receipt(source, destination):
        if Path(destination) == receipt:
            receipt.write_bytes(before)
        return link(source, destination)

    monkeypatch.setattr(registry.os, "link", occupy_receipt)
    result = registry._write_receipt(receipt, {"schema": "fixture.receipt.v1"})

    assert result["status"] == "NOT_WRITTEN_DESTINATION_OCCUPIED"
    assert result["written"] is False
    assert receipt.read_bytes() == before
    assert not list(tmp_path.glob(".race.json.gravity-*.receipt-staging"))


def test_historical_reuse_receipts_remain_v1_evidence():
    for name in [
        "GRAVITY_METHOD_REUSE_COMPATIBLE_20260912.json",
        "GRAVITY_METHOD_REUSE_INCOMPATIBLE_20260912.json",
    ]:
        document = json.loads((ROOT / "receipts/future" / name).read_text())
        assert document["schema"] == "hawking.gravity.method_reuse.v1"


def test_shared_legacy_seal_rule_accepts_the_historical_flash_accounting_artifact():
    document = json.loads(
        (ROOT / "receipts/headless/FLASH_NR_AUTOMATIC_REUSE_20260912.nr.json").read_text()
    )
    recorded_seal = document.pop("seal_sha256")
    assert seal_payload_sha256_v1(document) == recorded_seal


def _fixture_ple_static_traits(
    *, revision="fixture-revision", include_shard=True, include_header=True
):
    names = [
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.2.ple.proj.weight",
        "model.language_model.layers.2.ple.norm.weight",
    ]
    if not include_shard:
        names = names[1:]
    tensor_headers = None
    if include_header and include_shard:
        descriptor = {
            "name": names[0],
            "shard": "model-00001-of-00001.safetensors",
            "dtype": "BF16",
            "shape": [2, 160],
            "data_offsets": [0, 640],
        }
        tensor_headers = {
            "schema": registry.SELECTED_TENSOR_HEADERS_SCHEMA,
            "index_name": "model.safetensors.index.json",
            "index_sha256": "a" * 64,
            "headers": [{
                **descriptor,
                "header_descriptor_sha256": registry._canonical_json_sha256(descriptor),
            }],
            "header_bytes_read": 512,
            "tensor_payload_bytes_read": 0,
            "loaded_weights": False,
            "source_classification": "INDEX_BOUND_SELECTED_SAFETENSORS_HEADERS_ONLY",
        }
    return arch_recognizer.static_trait_projection(
        "fixture/flash",
        revision,
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "ple_layer_ids": [2],
                "ngram_size": 3,
                "hc_count": 4,
                "use_cache": True,
                "vocab_size": 248320,
                "bos_token_id": 248044,
            },
        },
        names,
        [{"organ": "recurrent_state"}, {"organ": "rmsnorm"}],
        tensor_headers=tensor_headers,
    )


def _rebind_fixture_header_identity(traits):
    headers = traits["selected_tensor_headers"]["headers"]
    for descriptor in headers:
        bound = {
            key: descriptor[key]
            for key in ("name", "shard", "dtype", "shape", "data_offsets")
        }
        descriptor["header_descriptor_sha256"] = registry._canonical_json_sha256(bound)
    traits["identity"]["tensor_header_manifest_sha256"] = registry._canonical_json_sha256(
        traits["selected_tensor_headers"]
    )


def _rebind_fixture_static_identity(traits):
    _rebind_fixture_header_identity(traits)
    traits["identity"]["tensor_manifest_sha256"] = hashlib.sha256(
        "\n".join(sorted(traits["tensor_names"])).encode("utf-8")
    ).hexdigest()
    selected = traits.get("selected_tensor_headers")
    if selected is not None:
        traits["identity"]["tensor_header_index_name"] = selected["index_name"]
        traits["identity"]["tensor_header_index_sha256"] = selected["index_sha256"]
        traits["identity"]["tensor_header_source_classification"] = selected[
            "source_classification"
        ]


def _append_fixture_header(traits, number, *, shard=None):
    name = f"zz.fixture.extra.{number:02d}.weight"
    descriptor = {
        "name": name,
        "shard": shard or "model-00001-of-00001.safetensors",
        "dtype": "BF16",
        "shape": [2, 160],
        "data_offsets": [number * 640, (number + 1) * 640],
    }
    traits["tensor_names"].append(name)
    traits["selected_tensor_headers"]["headers"].append(
        {
            **descriptor,
            "header_descriptor_sha256": registry._canonical_json_sha256(descriptor),
        }
    )
    traits["selected_tensor_headers"]["headers"].sort(key=lambda item: item["name"])


def _fixture_ple_observation():
    return {
        "schema": registry.OBSERVATION_SCHEMA,
        "artifact_id": "fixture/flash",
        "objective_family": "architecture_contract",
        "architecture_tags": [
            "flash_qwen4_exp",
            "hashed_ngram",
            "sharded_lookup",
            "hyperconnection",
            "stateful_decoder",
        ],
        "features": [
            "versioned_reference_algorithm",
            "addressable_concatenated_lookup_checkpoint",
            "bounded_token_sequence_or_explicit_control",
            "persistent_token_context",
            "source_bound_pre_layer_hidden_state",
            "versioned_PLE_reference_formula",
            "addressable_source_lookup_rows",
            "source_projection_norm_and_convolution_payloads",
        ],
        "candidate_levers": [],
    }


def _fixture_passport_registry(tmp_path, traits, *, scar_source_revision=None):
    def bind_fixture(document):
        passport = next(
            item
            for item in document["organ_passports"]["entries"]
            if item["id"] == "flash.ple.hashed-ngram.v1"
        )
        for predicate in passport["recognition"]["all"]:
            if predicate.get("kind") == "identity_equals":
                predicate["value"] = traits["identity"][predicate["path"]]
            elif predicate.get("kind") == "tensor_header_exact":
                headers = (traits.get("selected_tensor_headers") or {}).get("headers") or []
                if headers:
                    header = headers[0]
                    predicate.update(
                        name=header["name"], dtype=header["dtype"], shape=header["shape"]
                    )
        if scar_source_revision is not None:
            for scar in passport["scoped_scars"]:
                scar["scope"]["source_revision"] = scar_source_revision

    return _write_registry_copy(tmp_path, bind_fixture)


def _passport_row(result, passport_id):
    return next(row for row in result["passport_assessments"] if row["id"] == passport_id)


def _scar_candidate_scope(registry_path, passport_id, scar_id):
    document = json.loads(registry_path.read_text())
    passport = next(item for item in document["organ_passports"]["entries"] if item["id"] == passport_id)
    scar = next(item for item in passport["scoped_scars"] if item["id"] == scar_id)
    return {
        key: value
        for key, value in scar["scope"].items()
        if key != "source_revision"
    }


def test_passport_projects_static_traits_then_holds_or_applies_methods(tmp_path):
    traits = _fixture_ple_static_traits()
    registry_copy = _fixture_passport_registry(tmp_path, traits)
    result = registry.assess_passports(
        _fixture_ple_observation(),
        traits,
        evidence_handles={
            "source_ple_access_trace",
            "reference_ple_formula_oracle",
            "ple_inclusive_source_state_teacher",
        },
        path=registry_copy,
    )
    ple = _passport_row(result, "flash.ple.hashed-ngram.v1")
    assert result["schema"] == registry.PASSPORT_ASSESSMENT_SCHEMA
    assert result["static_only"] is True
    assert ple["source_identity_verified"] is True
    assert ple["static_projection_verified"] is True
    assert ple["status"] == "APPLY"
    assert any(
        row["id"] == "M-source-bound-ple-access-trace" and row["applicable"]
        for row in ple["method_assessment"]
    )

    withheld = registry.assess_passports(
        _fixture_ple_observation(), traits, path=registry_copy
    )
    assert _passport_row(withheld, "flash.ple.hashed-ngram.v1")["status"] == (
        "WITHHELD_MISSING_CONTROL"
    )


def test_passport_refuses_mismatched_identity_or_static_header_claim(tmp_path):
    traits = _fixture_ple_static_traits()
    registry_copy = _fixture_passport_registry(tmp_path, traits)
    changed_revision = _fixture_ple_static_traits(revision="different-revision")
    result = registry.assess_passports(
        _fixture_ple_observation(), changed_revision, path=registry_copy
    )
    row = _passport_row(result, "flash.ple.hashed-ngram.v1")
    assert row["status"] == "WITHHELD_SOURCE_IDENTITY_MISMATCH"
    assert row["source_identity_verified"] is False

    missing_shard = _fixture_ple_static_traits(include_shard=False)
    result = registry.assess_passports(
        _fixture_ple_observation(), missing_shard, path=registry_copy
    )
    assert _passport_row(result, "flash.ple.hashed-ngram.v1")["status"] == (
        "WITHHELD_SOURCE_IDENTITY_MISMATCH"
    )

    malformed = json.loads(json.dumps(traits))
    malformed["config"]["text_config"]["ple_layer_ids"] = []
    with pytest.raises(ValueError, match="config identity"):
        registry.assess_passports(_fixture_ple_observation(), malformed, path=registry_copy)


def test_shape_sensitive_passport_refuses_absent_wrong_or_tampered_header_witness(tmp_path):
    traits = _fixture_ple_static_traits()
    registry_copy = _fixture_passport_registry(tmp_path, traits)
    controls = {
        "source_ple_access_trace",
        "reference_ple_formula_oracle",
        "ple_inclusive_source_state_teacher",
    }
    accepted = registry.assess_passports(
        _fixture_ple_observation(), traits, evidence_handles=controls, path=registry_copy
    )
    assert _passport_row(accepted, "flash.ple.hashed-ngram.v1")["status"] == "APPLY"

    absent = _fixture_ple_static_traits(include_header=False)
    withheld = registry.assess_passports(
        _fixture_ple_observation(), absent, evidence_handles=controls, path=registry_copy
    )
    assert _passport_row(withheld, "flash.ple.hashed-ngram.v1")["status"] == (
        "WITHHELD_STATIC_TRAIT_MISMATCH"
    )

    wrong_shape = json.loads(json.dumps(traits))
    wrong_shape["selected_tensor_headers"]["headers"][0]["shape"] = [3, 160]
    _rebind_fixture_header_identity(wrong_shape)
    withheld = registry.assess_passports(
        _fixture_ple_observation(), wrong_shape, evidence_handles=controls, path=registry_copy
    )
    assert _passport_row(withheld, "flash.ple.hashed-ngram.v1")["status"] == (
        "WITHHELD_STATIC_TRAIT_MISMATCH"
    )

    stale_descriptor = json.loads(json.dumps(traits))
    stale_descriptor["selected_tensor_headers"]["headers"][0]["shape"] = [3, 160]
    stale_descriptor["identity"]["tensor_header_manifest_sha256"] = registry._canonical_json_sha256(
        stale_descriptor["selected_tensor_headers"]
    )
    with pytest.raises(ValueError, match="header descriptor hash does not bind"):
        registry.assess_passports(_fixture_ple_observation(), stale_descriptor, path=registry_copy)

    tampered = json.loads(json.dumps(traits))
    tampered["selected_tensor_headers"]["header_bytes_read"] = 513
    with pytest.raises(ValueError, match="selected tensor headers do not match"):
        registry.assess_passports(_fixture_ple_observation(), tampered, path=registry_copy)


def test_selected_header_witness_refuses_rebound_oversize_or_zero_accounting(tmp_path):
    traits = _fixture_ple_static_traits()
    registry_copy = _fixture_passport_registry(tmp_path, traits)

    too_many = json.loads(json.dumps(traits))
    for number in range(registry.SELECTED_TENSOR_HEADER_MAX_NAMES):
        _append_fixture_header(too_many, number)
    _rebind_fixture_static_identity(too_many)
    with pytest.raises(ValueError, match="bounded descriptor contract"):
        registry.assess_passports(_fixture_ple_observation(), too_many, path=registry_copy)

    too_many_shards = json.loads(json.dumps(traits))
    for number in range(registry.SELECTED_TENSOR_HEADER_MAX_SHARDS):
        _append_fixture_header(
            too_many_shards,
            number,
            shard=f"model-{number + 2:05d}-of-00009.safetensors",
        )
    _rebind_fixture_static_identity(too_many_shards)
    with pytest.raises(ValueError, match="bounded shard contract"):
        registry.assess_passports(_fixture_ple_observation(), too_many_shards, path=registry_copy)

    zero_header_bytes = json.loads(json.dumps(traits))
    zero_header_bytes["selected_tensor_headers"]["header_bytes_read"] = 0
    _rebind_fixture_static_identity(zero_header_bytes)
    with pytest.raises(ValueError, match="bounded header accounting"):
        registry.assess_passports(
            _fixture_ple_observation(), zero_header_bytes, path=registry_copy
        )


def test_ple_passport_binds_tensor_manifest_and_selected_index_identity(tmp_path):
    traits = _fixture_ple_static_traits()
    registry_copy = _fixture_passport_registry(tmp_path, traits)

    changed_tensor_names = json.loads(json.dumps(traits))
    changed_tensor_names["tensor_names"].append("model.language_model.extra.weight")
    _rebind_fixture_static_identity(changed_tensor_names)
    result = registry.assess_passports(
        _fixture_ple_observation(), changed_tensor_names, path=registry_copy
    )
    assert _passport_row(result, "flash.ple.hashed-ngram.v1")["status"] == (
        "WITHHELD_SOURCE_IDENTITY_MISMATCH"
    )

    changed_index = json.loads(json.dumps(traits))
    changed_index["selected_tensor_headers"]["index_sha256"] = "c" * 64
    _rebind_fixture_static_identity(changed_index)
    result = registry.assess_passports(
        _fixture_ple_observation(), changed_index, path=registry_copy
    )
    assert _passport_row(result, "flash.ple.hashed-ngram.v1")["status"] == (
        "WITHHELD_SOURCE_IDENTITY_MISMATCH"
    )


def test_passport_scars_are_exact_scope_predicates_not_text_transfer(tmp_path):
    traits = _fixture_ple_static_traits()
    registry_copy = _fixture_passport_registry(
        tmp_path, traits, scar_source_revision=traits["identity"]["revision"]
    )
    controls = {
        "source_ple_access_trace",
        "reference_ple_formula_oracle",
        "ple_inclusive_source_state_teacher",
    }
    exact_scope = _scar_candidate_scope(
        registry_copy,
        "flash.ple.hashed-ngram.v1",
        "flash.ple.smooth-address-only-generator",
    )
    exact = registry.assess_passports(
        _fixture_ple_observation(),
        traits,
        evidence_handles=controls,
        candidate_scope=exact_scope,
        path=registry_copy,
    )
    row = _passport_row(exact, "flash.ple.hashed-ngram.v1")
    assert row["status"] == "SCAR_SKIP"
    assert row["matched_scars"][0]["id"] == "flash.ple.smooth-address-only-generator"

    changed_contract = registry.assess_passports(
        _fixture_ple_observation(),
        traits,
        evidence_handles=controls,
        candidate_scope={**exact_scope, "tested_contract": "different-generator-contract"},
        path=registry_copy,
    )
    assert _passport_row(changed_contract, "flash.ple.hashed-ngram.v1")["status"] == "APPLY"

    missing_rate_scope = dict(exact_scope)
    del missing_rate_scope["rate_scope"]
    missing_rate = registry.assess_passports(
        _fixture_ple_observation(),
        traits,
        evidence_handles=controls,
        candidate_scope=missing_rate_scope,
        path=registry_copy,
    )
    assert _passport_row(missing_rate, "flash.ple.hashed-ngram.v1")["status"] == "APPLY"

    changed_control = registry.assess_passports(
        _fixture_ple_observation(),
        traits,
        evidence_handles=controls,
        candidate_scope={**exact_scope, "control_sha256": "f" * 64},
        path=registry_copy,
    )
    assert _passport_row(changed_control, "flash.ple.hashed-ngram.v1")["status"] == "APPLY"

    withheld = registry.assess_passports(
        _fixture_ple_observation(),
        traits,
        candidate_scope=exact_scope,
        path=registry_copy,
    )
    assert _passport_row(withheld, "flash.ple.hashed-ngram.v1")["status"] == (
        "SCAR_RECORDED__WITHHELD_MISSING_CONTROL"
    )

    with pytest.raises(ValueError, match="cannot override"):
        registry.assess_passports(
            _fixture_ple_observation(),
            traits,
            evidence_handles=controls,
            candidate_scope={"source_revision": "other-source"},
            path=registry_copy,
        )


def test_scoped_scar_receipt_digests_are_registry_bound(tmp_path):
    traits = _fixture_ple_static_traits()

    def corrupt(document):
        passport = next(
            item
            for item in document["organ_passports"]["entries"]
            if item["id"] == "flash.ple.hashed-ngram.v1"
        )
        passport["scoped_scars"][0]["evidence_sha256"] = "0" * 64

    registry_copy = _write_registry_copy(tmp_path, corrupt)
    with pytest.raises(ValueError, match="does not bind its evidence file"):
        registry.load_registry(registry_copy)


def test_flash_paid_negative_scars_are_protocol_bound():
    document = registry.load_registry()
    expected = {
        "flash.routed-moe.v1": {
            "flash.routed-moe.fixed-shared-vector-code",
            "flash.routed-moe.fixed-shared-input-basis",
            "flash.routed-moe.fixed-pairwise-atom-sharing",
        },
        "flash.ple.hashed-ngram.v1": {
            "flash.ple.uniform-pq-fixed-sparse-repair",
            "flash.ple.fixed-shard-local-pq",
            "flash.ple.fixed-greedy-additive-pq",
            "flash.ple.smooth-address-only-generator",
            "flash.ple.fixed-additive-hash-address-prior",
        },
    }
    passports = {passport["id"]: passport for passport in document["organ_passports"]["entries"]}
    for passport_id, scar_ids in expected.items():
        scars = {scar["id"]: scar for scar in passports[passport_id]["scoped_scars"]}
        assert scar_ids <= set(scars)
        for scar_id in scar_ids:
            scope = scars[scar_id]["scope"]
            assert {
                "source_revision",
                "organ_scope",
                "representation_family",
                "tested_contract",
                "rate_scope",
                "control_id",
                "control_sha256",
            } <= set(scope)
