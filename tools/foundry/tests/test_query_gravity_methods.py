from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "foundry"))

import query_gravity_methods as registry


def test_registry_is_queryable_and_source_bound():
    document = registry.load_registry()
    assert document["schema"] == "hawking.foundry.gravity_method_registry.v2"
    assert {entry["id"] for entry in document["methods"]} >= {
        "M-stateful-repeated-accepted-decode",
        "M-teacher-route-union-compact-experts",
        "M-device-only-immutable-weight-release",
        "M-device-state-handoff",
    }
    assert all(entry["evidence"] and entry["implementation_owner"] for entry in document["methods"])


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


def test_automatic_reuse_executes_and_verifies_compatible_flash_accounting(tmp_path):
    receipt = tmp_path / "reuse.json"
    artifact = tmp_path / "candidate.nr.json"
    result = registry.reuse_method(
        _accounting_observation(), receipt_path=receipt, artifact_path=artifact
    )
    assert result["status"] == "EXECUTED_VERIFIED"
    assert result["selected_method"] == "M-closed-nr-rate-accounting"
    assert result["cheap_falsifier"]["passed"] is True
    assert result["verification"] == {
        "verifier": "tools/nr_container.py:validate",
        "valid": True,
        "problems": [],
        "seal_matches": True,
    }
    assert result["scars"][0]["id"] == "raw_weight_pq_vq_at_one_bit"
    assert artifact.is_file() and receipt.is_file()


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
