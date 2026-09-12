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
