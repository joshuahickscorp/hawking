"""P10 NX implementation contract v2: the load-time genome gate must recompute the
sealed digest from the sealed facts instead of replaying a stored digest as authority.

Scope: tools/nx_genome.py genome_matches / genome_digest. No release, no hardware
qualification, no phase marker is claimed by this test.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NX_GENOME = ROOT / "tools" / "nx_genome.py"


def _load_nx_genome():
    spec = importlib.util.spec_from_file_location("nx_genome_under_test", NX_GENOME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nx_genome = _load_nx_genome()

FACTS = {
    "chipset": "Apple M3 Max",
    "gpu_cores": 40,
    "unified_memory_bytes": 137438953472,
    "metal_family": "Metal 3",
    "measured_roof_gb_s": 595.9,
}


def _sealed(**overrides):
    facts = dict(FACTS)
    facts.update(overrides)
    facts["genome_digest"] = nx_genome.genome_digest(facts)
    return facts


def test_identical_machine_is_admitted():
    sealed = _sealed()
    assert nx_genome.genome_matches(sealed, dict(FACTS)) is True


@pytest.mark.parametrize("key,drifted", [
    ("chipset", "Apple M2 Ultra"),
    ("gpu_cores", 30),
    ("unified_memory_bytes", 68719476736),
    ("metal_family", "Metal 2"),
    ("measured_roof_gb_s", 512.0),
])
def test_any_fact_drift_is_refused(key, drifted):
    sealed = _sealed()
    current = dict(FACTS)
    current[key] = drifted
    assert nx_genome.genome_matches(sealed, current) is False


def test_seal_whose_stored_digest_disagrees_with_its_facts_is_refused():
    sealed = _sealed()
    sealed["genome_digest"] = "0" * 64
    assert nx_genome.genome_matches(sealed, dict(FACTS)) is False


def test_seal_without_digest_is_refused():
    sealed = _sealed()
    del sealed["genome_digest"]
    assert nx_genome.genome_matches(sealed, dict(FACTS)) is False


def test_non_dict_inputs_are_refused():
    assert nx_genome.genome_matches(None, dict(FACTS)) is False
    assert nx_genome.genome_matches(_sealed(), None) is False


def test_digest_projection_ignores_provenance_fields():
    sealed = _sealed()
    current = dict(FACTS)
    current["roof_provenance"] = "different note, same machine"
    assert nx_genome.genome_matches(sealed, current) is True