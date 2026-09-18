"""Current-source reconciliation for the NX kernel binding boundary."""

from tools import nx_genome
from tools.headless import noetic_kernel_census


def test_nx_binding_matches_the_current_census_extractors():
    """The two admitted source extractors must agree before a binding is sealed."""
    bound, declared_count = nx_genome.bound_kernels()
    census_bound, census_declared_count = noetic_kernel_census.current_source_binding()

    assert bound == census_bound
    assert declared_count == census_declared_count
    assert 0 < len(bound) <= declared_count


def test_nx_seal_records_live_binding_counts_without_hardware_qualification(monkeypatch):
    """Seal metadata is source evidence only; hardware qualification stays separate."""
    monkeypatch.setattr(
        nx_genome,
        "machine_genome",
        lambda: {
            "chipset": "test",
            "gpu_cores": 1,
            "unified_memory_bytes": 1,
            "metal_family": "test",
            "measured_roof_gb_s": 0.0,
            "genome_digest": "test-genome",
        },
    )
    sealed = nx_genome.seal()
    binding = sealed["kernel_binding"]
    bound, declared_count = nx_genome.bound_kernels()

    assert binding["dispatched"] == bound
    assert binding["count"] == len(bound)
    assert binding["declared_in_tree"] == declared_count
    assert "historical counts are not treated as invariants" in binding["extraction"]


def test_current_census_has_source_facts_for_every_bound_kernel():
    """All current literal-bound kernels are classified without guessing."""
    receipt, _ = noetic_kernel_census.build()
    reconciliation = receipt["reconciliation"]

    assert reconciliation["live_bound"] == 98
    assert reconciliation["live_declared"] == 818
    assert reconciliation["dispatched_reconstructs_dense"] == {"NO": 98}
    assert not [
        row for row in receipt["dispatched"] if row.get("reconstructs_dense") == "UNKNOWN"
    ]
    assert reconciliation["historical_drift"] == {"bound": True, "declared": True}
