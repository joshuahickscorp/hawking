"""Lifecycle is derived from disk, and the derivation had to be corrected once.

Counting only .safetensors classified ten complete specimens as DISCOVERED -
evo2 ships .pt, boltz .ckpt, mamba3 .bin, Wan .pth. The registry's own
classifier was wrong before any specimen was.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import specimen_registry as sr


def test_the_registry_is_built_from_disk_not_declared():
    rows = sr.registry()
    assert len(rows) > 40, "the lake has a real library"
    assert all(Path(r["path"]).exists() for r in rows)


def test_an_unmounted_lake_refuses_rather_than_reporting_zero(monkeypatch):
    monkeypatch.setattr(sr, "LAKE", Path("/no/such/volume"))
    with pytest.raises(sr.RegistryRefused, match="is not attached"):
        sr.registry()


def test_every_state_is_a_declared_lifecycle_value():
    assert all(r["lifecycle"] in sr.LIFECYCLE for r in sr.registry())


def test_every_state_carries_the_evidence_that_derived_it():
    for r in sr.registry():
        assert r["lifecycle_derived_from"], r["id"]


def test_sealed_requires_a_manifest_with_a_sha_and_a_byte_count():
    for r in sr.registry():
        if r["lifecycle"] in ("SEALED_SOURCE", "FINGERPRINTED"):
            assert r["source_bytes"], r["id"]
            assert r["acquired_at"], r["id"]


def test_fingerprinted_requires_an_architecture():
    for r in sr.registry():
        if r["lifecycle"] == "FINGERPRINTED":
            assert r["architecture"]["model_type"], r["id"]


def test_non_safetensors_layouts_are_not_called_incomplete():
    """The classifier bug this test exists to prevent recurring."""
    assert ".pt" in sr.WEIGHT_SUFFIXES
    assert ".ckpt" in sr.WEIGHT_SUFFIXES
    assert ".bin" in sr.WEIGHT_SUFFIXES
    ids = {r["id"] for r in sr.registry() if r["lifecycle"] == "DISCOVERED"}
    assert not any("mamba3" in i for i in ids), \
        "mamba3 ships pytorch_model.bin and is complete"
    assert not any("evo2_7b" in i for i in ids), \
        "evo2_7b ships evo2_7b.pt and is complete"


def test_the_weaker_completeness_check_is_labelled_as_weaker():
    strengths = {r["shards"]["check_strength"] for r in sr.registry()
                 if r["shards"].get("check_strength")}
    assert "INDEX_MATCHED" in strengths
    b = sr.seal_backlog()
    assert set(b["check_strength_mix"]) == {"INDEX_MATCHED", "WEIGHTS_PRESENT_ONLY"}
    assert "which are worth queueing first" in b["not_every_check_is_equally_strong"]


def test_the_seal_backlog_is_the_finding_and_it_is_large():
    b = sr.seal_backlog()
    assert b["n_complete_unsealed"] > b["n_sealed"], \
        "most of the library is complete and unsealed"
    assert "NOT re-downloads" in b["statement"]
    assert "mutation writer pretending to be an observer" in \
        b["this_module_does_not_seal_them"]


def test_a_directory_under_specimens_is_not_evidence_of_completeness():
    inc = sr.build()["incomplete_in_the_specimens_directory"]
    assert inc["ids"], "some sit in specimens/ with a config and no weights"
    assert "Directory placement is not evidence of completeness" in inc["statement"]


def test_load_cost_is_admitted_unknown_rather_than_guessed():
    for r in sr.registry():
        assert r["measured_load_seconds"] is None
        assert "would be worse than one that says so" in \
            r["load_cost_is_unknown_because"]


def test_states_this_module_cannot_derive_are_named():
    n = sr.build()["states_not_derivable_here"]
    assert "NX_AVAILABLE" in n and "UMA_RESIDENT" in n
    assert all(v for v in n.values()), "each must say why"


def test_schedulable_is_only_sealed_material():
    ids = {r["id"] for r in sr.schedulable()}
    for r in sr.registry():
        if r["id"] in ids:
            assert r["lifecycle"] in ("SEALED_SOURCE", "FINGERPRINTED")


def test_architecture_families_keep_unknown_separate():
    fam = sr.architecture_families()
    assert len(fam) > 20, "the library is architecturally diverse"
    assert "UNKNOWN" in fam, "unknown must not be folded into a guess"


def test_ids_are_unique():
    ids = [r["id"] for r in sr.registry()]
    assert len(ids) == len(set(ids)), "one row per specimen"


def test_a_stale_partial_directory_is_reported_not_silently_deduped():
    sh = sr.shadowed()
    assert sh, "one specimen is shadowed by an empty partial directory today"
    for row in sh:
        assert row["specimens_bytes"] > 0
        assert row["reading"]
    frag = [r for r in sh if r["partial_is_a_fragment"]]
    assert frag, "the shadow found is a leftover fragment"
    assert frag[0]["partial_fraction_of_specimen"] < 0.01
    assert "leftover fragment of a finished download" in frag[0]["reading"]
    assert "believe a download is in flight" in frag[0]["reading"]


def test_the_shadowed_specimen_is_not_reported_as_downloading():
    shadowed_ids = {r["id"] for r in sr.shadowed()}
    for r in sr.registry():
        if r["id"] in shadowed_ids:
            assert r["lifecycle"] != "DOWNLOADING", \
                "a complete specimen must not read as in-flight"
