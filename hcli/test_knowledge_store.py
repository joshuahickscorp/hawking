"""Compounding memory must forget nothing AND overgeneralise nothing.

S008 section 12 names both failures. The store is tested against both, because
a prior that is wrong is worse than a prior that is missing: every later
specimen inherits it.
"""
import pytest

from hcli.knowledge_store import (
    FAMILY_PRIOR, MACHINE_LAW, SPECIMEN, TRANSFERABLE_LAW, Overgeneralisation,
    load, priors_for, validate,
)

HOST = "Apple M3 Ultra 96GB, MLX Metal"


def _entry(**kw):
    base = {"id": "T", "scope": SPECIMEN, "claim": "c", "measurement": "m",
            "reopen_condition": "r", "oracle_strength": "s",
            "sources": [{"specimen": "O003"}]}
    base.update(kw)
    return base


def test_a_law_from_one_specimen_is_REFUSED():
    """The whole point. One model's lesson is not a universal law."""
    with pytest.raises(Overgeneralisation, match="1 specimen"):
        validate(_entry(scope=TRANSFERABLE_LAW))


def test_a_law_from_two_specimens_is_allowed():
    validate(_entry(scope=TRANSFERABLE_LAW,
                    sources=[{"specimen": "O003"}, {"specimen": "O005"}]))


def test_family_prior_must_state_its_architecture_condition():
    with pytest.raises(ValueError, match="architecture_condition"):
        validate(_entry(scope=FAMILY_PRIOR))


def test_every_entry_needs_a_reopen_condition():
    """A finding that cannot say what would overturn it is an opinion."""
    with pytest.raises(ValueError, match="reopen_condition"):
        validate(_entry(reopen_condition=""))


def test_every_entry_needs_a_measurement_and_an_oracle_strength():
    with pytest.raises(ValueError, match="measurement"):
        validate(_entry(measurement=""))
    with pytest.raises(ValueError, match="oracle_strength"):
        validate(_entry(oracle_strength=""))


def test_specimen_findings_do_NOT_propagate():
    """K006 and K007 are true of O003 and carry nothing forward."""
    ids = {e["id"] for e in priors_for("kimi_vl", host=HOST)}
    assert "K006" not in ids and "K007" not in ids
    assert {e["id"] for e in load() if e["scope"] == SPECIMEN} >= {"K006", "K007"}


def test_a_dense_body_does_not_inherit_a_MoE_conditioned_law():
    """K004 is about top-k expert gather. A llama has none, and inheriting it
    would aim its optimisation at a stage it does not have."""
    ids = {e["id"] for e in priors_for("llama", host=HOST)}
    assert "K004" not in ids, ids
    assert "K001" not in ids and "K002" not in ids


def test_an_MoE_body_DOES_inherit_the_MoE_priors():
    ids = {e["id"] for e in priors_for("qwen3_moe", host=HOST)}
    assert {"K001", "K002", "K004"} <= ids, ids


def test_host_any_reaches_every_host():
    ids = {e["id"] for e in priors_for("llama", host="some other machine")}
    assert "K005" in ids and "K003" not in ids


def test_the_store_on_disk_validates():
    for e in load():
        validate(e)
