"""Device Ascension pins. This module had NO tests, which is exactly why a dict class
attribute inside a @dataclass shipped broken and 46 unrelated tests still passed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import device_ascension as da  # noqa: E402


def test_the_module_actually_imports_and_constructs():
    """The regression that got through: @dataclass rejects a mutable class attribute."""
    assert da.Ascension({"soc": "X"}) is not None


def test_a_profile_cannot_be_sealed_without_the_stages_it_depends_on():
    a = da.Ascension({"soc": "X"})
    a.record("sustained_qualification", {"passed": True})
    a.not_run("concurrency_sweep", "not run")
    assert a.seal("SUSTAINED", {}, "EXACT_MACHINE")["status"] == "SEALED"
    m = a.seal("MAX_THROUGHPUT", {}, "EXACT_MACHINE")
    assert m["status"] == "PROVISIONAL"
    assert "concurrency_sweep" in m["why"]


def test_sustained_evidence_is_required_for_a_production_seal():
    a = da.Ascension({"soc": "X"})
    a.not_run("sustained_qualification", "microbenchmark only")
    assert a.seal("SUSTAINED", {}, "EXACT_MACHINE")["status"] == "PROVISIONAL"


def test_failed_sustained_does_not_seal():
    a = da.Ascension({"soc": "X"})
    a.record("sustained_qualification", {"passed": False})
    assert a.seal("SUSTAINED", {}, "EXACT_MACHINE")["status"] == "PROVISIONAL"


def test_an_unnamed_profile_is_refused():
    a = da.Ascension({"soc": "X"})
    with pytest.raises(ValueError, match="not a named profile"):
        a.seal("FAST_MODE", {}, "EXACT_MACHINE")


def test_the_two_vocabularies_are_distinct():
    """§26 tuning scope and §80 knowledge level are different questions."""
    import receipt
    a = da.Ascension({"soc": "X"})
    a.record("sustained_qualification", {"passed": True})
    with pytest.raises(ValueError, match="tuning scope"):
        a.seal("SUSTAINED", {}, "INSTANCE")          # a §80 level, not a §26 scope
    assert "EXACT_MACHINE" not in receipt.KNOWLEDGE_LEVELS
    assert "INSTANCE" not in da.TUNING_SCOPES


def test_a_stage_outside_the_eleven_is_refused():
    a = da.Ascension({"soc": "X"})
    with pytest.raises(ValueError, match="not one of the eleven"):
        a.record("vibes", {})
