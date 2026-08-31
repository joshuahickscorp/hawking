"""A scar the index cannot see does not prune anything.

The model-bearing torture (G015) ran 30 minutes, made 88 model calls, generated
14,283 tokens and launched NOTHING. Its own summary named the cause: choose()
advertised WU.DEAD.mlp_function_replacement as the scripted policy "because
negative_index.refuse_if_dead does not key MLP_FUNCTION_REPLACEMENT_CLOSED". The
school had been closed hours earlier; the index could not see it, so the dead
option stayed on the menu 45 refills running and the resident kept picking it.

Root cause: SKIP_PREFIXES excludes receipts/future/ from the discovery sweep, and
only CAMPAIGN_SCARS.json and TPS_FALSIFICATIONS.jsonl were seeded back. Every
science scar this campaign landed was invisible to its own pruner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.future import negative_index as ni  # noqa: E402

WAVE_DEAD = (
    "MLP_FUNCTION_REPLACEMENT",
    "MONARCH",
    "BUTTERFLY",
    "FACTORIZE_THE_FACTORS",
    "PRODUCT_DICTIONARY",
    "CONDITIONAL_PROGRAM",
    "GENERATED_BLOCK",
    "NONLINEAR_GENERATOR",
)


def test_every_school_this_wave_closed_is_refused():
    for family in WAVE_DEAD:
        got = ni.refuse_if_dead(
            {"hypothesis_family": family, "organ": "mlp", "model": "qwen3.8-27b"}
        )
        assert got, f"{family} is closed on disk but refuse_if_dead does not prune it"
        assert got["refused"] is True


def test_the_umbrella_family_the_resident_actually_picked_is_refused():
    """WU.DEAD.mlp_function_replacement, lowercase, as choose() advertised it."""
    got = ni.refuse_if_dead(
        {"hypothesis_family": "mlp_function_replacement", "organ": "mlp",
         "model": "qwen3.8-27b"}
    )
    assert got, "the exact family the torture kept selecting is still not pruned"


def test_landed_science_receipts_reach_the_index():
    import json

    doc = json.loads(Path(ni.build()).read_text())
    sources = {str(s.get("source_path")) for s in doc["scars"]}
    for rel in (
        "receipts/future/MLP_STRUCTURED_OPERATOR.json",
        "receipts/future/MLP_NONLINEAR_PROGRAM.json",
    ):
        assert rel in sources, f"{rel} declares scars that never reach the index"


def test_a_live_school_is_not_pruned():
    """The pruner must not become a wall. Nothing closed full-width structured
    operators as a CLASS beyond this parent, and an unrelated organ stays open."""
    assert ni.refuse_if_dead(
        {"hypothesis_family": "gqa_kv_state_compression", "organ": "gqa",
         "model": "qwen3.8-27b"}
    ) is None
