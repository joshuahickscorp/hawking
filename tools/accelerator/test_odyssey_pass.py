"""Odyssey pass: which runtime can actually execute a lake specimen.

Split out because five G048 receipts justified weight-space grading with a
sentence that conflated Hawking's Rust body with mlx_lm.

2026-08-25, G061: the live half of this file used to rebuild ONE specimen --
Falcon-H1, the only one of the four that was never broken -- and skip when its
staged config was absent. Both halves of that were wrong. It is now parameterised
over ALL FOUR, a missing staged config is a FAILURE rather than a skip, and the
repair that makes the fourth build has a control arm that must raise without it.
"""
import functools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import odyssey_pass as op

# THE ONLY PERMITTED SKIP IN THIS FILE, and it is loud on purpose: if mlx is absent
# the runtime claim is UNMEASURED on this machine, which is a different thing from
# passing. Everything else -- a missing staged config above all -- is a failure.
try:
    import mlx.core as mx
    _MLX_ERR = None
except Exception as exc:                     # noqa: BLE001 -- absence is the answer
    mx = None
    _MLX_ERR = f"{type(exc).__name__}: {exc}"

requires_mlx = pytest.mark.skipif(
    mx is None,
    reason=("mlx.core DID NOT IMPORT under " + sys.executable + " (" + str(_MLX_ERR) +
            ") -- the four-specimen runtime claim is UNMEASURED here, NOT passing. "
            "The interpreter that carries mlx on this machine is "
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"))


# --- the blocker was misattributed, and a test is what stops it coming back ------

def test_mlx_lm_CAN_read_every_lake_specimen():
    """Five receipts justified weight-space grading with 'no runtime here executes a
    Qwen3-MoE, Falcon or Mamba forward pass'. mlx_lm is already a hard dependency of
    every AIR kernel in this program and carries all four classes. If this test ever
    fails the claim becomes true again -- until then it is false."""
    cov = op.runtime_coverage()
    assert cov["mlx_lm_covers_all_lake_specimens"] is True, cov["mlx_lm_module_present"]
    # ...AND THAT IS THE WEAKER CLAIM. Measured one block later: 3 of 4 actually
    # build and run; qwen3_vl_moe imports and does not construct. The import check
    # was right about three specimens and WRONG ABOUT ONE.
    assert op.execution_coverage()["built_and_ran"] == 4   # was 3 until prepare_config


def test_HAWKING_S_OWN_RUNTIME_still_covers_none_of_them():
    """The other half, and it is why the sentence sounded right: the Rust body really
    does dispatch only on llama/mistral/qwen2 families. Both halves are true and the
    receipts stated one."""
    assert op.runtime_coverage()["hawking_rust_covers_any_lake_specimen"] is False
    assert "qwen3_moe" not in op.HAWKING_RUST_ARCHS
    assert "falcon_h1" not in op.HAWKING_RUST_ARCHS


def test_the_gate_names_STORAGE_not_a_missing_reader():
    """A gap named 'no reader' invites weeks of Rust; a gap named 'contended bus'
    invites a quiesced window. Naming the wrong one parks the obligation forever."""
    cov = op.runtime_coverage()
    assert cov["blocked_on"] == "FAST_LOCAL_STORAGE"
    assert "reader" in cov["blocked_on_detail"]      # says what it is NOT


def test_IMPORTS_AND_EXECUTES_NOW_AGREE_AFTER_THE_ONE_FIELD_REPAIR():
    """The whole reason the stronger check had to be run. If these two ever agree
    again the disagreement has been fixed or reintroduced, and either way somebody
    should look rather than inheriting a stale claim."""
    imports = op.runtime_coverage()["mlx_lm_module_present"]
    ex = op.execution_coverage()["measured"]
    assert all(imports.values()), imports
    # The disagreement is CLOSED: prepare_config supplies the one field the wrapper
    # does not inherit. The gap text stays because the LIBRARY still has it -- what
    # changed is that Hawking carries the repair, and the two are different facts.
    assert [k for k, v in ex.items() if not v["builds"]] == []
    assert "tie_word_embeddings" in op.VL_GAP
    assert "NOT a library patch" in op.VL_GAP_CLOSED_BY


# --- LIVE: every row of the recorded table rebuilt, not one of four ---------------

SPECIMENS = sorted(op.SPECIMEN_STAGE)


def _logits(model, ids):
    import numpy as np
    y = model(mx.array([ids])); mx.eval(y)
    return np.asarray(y)[0]


def _moved_positions(model):
    """Change the token at position 2. In a causal model positions 0 and 1 must be
    BITWISE unchanged. Returns which of the four positions moved."""
    import numpy as np
    a, b = _logits(model, [11, 22, 33, 44]), _logits(model, [11, 22, 99, 44])
    return [not np.array_equal(a[i], b[i]) for i in range(4)]


def _build(cfg):
    """One layer of a REAL specimen config through the REAL library path, with the
    Hawking-side repair applied. No weights are loaded; mlx initialises randomly."""
    import importlib
    prepared = op.prepare_config(cfg)
    mod = importlib.import_module("mlx_lm.models." + prepared["model_type"])
    return mod.Model(mod.ModelArgs.from_dict(prepared))


@functools.lru_cache(maxsize=None)
def _live(specimen: str) -> dict:
    """MEASURE one specimen: build it and record the properties random weights
    cannot fake. Cached only because the assertions below are separate tests and a
    build costs seconds -- the measurement itself happens once per run, live.

    A MISSING STAGED CONFIG RAISES. It used to `pytest.skip`, and a skip inside a
    suite reported as "460 tests pass" reads exactly like a pass."""
    import json
    import numpy as np
    p = op.staged_config_path(specimen)
    assert p.is_file(), (
        f"staged config ABSENT for {specimen}: {p}. This is a FAILURE, not a skip: "
        f"the four specimen config.json files total 6268 bytes and staging them is "
        f"the input this test needs. Copy config.json ONLY (never weights) from "
        f"/Volumes/corpdrive/hawking-modellake/specimens/.")
    cfg = op.thin_to_one_layer(json.loads(p.read_text()))
    assert cfg.get("num_hidden_layers") == 1 or (
        cfg.get("text_config", {}).get("num_hidden_layers") == 1), (
        f"{specimen}: thinning found no layer count to cut, so this would build the "
        f"FULL depth -- refusing rather than silently running a different experiment")
    mx.random.seed(0)
    model = _build(cfg)
    base = _logits(model, [11, 22, 33, 44])
    again = _logits(model, [11, 22, 33, 44])
    return {
        "specimen": specimen,
        "model_type": cfg["model_type"],
        "shape": tuple(int(d) for d in base.shape),
        "finite": bool(np.isfinite(base).all()),
        "deterministic": bool(np.array_equal(base, again)),
        "moved": _moved_positions(model),
    }


@requires_mlx
@pytest.mark.parametrize("specimen", SPECIMENS)
def test_EVERY_specimen_ACTUALLY_BUILDS_AND_RUNS_not_just_a_recorded_string(specimen):
    """execution_coverage returns recorded evidence, which is a table somebody could
    edit into agreement with anything. This rebuilds EVERY specimen from its real
    config and pushes tokens through, so no row of that table is a string nobody
    checks. Before G061 only Falcon-H1 was rebuilt -- the one specimen that never
    needed the repair -- which left Qwen3-VL, the reason the repair exists, covered
    by nothing."""
    m = _live(specimen)
    assert m["shape"][0] == 4, f"expected 4 positions of logits: {m}"
    assert m["shape"][1] > 1000, f"expected FULL-vocab logits, not a stub head: {m}"
    assert m["finite"] is True, f"logits must be finite: {m}"
    assert m["deterministic"] is True, f"same input gave different logits: {m}"


@requires_mlx
@pytest.mark.parametrize("specimen", SPECIMENS)
def test_EVERY_specimen_is_CAUSAL_and_INPUT_SENSITIVE_not_merely_finite(specimen):
    """Finite logits pass for a graph that ignores its input entirely and for one
    that leaks information backwards in time. This checks the property that
    actually distinguishes an autoregressive decoder."""
    moved = _live(specimen)["moved"]
    assert moved[0] is False and moved[1] is False, f"{specimen} leaks backwards: {moved}"
    assert moved[2] is True, f"{specimen} ignores its input at the changed position: {moved}"


@requires_mlx
def test_the_VL_REPAIR_IS_LOAD_BEARING_two_arms_on_the_SAME_config():
    """The control arm. prepare_config is a four-line function and a decorative one
    would pass every test above unnoticed, because three of the four specimens do
    not need it. Same config, same library, one difference: WITHOUT the repair the
    build must RAISE, and the raise must name the field."""
    import importlib
    import json
    p = op.staged_config_path("Qwen3-VL-30B-A3B")
    assert p.is_file(), f"staged config ABSENT: {p} -- FAILURE, not a skip"
    cfg = op.thin_to_one_layer(json.loads(p.read_text()))
    assert "tie_word_embeddings" not in cfg["text_config"], (
        "the real config no longer withholds tie_word_embeddings from text_config, "
        "so this control arm can no longer fail and proves nothing -- re-derive "
        "which key the wrapper fails to inherit before trusting prepare_config")
    mod = importlib.import_module("mlx_lm.models." + cfg["model_type"])

    with pytest.raises(TypeError, match="tie_word_embeddings"):
        mod.Model(mod.ModelArgs.from_dict(cfg))          # CONTROL: no prepare_config

    repaired = _build(cfg)                               # REPAIRED: same config
    y = _logits(repaired, [11, 22, 33, 44])
    assert y.shape == (4, 151936), y.shape


@requires_mlx
def test_the_RECORDED_TABLE_equals_what_a_LIVE_RUN_produces_for_EVERY_specimen():
    """Recorded-plus-fully-checked is acceptable; recorded-plus-partially-checked is
    what G061 exists to end. Every literal in the table is compared against the
    measurement, so editing one without changing the world fails here."""
    cov = op.execution_coverage()
    assert set(op.SPECIMEN_STAGE) == set(cov["measured"]), (
        "a specimen in the recorded table has no staged config to rebuild it from, "
        "or a staged specimen is missing from the table -- either way a row would "
        "go unchecked, which is the exact hole this test closes")
    assert cov["live_rebuild_covers"] == SPECIMENS
    for specimen in SPECIMENS:
        m = _live(specimen)
        rec = cov["measured"][specimen]
        assert rec["builds"] is True, f"table says {specimen} does not build; it did: {m}"
        assert rec["forward"] is True and m["finite"] is True, (specimen, rec, m)
        assert cov["deterministic"][specimen] is m["deterministic"], (specimen, rec, m)
        assert cov["input_sensitive"][specimen] is m["moved"][2], (specimen, rec, m)
        assert cov["causal"][specimen] is (not m["moved"][0] and not m["moved"][1]), \
            (specimen, cov["causal"][specimen], m)
    assert cov["built_and_ran"] == len(SPECIMENS) == 4
    # the claim boundary travels with the table, so nobody reads it as adequacy
    assert cov["one_layer_only"] is True and cov["random_weights"] is True


@requires_mlx
def test_the_causality_check_FAILS_on_a_bidirectional_model():
    """Four of four passing is exactly the shape of a check that cannot fail, a
    thing this program has sealed five times. A bidirectional block -- identical but
    for the mask -- must be caught, or the tests above prove nothing."""
    import mlx.nn as nn

    class Bidirectional(nn.Module):
        def __init__(self, d=32, v=128):
            super().__init__()
            self.emb = nn.Embedding(v, d)
            self.q, self.k, self.v_ = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.out = nn.Linear(d, v)
        def __call__(self, ids):
            h = self.emb(ids)
            s = (self.q(h) @ self.k(h).transpose(0, 2, 1)) / 32 ** 0.5
            return self.out(mx.softmax(s, axis=-1) @ self.v_(h))

    mx.random.seed(0)
    moved = _moved_positions(Bidirectional())
    assert moved[0] is True and moved[1] is True, (
        "a bidirectional model must leak backwards; if it does not, the causality "
        f"check is measuring nothing: {moved}")


def test_prepare_config_copies_ONE_key_and_NEVER_overwrites():
    """A wider door that mistranslates is worse than a narrow one -- measured twice
    already in C2M. Both directions pinned: it supplies a MISSING key, and it leaves
    a key the nested config sets on purpose exactly where it was."""
    supplied = op.prepare_config({"tie_word_embeddings": False, "vocab_size": 9,
                                  "text_config": {"hidden_size": 4}})
    assert supplied["text_config"]["tie_word_embeddings"] is False
    assert "vocab_size" not in supplied["text_config"], "copied a key it was not asked to"
    kept = op.prepare_config({"tie_word_embeddings": False,
                              "text_config": {"tie_word_embeddings": True}})
    assert kept["text_config"]["tie_word_embeddings"] is True, "overwrote the nested value"


def test_prepare_config_does_not_mutate_its_input():
    src = {"tie_word_embeddings": False, "text_config": {}}
    op.prepare_config(src)
    assert src["text_config"] == {}, "mutated the caller's config in place"


def test_thin_to_one_layer_cuts_BOTH_TOWERS_and_invents_nothing():
    """The thinning is what makes a full-vocab pass affordable, so it has to be the
    thing it says it is: every layer count present goes to 1, and a key that was not
    there does not appear (inventing `depth` would build a tower the real config
    does not describe)."""
    thin = op.thin_to_one_layer({"num_hidden_layers": 48,
                                 "text_config": {"num_hidden_layers": 48},
                                 "vision_config": {"depth": 27}})
    assert thin["num_hidden_layers"] == 1
    assert thin["text_config"]["num_hidden_layers"] == 1
    assert thin["vision_config"]["depth"] == 1
    assert "depth" not in thin and "depth" not in thin["text_config"], "invented a key"
    src = {"num_hidden_layers": 48}
    op.thin_to_one_layer(src)
    assert src["num_hidden_layers"] == 48, "mutated the caller's config in place"
