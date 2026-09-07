"""The fusion must be output-identical and must patch the TYPE.

A speedup that changes outputs is refused (see O003_BATCH_EQUIVALENCE); a patch
applied to the instance does nothing at all, which is how an entire ablation run
was wasted earlier in this campaign.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "future"))
import moe_fusion_patch as P


class _Blk:
    pass


class _Layer:
    def __init__(self):
        self.mlp = _Blk()
        self.mlp.switch_mlp = object()


class _Model:
    def __init__(self):
        self.language_model = type("LM", (), {})()
        self.language_model.model = type("M", (), {})()
        self.language_model.model.layers = [_Layer() for _ in range(3)]


def test_apply_patches_every_block_and_is_idempotent():
    m = _Model()
    assert P.apply(m) == 3
    assert P.apply(m) == 0, "second apply must be a no-op, not a double patch"
    assert P.revert(m)


def test_patch_lands_on_the_TYPE_not_the_instance():
    m = _Model()
    t = type(m.language_model.model.layers[0].mlp)
    before = t.__call__
    P.apply(m)
    assert t.__call__ is not before, "must patch the type; Python ignores instance __call__"
    P.revert(m)
    assert t.__call__ is before


def test_revert_restores_and_reports():
    m = _Model()
    P.apply(m)
    assert P.revert(m) is True
    assert P.revert(m) is False, "revert on an unpatched type must report False"


def test_measured_claims_are_recorded_in_the_docstring():
    """The numbers must travel with the code, including the honest smaller one."""
    d = P.__doc__
    assert "117.27" in d and "115.01" in d, "end-to-end figures missing"
    assert "0.000e+00" in d, "output-identity evidence missing"
    assert "smaller than the isolated one" in d, "the caveat must not be dropped"
