"""Learned performance model pins. FRONT E (G047, §113)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import perf_model as pm  # noqa: E402


def test_it_imports_and_features_are_the_declared_four():
    assert len(pm.features(256, 2)) == 4
    assert pm.features(64, 1)[3] == 1 / 64      # the reciprocal that can see the cliff


def test_a_perfectly_linear_landscape_is_recovered():
    """If the model cannot fit data its own feature set generates exactly, nothing
    downstream is interpretable."""
    import math
    rows = [(t, e, 1.0 + 0.5 * math.log2(t) + 0.25 * e + 3.0 / (t * e))
            for t in (64, 128, 256, 512, 1024) for e in (1, 2, 4)]
    w = pm.fit(rows)
    for t, e, ms in rows:
        assert abs(pm.predict(w, t, e) - ms) < 1e-3, (t, e)


def test_grade_keeps_the_cliff_separate_from_the_flat_points():
    """The whole point of the grader: averaging one hard case into fourteen easy ones
    reports a good score for a model that fails the only question that matters."""
    loo = [{"threadgroup": 256, "ept": 1, "actual_ms": 0.51, "model_ms": 0.51,
            "mean_baseline_ms": 0.51, "median_baseline_ms": 0.51} for _ in range(14)]
    loo.append({"threadgroup": 64, "ept": 1, "actual_ms": 0.60, "model_ms": 0.52,
                "mean_baseline_ms": 0.515, "median_baseline_ms": 0.515})
    for r in loo:
        for k in ("model", "mean_baseline", "median_baseline"):
            r[f"{k}_abs_err_pct"] = abs(r[f"{k}_ms"] - r["actual_ms"]) / r["actual_ms"] * 100
    g = pm.grade(loo)
    assert g["n_cliff"] == 1 and g["n_flat"] == 14
    assert g["cliff_abs_err_pct"] > 10          # the failure is visible, not averaged away


def test_leave_one_out_never_trains_on_the_point_it_predicts():
    """A model that saw the answer is not a prediction. Pinned by making one point an
    extreme outlier: if it leaked into training the fit would chase it."""
    rows = [(t, e, 0.5) for t in (64, 128, 256, 512, 1024) for e in (1, 2, 4)]
    rows[0] = (64, 1, 50.0)
    loo = pm.leave_one_out(rows)
    outlier = next(r for r in loo if r["threadgroup"] == 64 and r["ept"] == 1)
    assert outlier["model_ms"] < 10.0, outlier["model_ms"]
