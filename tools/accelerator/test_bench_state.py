"""S032 §3: quiescence is a BENCHMARK INPUT and a receipt that quotes a duration
must carry the machine state it was measured under.

Every test here breaks the rule on purpose. The first one exists because a
checker that refused everything would make the rest pass.
"""
import pytest

import bench
import receipt as R


IDS = {k: R.absent("not under test") for k in R.IDENTITIES}


def _build(result, bench_block=None):
    return R.build(experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE",
                   identities=dict(IDS), result=result,
                   claim_boundary="test", passed=True, bench=bench_block)


def quiet_sample(**over):
    d = {"quiet": True, "method": "enumerate", "contenders": [], "n_contenders": 0,
         "max_rss_gib": 0.0, "total_cpu_pct": 0.0}
    d.update(over)
    return d


def test_ANTI_VACUITY_a_receipt_with_no_timing_and_no_bench_block_is_FINE():
    """Without this, a build() that raised on everything would pass every test
    below and the rule would look enforced while forbidding all work."""
    r = _build({"prunable_rows": 0, "cos_threshold": 0.30})
    assert r["bench"] is None


def test_a_TIMING_CLAIM_WITHOUT_A_BENCH_BLOCK_IS_REFUSED():
    with pytest.raises(ValueError) as e:
        _build({"median_ms": 0.3626})
    assert "median_ms" in str(e.value) and "S032" in str(e.value)


@pytest.mark.parametrize("result", [
    {"raw_tps": 34.14},
    {"arms": {"reuse": {"median_s": 0.0003}}},                  # nested dict
    {"sweeps": [{"gpu_ns": 12345}]},                            # inside a list
    {"speedup": 1.2},
    {"p95": 41.0},
    {"us_per_dispatch": 17.35},
])
def test_TIMING_IS_FOUND_ANYWHERE_IN_THE_RESULT_TREE(result):
    """A rule that only looks at top-level keys is defeated by one level of
    nesting, which is how every real receipt is shaped."""
    with pytest.raises(ValueError):
        _build(result)


def test_a_TIMING_CLAIM_WITH_A_BENCH_BLOCK_IS_ADMITTED():
    r = _build({"median_ms": 0.36},
               bench.bench_block(machine="M3 Ultra", before=quiet_sample(),
                                 after=quiet_sample()))
    assert r["bench"]["state"] == "QUIESCED"


def test_QUIESCED_CANNOT_BE_ASSERTED_WITHOUT_A_SAMPLE():
    """The steer's rule verbatim: if quiescence is unknown, UNKNOWN, not quiet."""
    with pytest.raises(ValueError) as e:
        _build({"median_ms": 0.36},
               {"state": "QUIESCED", "recorded_at": "now", "machine": "M3 Ultra"})
    assert "UNKNOWN, never quiet" in str(e.value)


def test_QUIESCED_IS_REFUSED_WHEN_CONTENDERS_WERE_RECORDED():
    bad = {"state": "QUIESCED", "recorded_at": "now", "machine": "M3 Ultra",
           "quiescence": quiet_sample(quiet=True, n_contenders=1,
                                      contenders=[{"comm": "mlx_boot.py"}])}
    with pytest.raises(ValueError) as e:
        _build({"median_ms": 0.36}, bad)
    assert "mlx_boot.py" in str(e.value)


def test_AN_INVENTED_STATE_IS_REFUSED():
    with pytest.raises(ValueError):
        _build({"median_ms": 0.36},
               {"state": "MOSTLY_QUIET", "recorded_at": "now", "machine": "x"})


def test_A_STATE_WITHOUT_A_TIMESTAMP_OR_A_MACHINE_IS_REFUSED():
    for drop in ("recorded_at", "machine"):
        b = {"state": "UNKNOWN", "recorded_at": "now", "machine": "M3 Ultra"}
        b.pop(drop)
        with pytest.raises(ValueError) as e:
            _build({"median_ms": 0.36}, b)
        assert drop in str(e.value)


# ---- the derivation itself -------------------------------------------------

def test_NO_SAMPLES_DERIVES_UNKNOWN_NOT_QUIESCED():
    assert bench.bench_block(machine="m")["state"] == "UNKNOWN"


def test_A_FAILED_ENUMERATION_DERIVES_UNKNOWN_NOT_QUIESCED():
    """`ps` exiting non-zero found nothing because it could not look. That is the
    0-of-0-cases shape this program has sealed four times."""
    failed = {"quiet": None, "method": "enumerate", "refused": "ps exited 1",
              "contenders": []}
    assert bench.bench_block(machine="m", before=failed,
                             after=quiet_sample())["state"] == "UNKNOWN"


def test_ONE_CONTENDED_SAMPLE_IS_ENOUGH_TO_DERIVE_CONTENDED():
    loud = quiet_sample(quiet=False, n_contenders=1, max_rss_gib=39.0,
                        contenders=[{"comm": "mlx_boot.py", "rss_gib": 39.0}])
    b = bench.bench_block(machine="m", before=quiet_sample(), after=loud)
    assert b["state"] == "CONTENDED"
    assert b["quiescence"]["max_rss_gib"] == 39.0, "the WORST sample must be kept"


def test_THE_STATE_IS_DERIVED_FROM_SAMPLES_NOT_PASSED_IN():
    """bench_block takes no state argument. A caller cannot stamp QUIESCED on a
    contended measurement without editing the derivation."""
    import inspect
    assert "state" not in inspect.signature(bench.bench_block).parameters


def test_QUIESCED_is_refused_when_a_SAMPLE_disagrees_with_the_summary():
    """A harness in this program wrote "bench_state": "QUIESCED" as a literal while
    all six of its quiescence probes read quiet=False. It never reached build(), but
    the hole it exposed is real: the summary was checked and the samples were not.

    Both arms run. The quiet-summary/noisy-sample dict must be REFUSED, and the
    all-quiet one must still be accepted -- otherwise this test would pass by
    refusing everything."""
    quiet = {"quiet": True, "n_contenders": 0, "contenders": []}
    noisy = {"quiet": False, "n_contenders": 2,
             "contenders": [{"comm": "avconferenced", "cpu_pct": 33.4, "rss_gib": 0.02}]}
    ids = {k: "x" for k in R.IDENTITIES}
    def build(before, after):
        return R.build(
            experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE", identities=ids,
            result={"median_ms": 1.0}, claim_boundary="b", passed=True,
            bench={"state": "QUIESCED", "recorded_at": "2026-08-26T00:00:00Z",
                   "machine": "m", "quiescence": quiet,
                   "samples": {"before": before, "after": after}})

    with pytest.raises(ValueError, match="ASSERTED"):
        build(quiet, noisy)
    with pytest.raises(ValueError, match="ASSERTED"):
        build(noisy, quiet)
    assert build(quiet, quiet)["bench"]["state"] == "QUIESCED"


def test_bench_block_NEVER_produces_a_dict_this_guard_would_refuse():
    """The guard above protects hand-built dicts. bench_block must be incapable of
    tripping it, or the tool and its check disagree."""
    quiet = {"quiet": True, "n_contenders": 0, "contenders": []}
    noisy = {"quiet": False, "n_contenders": 1, "contenders": [{"comm": "x"}]}
    for before, after in ((quiet, quiet), (quiet, noisy), (noisy, noisy), (None, quiet)):
        b = bench.bench_block(machine="m", before=before, after=after)
        R.build(experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE",
                      identities={k: "x" for k in R.IDENTITIES},
                      result={"median_ms": 1.0}, claim_boundary="b", passed=True, bench=b)
        if b["state"] == "QUIESCED":
            assert before is not None and after is not None


def test_ONE_SIDED_quiet_is_UNKNOWN_not_QUIESCED():
    """Both ends or it is not quiesced. A quiet `before` with no `after` says the
    machine was quiet at one instant, not across the window; contention that starts
    after the before-sample and ends before any after-sample leaves no trace.

    The noisy arm runs too: one-sided NOISE still reads CONTENDED, because evidence
    of noise is evidence while absence of observed noise on one side is not."""
    quiet = {"quiet": True, "n_contenders": 0, "contenders": []}
    noisy = {"quiet": False, "n_contenders": 1, "contenders": [{"comm": "x"}]}
    assert bench.bench_block(machine="m", before=quiet, after=None)["state"] == "UNKNOWN"
    assert bench.bench_block(machine="m", before=None, after=quiet)["state"] == "UNKNOWN"
    assert bench.bench_block(machine="m", before=quiet, after=quiet)["state"] == "QUIESCED"
    assert bench.bench_block(machine="m", before=noisy, after=None)["state"] == "CONTENDED"
