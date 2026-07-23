#!/usr/bin/env python3.12
"""Tests for the storage/Vulture lifecycle and the quality contract."""
import pathlib
import sys

import pytest

FOUNDRY = pathlib.Path(__file__).resolve().parents[1]
if str(FOUNDRY) not in sys.path:
    sys.path.insert(0, str(FOUNDRY))

import quality_contract as qc  # noqa: E402
import storage_modes as sm  # noqa: E402

GB = 10**9


def _capability_evidence(**over):
    ev = {
        "real_parent_forward": True,
        "real_packed_forward": True,
        "split": "holdout",
        "n_tokens": 4096,
        "domains": list(qc.PROTECTED_QUALITY_DOMAINS),
        "metrics": {"mean_symmetric_kl": 0.04, "argmax_agreement": 0.97},
    }
    ev.update(over)
    return ev


# ── storage mode from bytes, not params ───────────────────────────────────────

def test_mode_signature_has_no_parameter_count():
    import inspect

    assert "param" not in "".join(inspect.signature(sm.choose_mode).parameters)


def test_397b_bf16_is_bigger_on_disk_than_1t_int4():
    """Precision drives disk size. Params would rank these backwards."""
    bf16_397b = 397 * 10**9 * 2                 # ~794 GB, 397B params
    int4_1t = 1000 * 10**9 * 60 // 100          # ~595 GB, 1T params
    assert bf16_397b > int4_1t

    free, reserve = 700 * GB, 0
    small_params = sm.choose_mode(bf16_397b, free, reserve, "397B-bf16")
    big_params = sm.choose_mode(int4_1t, free, reserve, "1T-int4")

    assert small_params["mode"] == sm.VULTURE_SHARD_SERIAL   # fewer params, does NOT fit
    assert big_params["mode"] == sm.FULL_DISK_RESIDENT       # more params, DOES fit
    assert "parameter_count" in big_params["never_decided_from"]


def test_mode_ladder_by_free_space():
    manifest = 65 * GB
    assert sm.choose_mode(manifest, 200 * GB, 50 * GB, "F0")["mode"] == sm.FULL_DISK_RESIDENT
    assert sm.choose_mode(manifest, 90 * GB, 50 * GB, "F0")["mode"] == sm.VULTURE_SHARD_SERIAL
    assert sm.choose_mode(manifest, 60 * GB, 50 * GB, "F0")["mode"] == sm.BOUNDED_REMOTE_RANGE


def test_reserve_is_honoured_not_borrowed():
    d = sm.choose_mode(100 * GB, 120 * GB, 30 * GB, "F0")
    assert d["usable_bytes"] == 90 * GB
    assert d["mode"] != sm.FULL_DISK_RESIDENT


def test_expert_cache_cap_stays_at_measured_20gb():
    """64 GB gave 0 evictions and drove swap to 906 MB free on a lockstep pass."""
    assert sm.EXPERT_CACHE_CAP_BYTES == 20 * GB


@pytest.mark.parametrize(
    "args",
    [(0, GB, 0, "F0"), (-1, GB, 0, "F0"), (GB, -1, 0, "F0"), (GB, GB, 0, "")],
)
def test_bad_inputs_rejected(args):
    with pytest.raises((ValueError, TypeError)):
        sm.choose_mode(*args)


def test_param_count_passed_as_bytes_is_a_type_error():
    with pytest.raises(TypeError):
        sm.choose_mode(1.16e11, 200 * GB, 0, "F0")


# ── vulture coverage invariant ────────────────────────────────────────────────

def test_vulture_serial_pass_achieves_full_coverage():
    shards = {"s1": ["a.w", "a.b"], "s2": ["b.w"], "s3": ["c.w", "c.b"]}
    manifest = [t for ts in shards.values() for t in ts]
    packed = []
    for tensors in shards.values():          # fetch, harvest, pack, seal, release
        packed.extend(tensors)
        sm.assert_shard_release_ordered(tensors, packed)
    assert sm.assert_full_coverage(manifest, packed)["complete"]


def test_skipped_tensor_is_a_coverage_error():
    with pytest.raises(sm.CoverageError):
        sm.assert_full_coverage(["a", "b", "c"], ["a", "b"])
    assert sm.vulture_coverage(["a", "b", "c"], ["a", "b"])["missing"] == ["c"]


def test_tensor_outside_the_manifest_is_a_coverage_error():
    with pytest.raises(sm.CoverageError):
        sm.assert_full_coverage(["a"], ["a", "ghost"])


def test_shard_released_before_its_tensors_are_packed():
    with pytest.raises(sm.CoverageError):
        sm.assert_shard_release_ordered(["a", "b"], ["a"])


# ── vulture release predicate ─────────────────────────────────────────────────

PINNED = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"   # F0 gpt-oss-120b revision


def test_release_ok_with_pinned_revision_and_retained_metadata():
    r = sm.vulture_release_ok(PINNED, {"config", "index", "tokenizer"})
    assert r["ok"] and r["independent_of_capability_gate"] is True


def test_release_blocked_without_pinned_immutable_revision():
    for rev in ("main", "v1.0", "", None, "b5c939de"):
        assert not sm.vulture_release_ok(rev, {"config", "index", "tokenizer"})["ok"]


def test_release_blocked_without_retained_rehydration_metadata():
    r = sm.vulture_release_ok(PINNED, {"config", "index"})
    assert not r["ok"] and "tokenizer" in r["reasons"][0]


def test_release_does_not_require_a_passing_artifact():
    """F0 was NEGATIVE at sub-bit and still released. Release is not the gate."""
    assert sm.vulture_release_ok(PINNED, {"config", "index", "tokenizer"})["ok"]


# ── quality gate ──────────────────────────────────────────────────────────────

def test_gate_needs_both_conditions():
    ev = _capability_evidence()
    assert qc.evaluate({"mean_symmetric_kl": 0.10, "argmax_agreement": 0.95}, ev)["passed"]
    assert not qc.evaluate({"mean_symmetric_kl": 0.11, "argmax_agreement": 0.99}, ev)["passed"]
    assert not qc.evaluate({"mean_symmetric_kl": 0.01, "argmax_agreement": 0.94}, ev)["passed"]


def test_partial_metrics_are_not_a_gate():
    with pytest.raises(qc.ContractViolation):
        qc.evaluate({"mean_symmetric_kl": 0.01})


def test_extras_are_recorded_not_gating():
    r = qc.evaluate(
        {
            "mean_symmetric_kl": 0.5,
            "argmax_agreement": 0.4,
            "logit_cosine": 0.999,
            "top5_overlap": 0.8,
            "candidate_perplexity": 12.3,
        },
        _capability_evidence(),
    )
    assert r["passed"] is False and r["recorded"]["logit_cosine"] == 0.999


# ── evidence classes ──────────────────────────────────────────────────────────

def test_classify_each_rung():
    assert qc.classify({"metrics": {"bpw": 0.77}}) == "PHYSICAL"
    assert qc.classify({"metrics": {"weight_recon_error": 0.66}}) == "FUNCTIONAL_PROXY"
    assert qc.classify({"per_layer_divergence": [0.1, 0.2]}) == "LAYER"
    assert qc.classify(_capability_evidence(split="validation")) == "SHORT_END_TO_END"
    assert qc.classify(_capability_evidence(n_tokens=88)) == "SHORT_END_TO_END"
    assert qc.classify(_capability_evidence()) == "CAPABILITY"


def test_overclaim_is_rejected():
    proxy = {"metrics": {"weight_recon_error": 0.668}}   # d32k65536 space-filling gain
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_overclaimed("CAPABILITY", proxy)
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_overclaimed("LAYER", proxy)
    assert qc.assert_not_overclaimed("FUNCTIONAL_PROXY", proxy)["ok"]
    assert qc.assert_not_overclaimed("PHYSICAL", proxy)["ok"]   # underclaim is fine


def test_unknown_class_rejected():
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_overclaimed("VIBES", _capability_evidence())


def test_only_capability_selects_a_frontier():
    assert qc.may_select_frontier(_capability_evidence())
    assert not qc.may_select_frontier(_capability_evidence(n_tokens=88))
    assert not qc.may_select_frontier({"metrics": {"bpw": 0.3}})   # byte plan is not capability
    assert not qc.evaluate(
        {"mean_symmetric_kl": 0.01, "argmax_agreement": 0.99},
        _capability_evidence(split="calibration"),
    )["may_select_frontier"]


def test_protected_domains_required_for_capability():
    partial = _capability_evidence(domains=["code"])
    assert qc.classify(partial) == "SHORT_END_TO_END"


# ── splits ────────────────────────────────────────────────────────────────────

def test_splits_must_be_disjoint_and_present():
    ok = {"calibration": ["a"], "validation": ["b"], "holdout": ["c"]}
    assert qc.assert_splits_disjoint(ok)["sizes"]["holdout"] == 1
    with pytest.raises(qc.ContractViolation):
        qc.assert_splits_disjoint({"calibration": ["a"], "validation": ["a"], "holdout": ["c"]})
    with pytest.raises(qc.ContractViolation):
        qc.assert_splits_disjoint({"calibration": ["a"], "validation": ["b"], "holdout": []})


# ── threshold weakening ───────────────────────────────────────────────────────

def test_unchanged_contract_matches_the_seal():
    assert qc.assert_not_weakened(dict(qc.CONTRACT))["change"] == "unchanged"
    assert qc.contract_hash() == qc.SEALED_CONTRACT_SHA256


def test_weakening_after_a_failure_is_rejected():
    failed = qc.evaluate({"mean_symmetric_kl": 0.42, "argmax_agreement": 0.61})
    assert failed["passed"] is False
    for weaker in (
        {**qc.CONTRACT, "max_mean_symmetric_kl": 0.5},
        {**qc.CONTRACT, "min_argmax_agreement": 0.60},
        {**qc.CONTRACT, "min_capability_tokens": 88},
    ):
        with pytest.raises(qc.ContractViolation):
            qc.assert_not_weakened(weaker)


def test_removing_a_threshold_is_weakening():
    stripped = {k: v for k, v in qc.CONTRACT.items() if k != "min_argmax_agreement"}
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_weakened(stripped)


def test_tightening_is_allowed_and_reseals():
    tighter = {**qc.CONTRACT, "max_mean_symmetric_kl": 0.05}
    r = qc.assert_not_weakened(tighter)
    assert r["change"] == "tightened" and r["sealed_sha256"] != qc.SEALED_CONTRACT_SHA256
