#!/usr/bin/env python3.12
"""Synthetic-only tests for the Qwen3-235B Gravity campaign controller.

NO real forward and NO real Qwen weight is ever touched. The 235B source is replaced by a tiny
stubbed forward whose "experts" are 8x16 random matrices, so the whole lockstep driver, the
packers, the checkpointing and the verdict logic run in milliseconds.

Covered:
  * resume-skip: a sealed row is never recomputed
  * parent-logit persistence and reload (and the fact that they are loaded BEFORE the skip branch,
    which is the bug this controller exists to fix)
  * expert-OUTER / candidate-INNER ordering: each expert is read exactly once per layer no matter
    how many candidates are in flight
  * verdict thresholds at and either side of the sealed gate
  * source-absent clean no-op (WAITING_SOURCE, exit 0, no lease)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import qwen_gravity_campaign as G  # noqa: E402


# --------------------------------------------------------------------------- #
# tiny stubbed model
# --------------------------------------------------------------------------- #
HID, INTER, VOCAB, NLAYERS, NEXPERTS = 16, 8, 32, 2, 4


class _Geom:
    hidden, n_layers, n_experts, top_k = HID, NLAYERS, NEXPERTS, 2
    moe_inter, vocab, eps = INTER, VOCAB, 1e-6
    norm_topk_prob, tie = True, False


class _Reader:
    """In-memory stand-in for SafetensorsIndexReader. Counts expert reads."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.t: dict[str, np.ndarray] = {}
        self.source_dir = "."
        self.expert_reads: list[tuple[int, int, str]] = []
        self.t["model.embed_tokens.weight"] = rng.standard_normal((VOCAB, HID)).astype(np.float32) * 0.1
        self.t["lm_head.weight"] = rng.standard_normal((VOCAB, HID)).astype(np.float32) * 0.1
        self.t["model.norm.weight"] = np.ones(HID, np.float32)
        for L in range(NLAYERS):
            self.t[f"model.layers.{L}.post_attention_layernorm.weight"] = np.ones(HID, np.float32)
            self.t[f"model.layers.{L}.mlp.gate.weight"] = rng.standard_normal((NEXPERTS, HID)).astype(np.float32)
            for e in range(NEXPERTS):
                p = f"model.layers.{L}.mlp.experts.{e}"
                self.t[f"{p}.gate_proj.weight"] = rng.standard_normal((INTER, HID)).astype(np.float32) * 0.1
                self.t[f"{p}.up_proj.weight"] = rng.standard_normal((INTER, HID)).astype(np.float32) * 0.1
                self.t[f"{p}.down_proj.weight"] = rng.standard_normal((HID, INTER)).astype(np.float32) * 0.1

    def bf16(self, name: str) -> np.ndarray:
        if ".mlp.experts." in name:
            bits = name.split(".")
            self.expert_reads.append((int(bits[2]), int(bits[5]), bits[6]))
        return self.t[name]

    def bf16_rows(self, name, rows):
        return self.t[name][list(rows)].copy()

    def has(self, name):
        return name in self.t

    def source_present(self):
        return True

    def close(self):
        pass


class _Fwd:
    """Stubbed QwenRealForward: real routing/SwiGLU path, attention replaced by a cheap linear mix."""

    def __init__(self, reader=None):
        self.reader = reader or _Reader()
        self.g = _Geom()

    def source_present(self):
        return self.reader.source_present()

    def _attention(self, L: int, x: np.ndarray) -> np.ndarray:
        return 0.01 * x


def _tiny_ladder(monkeypatch):
    """Two cheap packed rungs plus the parent, sized for 8x16 / 16x8 matrices."""
    ladder = {
        "R0_parent": {"kind": "parent", "note": "parent"},
        "RA": {"kind": "packed", "note": "pq",
               "gate_up": {"family": "product_quant", "dim": 8, "subspaces": 2, "k": 4},
               "down": {"family": "product_quant", "dim": 8, "subspaces": 2, "k": 4}},
        "RB": {"kind": "packed", "note": "grammar",
               "gate_up": {"family": "shared_grammar", "dim": 8, "k": 8, "stages": 1},
               "down": {"family": "shared_grammar", "dim": 8, "k": 8, "stages": 1}},
    }
    monkeypatch.setattr(G, "LADDER", ladder)
    monkeypatch.setattr(G, "LADDER_ORDER", ["R0_parent", "RA", "RB"])
    return ladder


def _tiny_holdout(monkeypatch, n=2):
    hold = [{"id": f"p{i}", "domain": "test", "text": f"prompt {i}"} for i in range(n)]
    monkeypatch.setattr(G, "HOLDOUT", hold)
    return hold


def _sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(G, "LEASES", tmp_path / "leases")
    monkeypatch.setattr(G, "HEARTBEAT", tmp_path / "heartbeat")
    monkeypatch.setattr(G, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(G, "PARENT_LOGITS", tmp_path / "parent_logits")
    monkeypatch.setattr(G, "CONTROLLER", tmp_path / "controller")
    monkeypatch.setattr(G, "STATE_PATH", tmp_path / "STATE.json")
    monkeypatch.setattr(G, "WAITING_RECEIPT", tmp_path / "WAITING.json")
    monkeypatch.setattr(G, "PASS_STATE_NPZ", tmp_path / "pass_state.npz")
    monkeypatch.setattr(G, "PASS_STATE_JSON", tmp_path / "pass_state.json")
    monkeypatch.setattr(G, "LEASE_PATH", tmp_path / "leases" / "qwen_gravity.lease")
    monkeypatch.setattr(G, "HB_PATH", tmp_path / "heartbeat" / "hb.json")
    for d in ("leases", "heartbeat", "checkpoints", "parent_logits", "controller"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# source-absent clean no-op
# --------------------------------------------------------------------------- #
def test_source_absent_seals_waiting_source(tmp_path, monkeypatch):
    _sandbox(monkeypatch, tmp_path)

    class _Absent:
        def source_present(self):
            return False

    monkeypatch.setattr(G, "_parent_forward", lambda: _Absent())
    rc = G.run()
    assert rc == 0                                        # clean exit, never a crash
    st = json.loads((tmp_path / "STATE.json").read_text())
    assert st["status"] == "WAITING_SOURCE"
    assert (tmp_path / "WAITING.json").exists()
    assert not G.LEASE_PATH.exists()                      # no heavy lease taken


# --------------------------------------------------------------------------- #
# ladder shape + byte-ledger scope
# --------------------------------------------------------------------------- #
def test_ladder_is_data_driven_and_decisive_first():
    assert G.LADDER_ORDER[0] == "R0_parent"
    assert set(G.LADDER_ORDER) == set(G.LADDER)
    for name in G.LADDER_ORDER[1:]:
        c = G.LADDER[name]
        assert c["kind"] == "packed"
        assert set(c) >= {"gate_up", "down"}              # organ inversion is explicit
    rows = G._rows()
    assert len(rows) == len(G.LADDER_ORDER) * len(G.HOLDOUT)
    assert all(r["candidate"] == "R0_parent" for r in rows[:len(G.HOLDOUT)])
    assert len({r["row_id"] for r in rows}) == len(rows)


def test_routing_aware_rung_is_one_step_harsher_on_the_cold_quartile():
    # Deliberately the coldest QUARTILE, not the median split: the 88-token routing calibration is
    # only 63.6 percent stable at the median, and the instability is concentrated there. Allocating
    # bits on a coin-flip partition would be unfalsifiable, so the band has to be the stable one.
    r2, r3 = G.LADDER["R2_subhalf_best"], G.LADDER["R3_routing_aware"]
    assert r3["gate_up"] == r2["gate_up"] and r3["down"] == r2["down"]
    assert r3["cold_frac"] == 0.25
    assert r3["cold_gate_up"]["dim"] == 2 * r2["gate_up"]["dim"]     # halved index rate
    assert r3["cold_down"]["dim"] == 2 * r2["down"]["dim"]


def test_cold_partition_matches_the_byte_ledger_membership_rule():
    """The packer and the byte ledger MUST agree on who is cold, or the sealed 0.3554 whole-model
    BPW would not describe the artifact that was actually measured. The fallback therefore uses the
    exact stand-in qwen_subhalfbit_search.whole_model_bpw uses (expert % 100 < cold_frac*100), skew
    and all, until qwen_routing_calibration.load_partition supplies the real frequencies."""
    cold = G._cold_experts(0, 128, 0.5)
    assert G._PARTITION_SOURCE["source"] in (
        "deterministic_expert_index_standin", "qwen_routing_calibration.load_partition")
    if G._PARTITION_SOURCE["source"] == "deterministic_expert_index_standin":
        assert cold == frozenset(e for e in range(128) if (e % 100) < 50)


def test_acct_spec_maps_product_quant_onto_the_exact_bit_vocabulary():
    s = G._acct_spec({"family": "product_quant", "dim": 32, "subspaces": 8, "k": 8, "strata": 2})
    assert s["family"] == "transform_pq" and "strata" not in s
    # exact closed form: subspaces*log2(k)/dim bits per weight
    n = 1536 * 4096
    bits = G.SHB.expert_bits((1536, 4096), s, G.DEPLOY_CLUSTER)
    assert abs(bits / n - 8 * 3 / 32) < 1e-3


# --------------------------------------------------------------------------- #
# verdict thresholds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kl,agree,expect_pass,expect_verdict", [
    (0.10, 0.95, True, "PASS"),          # exactly at the sealed gate -> PASS
    (0.1001, 0.99, False, "degraded"),   # KL just over
    (0.01, 0.9499, False, "degraded"),   # agreement just under
    (5.0, 0.10, False, "collapse"),      # blown up
])
def test_verdict_thresholds(kl, agree, expect_pass, expect_verdict):
    ok, verdict = G._verdict({"mean_sym_kl": kl, "next_token_argmax_agreement": agree})
    assert ok is expect_pass and verdict == expect_verdict


def test_divergence_of_identical_logits_is_a_pass():
    rng = np.random.default_rng(3)
    lg = rng.standard_normal((5, 32)).astype(np.float32)
    div = G._divergence(lg, lg)
    assert div["mean_sym_kl"] == 0.0
    assert div["next_token_argmax_agreement"] == 1.0
    assert G._verdict(div)[0] is True


# --------------------------------------------------------------------------- #
# chunked k-means (the OOM / MPS-sync fix)
# --------------------------------------------------------------------------- #
def test_chunked_kmeans_matches_a_blocked_and_unblocked_assignment():
    torch = G.gf._torch()
    rng = np.random.default_rng(7)
    v = torch.from_numpy(rng.standard_normal((512, 8)).astype(np.float32))
    cb = G._kmeans_chunked(v, 16, iters=4, seed=0)
    assert cb.shape == (16, 8)
    assert torch.isfinite(cb).all()
    # blocking must not change the answer
    G_row = G._row_chunk
    try:
        G._row_chunk = lambda k: 37          # force many blocks
        blocked = G._assign_chunked(v, cb)
    finally:
        G._row_chunk = G_row
    assert torch.equal(blocked, G.gf._assign(v, cb))


def test_expert_cache_bytes_are_actually_billed():
    """A dict value is billed as 0 bytes by bounded_cache._entry_bytes, which would silently defeat
    HAWKING_CACHE_MAX_GB on a 103 GB box. Experts must be cached as an organ-ordered tuple."""
    cache = G.bc.PressureAwareCache("t", floor_gb=0.0, disk_reserve_gb=0.0, verbose=False)
    ex = {o: np.zeros((8, 16), np.float32) for o in G.ORGANS}
    G._cache_put(cache, (0, 0), ex)
    assert cache.stats()["entries"] == 1
    assert G.bc._entry_bytes(cache.get((0, 0))) == 3 * 8 * 16 * 4
    back = G._cache_get(cache, (0, 0))
    assert set(back) == set(G.ORGANS) and back["gate"].shape == (8, 16)
    assert G._cache_get(cache, (9, 9)) is None


def test_memory_policy_defaults_match_the_box():
    assert os.environ["HAWKING_CACHE_MAX_GB"] == "64"
    assert os.environ["HAWKING_CACHE_FLOOR_GB"] == "12"


def test_row_norm_strata_split_is_balanced_and_norm_ordered():
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((16, 8)) * np.logspace(-5, 0, 16)[:, None]).astype(np.float32)
    parts = G._strata_rows(w, 2)
    assert sorted(np.concatenate(parts).tolist()) == list(range(16))
    assert len(parts[0]) == len(parts[1]) == 8
    nrm = np.linalg.norm(w, axis=1)
    assert nrm[parts[0]].max() <= nrm[parts[1]].min()      # low-norm stratum is separated


# --------------------------------------------------------------------------- #
# FIX 1: expert OUTER, candidate INNER
# --------------------------------------------------------------------------- #
def test_each_expert_is_read_once_per_layer_regardless_of_candidate_count(tmp_path, monkeypatch):
    _sandbox(monkeypatch, tmp_path)
    _tiny_ladder(monkeypatch)
    hold = _tiny_holdout(monkeypatch)
    ids = {h["id"]: [1, 2, 3, 4] for h in hold}

    def _reads_for(variants):
        reader = _Reader()
        fwd = _Fwd(reader)
        plan = {v: [h["id"] for h in hold] for v in variants}
        out = G.lockstep_logits(fwd, variants, plan, ids, fit_experts=2, resume=False)
        assert len(out) == len(variants) * len(hold)
        for arr in out.values():
            assert arr.shape == (4, VOCAB) and np.isfinite(arr).all()
        # one (layer, expert) is read once: 3 organ tensors per read
        per_expert = {}
        for L, e, organ in reader.expert_reads:
            per_expert.setdefault((L, e), []).append(organ)
        return per_expert

    one = _reads_for(["R0_parent"])
    three = _reads_for(["R0_parent", "RA", "RB"])
    # every touched expert read exactly once (3 organs), and adding two more candidates adds ZERO
    # extra streaming: this is the 2.7 h of re-reading the restructure removes.
    assert one and all(sorted(v) == ["down_proj", "gate_proj", "up_proj"] for v in one.values())
    assert all(sorted(v) == ["down_proj", "gate_proj", "up_proj"] for v in three.values())
    assert set(three) >= set(one)
    assert sum(len(v) for v in three.values()) == 3 * len(three)


# --------------------------------------------------------------------------- #
# FIX 2: parent-logit persistence + reload, and resume-skip
# --------------------------------------------------------------------------- #
def _run_tiny(monkeypatch, reader=None, **kw):
    fwd = _Fwd(reader)
    monkeypatch.setattr(G, "_parent_forward", lambda: fwd)
    monkeypatch.setattr(G, "_other_heavy_lease_live", lambda: None)
    monkeypatch.setattr(G, "A", _StubAdapter())
    monkeypatch.setattr(G, "_ladder_bpw", lambda inv, cand: {"whole_model_bpw": 0.5,
                                                             "scope": "stub"})
    return G.run(**kw), fwd


class _StubAdapter:
    """Only build_inventory/load_* are used by the controller; BPW itself is stubbed out."""

    def build_inventory(self, *a, **k):
        return object()

    def load_config(self, *a, **k):
        return {}

    def load_index(self, *a, **k):
        return {}


def test_parent_logits_persist_and_reload_and_rows_resume_skip(tmp_path, monkeypatch):
    _sandbox(monkeypatch, tmp_path)
    _tiny_ladder(monkeypatch)
    hold = _tiny_holdout(monkeypatch)

    class _Tok:
        def encode(self, text):
            class _E:
                ids = [1, 2, 3, 4]
            return _E()

    monkeypatch.setattr(G, "_tokenizer", lambda: _Tok())

    reader = _Reader()
    rc, _ = _run_tiny(monkeypatch, reader)
    assert rc == 0

    # every row sealed, parent logits on disk (the durable 53 MB artifact at real scale)
    rows = G._rows()
    for r in rows:
        assert (G.CHECKPOINTS / f"{r['row_id']}.json").exists(), r["row_id"]
    for h in hold:
        p = G.PARENT_LOGITS / f"{h['id']}.npy"
        assert p.is_file()
        assert np.load(p).shape == (4, VOCAB)
    assert set(G._load_parent_logits()) == {h["id"] for h in hold}
    assert not G.PASS_STATE_NPZ.exists()          # resume point cleared on a clean finish

    # a packed row records a real divergence + a verdict against the sealed thresholds
    rec = json.loads((G.CHECKPOINTS / f"{hold[0]['id']}__RA.json").read_text())
    assert set(rec["divergence_vs_parent"]) == {"mean_sym_kl", "mean_logit_cosine",
                                                "mean_top5_overlap", "next_token_argmax_agreement"}
    assert rec["verdict"] in ("PASS", "degraded", "collapse")
    assert rec["capability_pass"] is (rec["verdict"] == "PASS")
    assert "whole_model_bpw" in rec["bpw"]        # whole-model, never expert-only

    # RESUME: rerun with everything sealed -> zero expert streaming, zero parent forwards
    reader2 = _Reader()
    rc2, _ = _run_tiny(monkeypatch, reader2)
    assert rc2 == 0
    assert reader2.expert_reads == []             # resume-skip: nothing recomputed
    st = json.loads((tmp_path / "STATE.json").read_text())
    assert st["status"] == "SEALED" and st["final"] is True

    # RESUME with one candidate row deleted -> the parent is NOT recomputed (loaded from disk)
    (G.CHECKPOINTS / f"{hold[0]['id']}__RA.json").unlink()
    reader3 = _Reader()
    rc3, _ = _run_tiny(monkeypatch, reader3)
    assert rc3 == 0
    assert reader3.expert_reads, "the deleted candidate row must be recomputed"
    variants_seen = json.loads((G.CHECKPOINTS / f"{hold[0]['id']}__RA.json").read_text())
    assert variants_seen["variant"] == "RA"


# --------------------------------------------------------------------------- #
# durability primitives
# --------------------------------------------------------------------------- #
def test_lease_claim_refuse_and_atomic_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_other_heavy_lease_live", lambda: None)
    monkeypatch.setattr(G, "LEASES", tmp_path / "leases")
    monkeypatch.setattr(G, "LEASE_PATH", tmp_path / "leases" / "qwen_gravity.lease")
    G._acquire_lease()
    assert G.LEASE_PATH.exists()
    assert json.loads(G.LEASE_PATH.read_text())["owner"] == G.LABEL
    with pytest.raises(SystemExit):
        G._acquire_lease()


def test_other_heavy_lease_blocks_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_other_heavy_lease_live", lambda: "QWEN_TRANSFER pid 42390")
    monkeypatch.setattr(G, "LEASES", tmp_path / "leases")
    monkeypatch.setattr(G, "LEASE_PATH", tmp_path / "leases" / "qwen_gravity.lease")
    with pytest.raises(SystemExit):
        G._acquire_lease()
    assert not G.LEASE_PATH.exists()


def test_pass_state_roundtrip_and_signature_guard(tmp_path, monkeypatch):
    _sandbox(monkeypatch, tmp_path)
    states = {("R0_parent", "p0"): np.arange(8, dtype=np.float32).reshape(2, 4)}
    G._save_pass_state(states, 7, "sig-a")
    back, layer = G._load_pass_state("sig-a")
    assert layer == 7 and np.array_equal(back[("R0_parent", "p0")], states[("R0_parent", "p0")])
    # a different ladder / variant set must invalidate the resume point rather than corrupt a run
    assert G._load_pass_state("sig-b") == (None, 0)


def test_unknown_candidate_is_rejected_by_the_cli():
    with pytest.raises(SystemExit):
        G.main(["run", "--candidates", "R9_not_a_rung"])
