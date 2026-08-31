"""Negative controls for the model-bearing cognition seam.

A validator nobody has watched reject is a validator that will silently
drift into fiction. These tests prove: a reworded restatement fails the
difference check; a missing provider reports cognition UNAVAILABLE and
does not claim participation; a decision with no reason does not count;
the receipt keeps model_decided off the tools_established column.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import frontiers as fr
from tools.future import model_bearing as mb
from tools.future import negative_index as ni
from tools.future import status_causality as sc
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)

CAND_HBM = {
    "id": "WU.hbm",
    "expected_information_gain": 3,
    "title": "rank HBM residency",
    "description": "rank HBM residency from the doctor receipt",
    "frontier": "ACTIVE_BYTES",
    "frontier_id": "FT.ACTIVE_BYTES.hbm-rank",
}
CAND_GATE = {
    "id": "WU.gate",
    "expected_information_gain": 2,
    "title": "latent readout on gate_up",
    "description": mb.RESTATEMENT_PRIOR["text"],
    "frontier": "MODEL_REPRESENTATION",
    "frontier_id": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
    "hypothesis_family": mb.RESTATEMENT_PRIOR["hypothesis_family"],
    "mechanism": mb.RESTATEMENT_PRIOR["mechanism"],
    "surface": mb.RESTATEMENT_PRIOR["surface"],
    "organ": mb.RESTATEMENT_PRIOR["organ"],
}
CAND_NEG = {
    "id": "WU.neg",
    "expected_information_gain": 2,
    "title": "rebuild scar index",
    "description": "rebuild the scar index that prunes work",
    "frontier": "VERIFICATION",
    "frontier_id": "FT.VERIFICATION.negative-index",
}
CANDS = (CAND_HBM, CAND_GATE, CAND_NEG)

CAND_POLICY_B = {
    "id": "WU.policy2",
    "expected_information_gain": 5,
    "title": "freshness refresh",
    "description": "reclassify derived artifacts",
    "frontier": "EXPERIMENT_TURNAROUND",
    "frontier_id": "FT.EXPERIMENT_TURNAROUND.refresh",
}
CAND_EMBED = {
    "id": "WU.embed",
    "expected_information_gain": 1,
    "title": "ngram product on embed",
    "description": "ngram product codebooks on the embedding table",
    "frontier": "MODEL_REPRESENTATION",
    "frontier_id": "FT.MODEL_REPRESENTATION.ngram-school",
    "hypothesis_family": "n_gram_product_codebook_table",
    "mechanism": "ngram product codebook table",
    "surface": "embed",
    "organ": "embed",
}
CANDS_B = (CAND_POLICY_B, CAND_EMBED)

FAIL = {
    "id": "WU.gate",
    "exit_code": 1,
    "error": "screen failed the coherence contract on gate_up",
    "status": "BLOCKED_NO_METAL_GPU",
}


class FakeProvider:
    """Test seam only. Production receipts call load_provider() and ignore this."""

    def __init__(self, replies=None, *, health_ok=True, seed_sessions=None):
        self._q = []
        for item in replies or []:
            self.queue(item)
        self.health_ok = health_ok
        self._sessions = list(seed_sessions if seed_sessions is not None else ["main"])
        self.prompts: list[str] = []
        self.started = False

    def queue(self, obj) -> None:
        self._q.append(json.dumps(obj) if isinstance(obj, dict) else str(obj))

    def start(self, session=None, **kwargs):
        self.started = True
        if not self.health_ok:
            return {"ok": False, "status": "not_ready"}
        sid = session or kwargs.get("session") or "main"
        if sid not in self._sessions:
            self._sessions.append(sid)
        return {"ok": True, "status": "ready", "session": sid}

    def ask(self, prompt, session=None):
        self.prompts.append(prompt)
        if not self.health_ok:
            raise RuntimeError("resident not ready")
        if session and session not in self._sessions:
            self._sessions.append(session)
        if not self._q:
            raise RuntimeError("FakeProvider has no queued reply")
        return {"text": self._q.pop(0), "ok": True, "session": session or "main"}

    def sessions(self):
        return [{"id": s} for s in self._sessions]

    def health(self):
        return {"ok": self.health_ok, "status": "ready" if self.health_ok else "absent"}

    def stop(self):
        return {"ok": True}

    def restart(self):
        return {"ok": self.health_ok, "status": "ready" if self.health_ok else "absent"}


class StickyProvider(FakeProvider):
    """Always one session. Delegation must refuse to claim a subagent."""

    def start(self, session=None, **kwargs):
        self.started = True
        return {"ok": True, "status": "ready", "session": "main"}

    def ask(self, prompt, session=None):
        self.prompts.append(prompt)
        if not self._q:
            raise RuntimeError("StickyProvider has no queued reply")
        return {"text": self._q.pop(0), "ok": True, "session": "main"}

    def sessions(self):
        return ["main"]


class IncompleteProvider:
    def ask(self, prompt, session=None):
        return {"text": "{}"}


@pytest.fixture
def fake():
    provider = FakeProvider()
    mb.bind_provider(provider)
    mb.reset_log()
    yield provider
    mb.unbind_provider()
    mb.reset_log()


def _queue_trajectory(provider: FakeProvider, *, pick="WU.gate", pick_b="WU.embed") -> None:
    provider.queue(
        {
            "reading": "CPU work remains on representation and verification",
            "worth_doing_next": [pick],
            "why": f"cite {pick}; it is a live frontier id not the gain-3 default",
        }
    )
    provider.queue(
        {
            "choice_id": pick,
            "reason": f"pick {pick} over the HBM unit; gate_up still has an untested codec",
            "mechanism": CAND_GATE["mechanism"],
            "surface": CAND_GATE["surface"],
            "hypothesis_family": CAND_GATE["hypothesis_family"],
        }
    )
    provider.queue(
        {
            "why": "the probe did not establish a missing GPU; the screen failed a contract",
            "mechanism": "coherence-contract miss, not host absence",
            "status_claim": None,
        }
    )
    provider.queue(
        {
            **mb.RESTATEMENT_PIVOT,
            "why_different": "different organ and family, not a reword of gate_up latent readout",
        }
    )
    provider.queue(
        {
            "choice_id": pick_b,
            "reason": f"second experiment on {pick_b}; leave the exhausted gate_up surface",
            "mechanism": CAND_EMBED["mechanism"],
            "surface": CAND_EMBED["surface"],
            "hypothesis_family": CAND_EMBED["hypothesis_family"],
        }
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    mb.unbind_provider()
    out = mb.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == mb.RECEIPT
    assert doc["schema"] == mb.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["cognition"]["state"] == mb.UNAVAILABLE
    assert doc["cognition"]["faked"] is False
    assert doc["cognition"]["asked"] is False
    assert doc["model_decided"] is None
    assert doc["declared_vs_executed"]["resident_asked_in_this_receipt"] is False
    assert doc["declared_vs_executed"]["live_probe_re_run"] is False
    assert doc["tools_established"]["restatement_fails"] is True
    assert doc["tools_established"]["pivot_passes"] is True
    assert doc["separation"]["model_decided"] is None
    assert doc["separation"]["tools_established"]["restatement_check"]["different"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["frontier"] == "FT.CHILD_RESIDENT.launch"


def test_fake_provider_never_appears_in_receipt(fake):
    """NEGATIVE CONTROL: a bound fake must not become receipt evidence."""
    fake.queue({"reading": "no", "worth_doing_next": ["WU.hbm"], "why": "should not land in the receipt"})
    out = mb.build()
    doc = json.loads(out.read_text())
    assert doc["cognition"]["faked"] is False
    assert doc["cognition"]["asked"] is False
    assert doc["cognition"]["state"] == mb.UNAVAILABLE
    assert doc["model_decided"] is None
    assert "test seam" not in str(doc["cognition"].get("provider_source") or "")
    assert doc["declared_vs_executed"]["resident_asked_in_this_receipt"] is False


# ---------------------------------------------------------------------------
# Difference check
# ---------------------------------------------------------------------------


def test_reworded_restatement_fails_meaningfully_different():
    """NEGATIVE CONTROL: a reworded hypothesis A cannot pass as hypothesis B."""
    row = mb.meaningfully_different(mb.RESTATEMENT_PRIOR, mb.RESTATEMENT_REWORD)
    assert row["different"] is False
    assert "restatement" in row["why"]
    text_only = mb.meaningfully_different(
        {"text": mb.RESTATEMENT_PRIOR["text"]},
        {"text": mb.RESTATEMENT_REWORD["text"]},
    )
    assert text_only["different"] is False
    assert mb.meaningfully_different(mb.RESTATEMENT_PRIOR, mb.RESTATEMENT_PRIOR)["different"] is False


def test_different_mechanism_or_surface_passes_meaningfully_different():
    row = mb.meaningfully_different(mb.RESTATEMENT_PRIOR, mb.RESTATEMENT_PIVOT)
    assert row["different"] is True
    assert "embed" in row["why"] or "family" in row["why"]
    text_only = mb.meaningfully_different(
        {"text": mb.RESTATEMENT_PRIOR["text"]},
        {"text": mb.RESTATEMENT_PIVOT["text"]},
    )
    assert text_only["different"] is True


def test_empty_hypotheses_are_refused_not_called_different():
    """NEGATIVE CONTROL: emptiness is not a new mechanism."""
    row = mb.meaningfully_different({}, {})
    assert row["different"] is False
    assert row["refused"] is True
    assert mb.meaningfully_different(None, mb.RESTATEMENT_PRIOR)["different"] is False


# ---------------------------------------------------------------------------
# Provider absence / incomplete / unhealthy
# ---------------------------------------------------------------------------


def test_provider_unavailable_reports_cognition_unavailable_and_refuses_participation():
    """NEGATIVE CONTROL: no provider -> UNAVAILABLE, never a scripted success."""
    mb.bind_provider(None)
    mb.reset_log()
    row = mb.interpret(list(CANDS))
    assert row["cognition"] == mb.UNAVAILABLE
    assert row["participated"] is False
    assert row["model_decided"] is None
    assert row["fall_back_to_scripted"] is False
    assert "scripted" in (row["refused"] or "").lower() or "UNAVAILABLE" in row["refused"]
    picked = mb.choose(CANDS, scar_pool=[])
    assert picked["cognition"] == mb.UNAVAILABLE
    assert picked["chose"] is None
    assert picked["fall_back_to_scripted"] is False
    mp = mb.materially_participated()
    assert mp["participated"] is False
    assert mp["cognition"] == mb.UNAVAILABLE
    assert "UNAVAILABLE" in mp["why"]
    mb.unbind_provider()


def test_unhealthy_provider_is_unavailable(fake):
    fake.health_ok = False
    row = mb.choose(CANDS, scar_pool=[])
    assert row["cognition"] == mb.UNAVAILABLE
    assert row["chose"] is None
    assert row["participated"] is False


def test_incomplete_provider_is_unavailable():
    mb.bind_provider(IncompleteProvider())
    mb.reset_log()
    row = mb.choose(CANDS, scar_pool=[])
    assert row["cognition"] == mb.UNAVAILABLE
    assert row["chose"] is None
    mb.unbind_provider()


def test_ask_failure_after_health_ok_is_unavailable_not_a_scripted_pick():
    """NEGATIVE CONTROL: a crashing ask is UNAVAILABLE, never the fixed policy."""

    class Boom(FakeProvider):
        def ask(self, prompt, session=None):
            raise RuntimeError("eof on stdin")

    mb.bind_provider(Boom())
    mb.reset_log()
    row = mb.choose(CANDS, scar_pool=[])
    assert row["cognition"] == mb.UNAVAILABLE
    assert row["chose"] is None
    assert row["fall_back_to_scripted"] is False
    mb.unbind_provider()


def test_empty_frontier_refuses_without_inventing_work(fake):
    row = mb.interpret([])
    assert row["participated"] is False
    assert row["grounded"] is False
    assert row["model_decided"] is None
    assert "invent" in (row["refused"] or "").lower()


# ---------------------------------------------------------------------------
# Choose / interpret
# ---------------------------------------------------------------------------


def test_choose_records_divergence_from_fixed_policy(fake):
    policy = mb.fixed_policy_choose(CANDS, scar_pool=[])
    assert policy["id"] == "WU.hbm"
    fake.queue(
        {
            "choice_id": "WU.gate",
            "reason": "leave HBM; gate_up still has an untested codec surface",
            "mechanism": CAND_GATE["mechanism"],
            "surface": CAND_GATE["surface"],
            "hypothesis_family": CAND_GATE["hypothesis_family"],
        }
    )
    row = mb.choose(CANDS, scar_pool=[])
    assert row["cognition"] == mb.AVAILABLE
    assert row["chose"]["id"] == "WU.gate"
    assert row["reason"]
    assert row["diverged_from_fixed_policy"] is True
    assert row["tools_established"]["fixed_policy_id"] == "WU.hbm"
    assert row["model_decided"]["choice_id"] == "WU.gate"
    assert row["model_decided"]["reason"] == row["reason"]
    assert row["tools_established"]["fixed_policy_id"] != row["model_decided"]["choice_id"]
    enacted = mb.record_outcome(row["seq"], {"id": "WU.gate"})
    assert enacted["changed_what_ran_next"] is True


def test_choose_matching_fixed_policy_is_recorded_as_no_divergence(fake):
    """NEGATIVE CONTROL: agreeing with the policy is not material participation."""
    fake.queue(
        {
            "choice_id": "WU.hbm",
            "reason": "highest gain HBM unit is still the cheapest discriminating work",
        }
    )
    row = mb.choose(CANDS, scar_pool=[])
    assert row["chose"]["id"] == "WU.hbm"
    assert row["diverged_from_fixed_policy"] is False
    enacted = mb.record_outcome(row["seq"], {"id": "WU.hbm"})
    assert enacted["changed_what_ran_next"] is False
    mp = mb.materially_participated()
    assert mp["participated"] is False
    assert mp["publishable_finding"]


def test_decision_with_no_recorded_reason_does_not_count(fake):
    """NEGATIVE CONTROL: a pick without a reason is not participation."""
    fake.queue({"choice_id": "WU.gate"})
    row = mb.choose(CANDS, scar_pool=[])
    assert row["chose"] is None
    assert row["participated"] is False
    assert row["counts_as_decision"] is False
    assert row["reason"] is None
    assert "reason" in (row["refused"] or "")


def test_invented_choice_id_is_refused(fake):
    """NEGATIVE CONTROL: the model cannot mint an id the candidate set does not have."""
    fake.queue({"choice_id": "WU.ghost", "reason": "because I invented it"})
    row = mb.choose(CANDS, scar_pool=[])
    assert row["chose"] is None
    assert row["participated"] is False
    assert "WU.ghost" in (row["refused"] or "")


def test_interpret_rejects_invented_frontier_ids(fake):
    fake.queue(
        {
            "reading": "invented work",
            "worth_doing_next": ["FT.INVENTED.nope"],
            "why": "cite an id that is not on the frontier",
        }
    )
    row = mb.interpret(list(CANDS))
    assert row["grounded"] is False
    assert row["participated"] is False
    assert "FT.INVENTED.nope" in row["tools_established"]["rejected_ids"]
    assert "FT.INVENTED.nope" not in row["tools_established"]["accepted_ids"]


def test_interpret_grounds_in_live_next_work(fake):
    units = fr.next_work(fr.THIS_HOST_LANES)
    assert units, "live next_work is empty; the frontier book refused to load"
    pick = units[min(3, len(units) - 1)]["id"]
    fake.queue(
        {
            "reading": "CPU work remains on the live frontier",
            "worth_doing_next": [pick],
            "why": f"cite {pick}; it is a real next_work id",
        }
    )
    row = mb.interpret()
    assert row["cognition"] == mb.AVAILABLE
    assert row["grounded"] is True
    assert pick in row["tools_established"]["accepted_ids"]
    assert row["tools_established"]["rejected_ids"] == []
    assert row["participated"] is True


def test_scar_dead_pick_is_refused_by_tools_even_if_model_picks_it(fake):
    """NEGATIVE CONTROL: the model proposes; tools still kill known-dead ideas."""
    scar = ni.Scar(
        scar_id="NS.TEST.cross_expert",
        source_path="receipts/future/NEGATIVE_SCIENCE_INDEX.json",
        source_origin="test",
        parse_status=ni.PARSED,
        model="qwen3-80b",
        models=["qwen3-80b"],
        organ="routed_experts",
        organs=["routed_experts"],
        hypothesis_family="cross_expert_structure",
        refuse_eligible=True,
        verdict="DEAD",
        failure_mechanism="tying",
    ).finalize()
    dead = {
        "id": "WU.dead",
        "expected_information_gain": 9,
        "model": "qwen3-80b",
        "organ": "routed_experts",
        "hypothesis_family": "cross_expert_structure",
        "description": "retry cross-expert tying",
    }
    fake.queue(
        {
            "choice_id": "WU.dead",
            "reason": "retry the tying idea under a new label",
        }
    )
    row = mb.choose([dead, CAND_HBM], scar_pool=[scar])
    assert row["chose"] is None
    assert row["model_decided"]["choice_id"] == "WU.dead"
    assert row["tools_established"]["scar_refusal"]["refused"] is True
    assert row["changed_what_ran_next"] is False
    policy = mb.fixed_policy_choose([dead, CAND_HBM], scar_pool=[scar])
    assert policy["id"] == "WU.hbm"


# ---------------------------------------------------------------------------
# Failure explanation / status causality
# ---------------------------------------------------------------------------


def test_explain_failure_keeps_unsupported_cause_as_hypothesis(fake):
    fake.queue(
        {
            "why": "the Metal GPU is missing so the screen could not run",
            "mechanism": "host absence",
            "status_claim": "BLOCKED_NO_METAL_GPU",
        }
    )
    row = mb.explain_failure(FAIL)
    tools = row["tools_established"]
    assert tools["status_challenge"]["verdict"] == sc.OVERREACHING
    assert tools["cause_is_hypothesis"] is True
    assert tools["model_cause_not_entailed"] is True
    assert row["model_decided"]["why"]
    assert row["model_decided"]["status_claim"] == "BLOCKED_NO_METAL_GPU"


# ---------------------------------------------------------------------------
# next hypothesis / trajectory
# ---------------------------------------------------------------------------


def test_next_hypothesis_reword_is_not_participation(fake):
    """NEGATIVE CONTROL: hyp B that restates A is recorded, and does not count."""
    fake.queue({**mb.RESTATEMENT_REWORD, "why_different": "it is phrased more clearly"})
    row = mb.next_hypothesis(mb.RESTATEMENT_PRIOR)
    assert row["meaningfully_different"] is False
    assert row["participated"] is False
    assert row["chose"] is None
    assert "restatement" in (row["refused"] or "")


def test_next_hypothesis_pivot_counts(fake):
    fake.queue({**mb.RESTATEMENT_PIVOT, "why_different": "leave gate_up for the embed table"})
    row = mb.next_hypothesis(mb.RESTATEMENT_PRIOR)
    assert row["meaningfully_different"] is True
    assert row["participated"] is True
    assert row["model_decided"]["surface"] == "embed"


def test_trajectory_enacts_divergence_and_materially_participates(fake):
    _queue_trajectory(fake)
    traj = mb.run_trajectory(
        "advance representation without replaying gate_up",
        CANDS,
        FAIL,
        CANDS_B,
        scar_pool=[],
        enact=True,
    )
    mp = traj["materially_participated"]
    assert mp["participated"] is True
    assert mp["divergence_count"] >= 1
    assert mp["different_hypothesis_count"] >= 1
    assert mp["changed_what_ran_next_count"] >= 1
    assert traj["hypothesis_b"]["meaningfully_different"] is True
    assert traj["model_decided"]["choose_a"]["choice_id"] == "WU.gate"
    assert traj["tools_established"]["choose_a"]["fixed_policy_id"] == "WU.hbm"
    assert traj["model_decided"]["choose_a"]["choice_id"] != traj["tools_established"]["choose_a"]["fixed_policy_id"]


def test_ignored_model_pick_does_not_count_as_changing_what_ran(fake):
    """NEGATIVE CONTROL: Python running the policy after the model spoke is not participation."""
    _queue_trajectory(fake)
    traj = mb.run_trajectory(
        "advance representation without replaying gate_up",
        CANDS,
        FAIL,
        CANDS_B,
        scar_pool=[],
        enact=False,
    )
    mp = traj["materially_participated"]
    assert mp["participated"] is False
    assert mp["changed_what_ran_next_count"] == 0
    assert "changed what ran" in mp["why"]


# ---------------------------------------------------------------------------
# Delegate
# ---------------------------------------------------------------------------


def test_delegate_requires_a_distinct_session(fake):
    fake.queue({"plan": "rank remaining organs; stop when the receipt lands", "stop_condition": "receipt"})
    row = mb.delegate("rank remaining Flash organs after the gate_up scar")
    assert row["participated"] is True
    assert row["tools_established"]["session_distinct"] is True
    assert row["tools_established"]["does_not_wait_on_tools"] is True
    assert row["model_decided"]["session"] not in {"main", None, ""}


def test_delegate_without_distinct_session_is_refused():
    """NEGATIVE CONTROL: reusing the only session is not subagent work."""
    sticky = StickyProvider()
    sticky.queue({"plan": "looks like a subagent if you do not check sessions"})
    mb.bind_provider(sticky)
    mb.reset_log()
    row = mb.delegate("rank remaining Flash organs after the gate_up scar")
    assert row["participated"] is False
    assert row["tools_established"]["session_distinct"] is False
    assert "distinct session" in (row["refused"] or "")
    mb.unbind_provider()


def test_delegate_without_provider_is_unavailable():
    mb.bind_provider(None)
    mb.reset_log()
    row = mb.delegate("anything")
    assert row["cognition"] == mb.UNAVAILABLE
    assert row["participated"] is False
    mb.unbind_provider()


# ---------------------------------------------------------------------------
# Tape / separation / placeholders
# ---------------------------------------------------------------------------


def test_decision_log_separates_model_decided_from_tools_established(fake):
    fake.queue(
        {
            "choice_id": "WU.gate",
            "reason": "gate_up still has an untested codec surface",
            "mechanism": CAND_GATE["mechanism"],
            "surface": CAND_GATE["surface"],
        }
    )
    row = mb.choose(CANDS, scar_pool=[])
    tape = mb.decision_log()
    assert tape
    last = tape[-1]
    assert last["seq"] == row["seq"]
    assert "choice_id" in last["model_decided"]
    assert "fixed_policy_id" in last["tools_established"]
    assert last["model_decided"]["choice_id"] != last["tools_established"]["fixed_policy_id"]
    assert "fixed_policy_id" not in last["model_decided"]
    assert "choice_id" not in last["tools_established"]


def test_receipt_separates_what_model_decided_from_what_tools_established():
    mb.unbind_provider()
    doc = json.loads(mb.build().read_text())
    assert "model_decided" in doc
    assert "tools_established" in doc
    assert doc["model_decided"] is None
    assert doc["tools_established"]["restatement_fails"] is True
    assert doc["separation"]["model_decided"] is None
    assert "the model proposes" in doc["separation"]["rule"]


def test_enter_loop_unavailable_does_not_fall_back_to_scripted():
    mb.bind_provider(None)
    mb.reset_log()
    row = mb.enter_loop(list(CANDS), CANDS, scar_pool=[])
    assert row["chose"] is None
    assert row["fall_back_to_scripted"] is False
    assert row["cognition"] == mb.UNAVAILABLE
    mb.unbind_provider()


def test_enter_loop_honours_the_model_pick(fake):
    fake.queue(
        {
            "reading": "CPU work remains",
            "worth_doing_next": ["WU.gate"],
            "why": "cite WU.gate; it is live and not the gain-3 default",
        }
    )
    fake.queue(
        {
            "choice_id": "WU.gate",
            "reason": "leave HBM; gate_up still has an untested codec",
        }
    )
    row = mb.enter_loop(list(CANDS), CANDS, scar_pool=[])
    assert row["chose"]["id"] == "WU.gate"
    assert row["choose"]["diverged_from_fixed_policy"] is True
    assert row["fall_back_to_scripted"] is False


def test_empty_log_is_not_participation():
    mb.reset_log()
    mp = mb.materially_participated()
    assert mp["participated"] is False
    assert "no decisions" in mp["why"]


def test_no_hardware_claims_on_public_records(fake):
    fake.queue(
        {
            "choice_id": "WU.gate",
            "reason": "gate_up still has an untested codec surface",
        }
    )
    row = mb.choose(CANDS, scar_pool=[])
    diff = mb.meaningfully_different(mb.RESTATEMENT_PRIOR, mb.RESTATEMENT_REWORD)
    for node in (row, diff, mb.live_probe_record(), mb.cognition_state()):
        _assert_no_hardware_claims(node)
        for key in HARDWARE_FIELDS:
            if key in node and isinstance(node[key], (int, float)):
                raise AssertionError(f"{key} leaked into a public record")


def test_source_has_no_placeholders():
    src = Path("tools/future/model_bearing.py").read_text()
    assert "raise NotImplementedError" not in src
    assert "\npass\n" not in src
    assert "TODO" not in src
    assert "pytest.skip" not in src
    tree = ast.parse(src)
    assert tree.body


def test_hardware_claim_in_receipt_would_raise():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 35})
