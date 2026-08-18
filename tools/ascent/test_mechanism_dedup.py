"""Mechanism-level dedup and dry-queue synthesis.

Required evidence, produced by these tests and written to the receipt:
- a semantic duplicate is REFUSED, naming the prior attempt it matched
- a legitimate retry is ALLOWED, naming which precondition changed
- synthesis from a dry queue produces a NEW mechanism
- each gate goes red when its input is corrupted
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ascent.mechanism_dedup import (
    CAMPAIGN_EXHAUSTED,
    DEFAULT_LEDGER,
    DEFAULT_RECEIPT,
    DEFAULT_SCIENCE,
    GENESIS_VEHICLE,
    SYNTHESIS_CATALOG,
    Attempt,
    CatalogItem,
    Proposal,
    Verdict,
    admit,
    coerce_science,
    corrupt_gate_cases,
    decide_next,
    demonstration_legitimate_retry,
    demonstration_semantic_duplicate,
    demonstration_synthesis,
    emit_evidence,
    load_measured_bottlenecks,
    load_science_register,
    remaining_bottlenecks,
    retrieve,
    synthesize,
)
from ascent.mechanism_identity import same_mechanism as ident_same


REPO = Path(__file__).resolve().parents[2]
RECEIPT_DIR = REPO / "receipts" / "ascent-2026-08-16"


def _prop(mechanism: str, **kw) -> Proposal:
    data = dict(
        mechanism=mechanism,
        bottleneck="weight_addressing",
        model="qwen38",
        representation=GENESIS_VEHICLE["representation"],
        implementation=GENESIS_VEHICLE["implementation"],
        genome=GENESIS_VEHICLE["genome"],
        bytes_per_token=GENESIS_VEHICLE["bytes_per_token"],
        launch_geometry=GENESIS_VEHICLE["launch_geometry"],
        command_topology=GENESIS_VEHICLE["command_topology"],
        artifact=GENESIS_VEHICLE["artifact"],
    )
    data.update(kw)
    return Proposal(**data)


def _ledger() -> list[dict]:
    return [
        {
            "component": "weight_addressing",
            "ns_per_token": 21_293_103,
            "pct": 60.44,
            "triage": "EXISTENTIAL",
        },
        {
            "component": "deltanet",
            "ns_per_token": 3_732_795,
            "pct": 10.60,
            "triage": "research",
        },
        {
            "component": "command_submission",
            "ns_per_token": 12_084,
            "pct": 0.03,
            "triage": "noise",
        },
    ]


# ---------------------------------------------------------------- required demonstrations


def test_semantic_duplicate_is_refused_naming_the_prior() -> None:
    decision = demonstration_semantic_duplicate()
    assert decision.verdict == Verdict.REFUSE
    assert decision.red
    assert decision.gate == "semantic_duplicate"
    assert decision.matched_prior is not None
    assert decision.matched_prior.id == "wa-exhausted-01"
    assert "fusing" in decision.matched_prior.mechanism.lower()
    assert "gemv" in decision.matched_prior.mechanism.lower()
    assert ident_same(
        "fuse small Metal kernels into the next GEMV",
        decision.matched_prior.mechanism,
    ).same


def test_omitting_prior_metadata_does_not_forge_a_retry_delta() -> None:
    prior = CAMPAIGN_EXHAUSTED[0]
    incomplete = Proposal(
        mechanism="fuse small Metal kernels into the next GEMV",
        bottleneck="weight_addressing",
        model="qwen38",
    )
    decision = admit(
        incomplete,
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "semantic_duplicate"
    assert decision.changed == []


def test_legitimate_retry_is_allowed_naming_the_changed_precondition() -> None:
    decision = demonstration_legitimate_retry()
    assert decision.verdict == Verdict.ALLOW
    assert not decision.red
    assert decision.gate == "retry_delta"
    assert "precondition" in decision.changed
    which = (decision.details or {}).get("precondition", {}).get("which")
    assert which is not None
    assert "launch_geometry" in which
    assert decision.matched_prior is not None
    assert decision.matched_prior.launch_geometry == "tpr64_tg128"


def test_synthesis_from_dry_queue_produces_a_new_mechanism() -> None:
    decision = demonstration_synthesis()
    assert decision.verdict == Verdict.SYNTHESIZE
    assert decision.synthesized is not None
    syn = decision.synthesized
    assert syn.mechanism_id == "per_layer_per_head_assignment"
    assert syn.bottleneck == "weight_addressing"
    assert syn.target["synthesized"] is True
    assert syn.target["status"] == "pending"
    for prior in CAMPAIGN_EXHAUSTED:
        assert not ident_same(syn.mechanism, prior.mechanism).same, prior.id
    assert "fuse_tiny" not in syn.mechanism_id
    assert decision.details.get("n_exhausted") == 9


def test_same_bottleneck_different_mechanism_is_allowed() -> None:
    prior = CAMPAIGN_EXHAUSTED[0]
    decision = admit(
        _prop("assign codecs per layer and per head on the coherent vehicle"),
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.ALLOW
    assert decision.gate == "new_mechanism"


# ---------------------------------------------------------------- listed reopen axes


def test_representation_change_allows_retry() -> None:
    prior = CAMPAIGN_EXHAUSTED[0]
    decision = admit(
        _prop(
            "fusing tiny kernels into the following GEMV",
            representation="per-layer mixed q3/q4 on coherent 4.2527 vehicle",
        ),
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.ALLOW
    assert decision.changed == ["representation"]


def test_implementation_change_allows_retry() -> None:
    prior = CAMPAIGN_EXHAUSTED[0]
    decision = admit(
        _prop(
            "fusing tiny kernels into the following GEMV",
            implementation="new_kernel_identity_tpr64",
        ),
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.ALLOW
    assert "implementation" in decision.changed


def test_evidence_change_allows_retry() -> None:
    prior = Attempt(
        id="wa-1",
        mechanism="cross-token cache reuse",
        bottleneck="weight_addressing",
        representation=GENESIS_VEHICLE["representation"],
        implementation=GENESIS_VEHICLE["implementation"],
        evidence_digest="old-hot-cold-2p5",
        artifact=GENESIS_VEHICLE["artifact"],
    )
    decision = admit(
        _prop(
            "retain weights between tokens",
            evidence_digest="new-capture-coherent-4.2527",
        ),
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.ALLOW
    assert "evidence" in decision.changed


def test_invalid_prior_test_allows_retry() -> None:
    prior = CAMPAIGN_EXHAUSTED[7]  # gaussian, test_valid=False
    assert prior.test_valid is False
    decision = admit(
        _prop("evaluate or fit compression on gaussian / synthetic proxy activations"),
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.ALLOW
    assert decision.gate == "invalid_prior"
    assert decision.changed == ["previous_test_invalid"]


# ---------------------------------------------------------------- retrieve


def test_retrieve_pulls_science_attempts_and_running() -> None:
    science = [
        {
            "id": "NS-009",
            "mechanism": "Evaluate or fit compression on Gaussian / synthetic proxy activations",
            "class": "REFUTED",
            "models": ["qwen38"],
            "retry_when": "never",
        }
    ]
    attempts = [CAMPAIGN_EXHAUSTED[7]]
    running = [
        Attempt(
            id="live-gauss",
            mechanism="gaussian activations",
            bottleneck="weight_addressing",
            status="running",
        )
    ]
    got = retrieve(
        _prop("gaussian activations"),
        science=science,
        attempts=attempts,
        running=running,
    )
    assert len(got.science) == 1
    assert got.science[0].entry.id == "NS-009"
    assert any(a.id == "wa-exhausted-08" for a in got.attempts)
    assert any(a.id == "live-gauss" for a in got.running)


def test_admit_against_the_real_register_refuses_gaussian() -> None:
    if not DEFAULT_SCIENCE.is_file():
        pytest.skip("NEGATIVE_SCIENCE_REGISTER.json not on disk")
    entries = load_science_register(DEFAULT_SCIENCE)
    decision = admit(_prop("gaussian activations"), science=entries)
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "negative_science"
    assert decision.matched_science is not None
    assert decision.matched_science.id == "NS-009"


def test_retrieve_against_the_real_register_finds_ns009() -> None:
    if not DEFAULT_SCIENCE.is_file():
        pytest.skip("NEGATIVE_SCIENCE_REGISTER.json not on disk")
    entries = load_science_register(DEFAULT_SCIENCE)
    assert any(e.id == "NS-009" for e in entries)
    got = retrieve(
        _prop("gaussian activations"),
        science=entries,
        attempts=[],
        running=[],
    )
    assert any(h.entry.id == "NS-009" for h in got.science)


def test_real_token_ledger_keeps_weight_addressing_existential() -> None:
    if not DEFAULT_LEDGER.is_file():
        pytest.skip("TOKEN_NS_QWEN38.json not on disk")
    rows = load_measured_bottlenecks(DEFAULT_LEDGER)
    remain = remaining_bottlenecks(rows)
    assert remain
    assert remain[0]["component"] == "weight_addressing"
    assert remain[0]["ns_per_token"] > 20_000_000


# ---------------------------------------------------------------- science / sealed / inflight


def test_terminal_science_is_refused() -> None:
    decision = admit(
        _prop("evaluate or fit compression on gaussian / synthetic proxy activations"),
        science=[
            {
                "id": "NS-009",
                "mechanism": "Evaluate or fit compression on Gaussian / synthetic proxy activations",
                "class": "REFUTED",
                "models": ["qwen38"],
                "retry_when": "never. Real captured activations only.",
            }
        ],
    )
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "negative_science"
    assert decision.matched_science is not None
    assert decision.matched_science.id == "NS-009"


def test_insufficient_science_does_not_refuse() -> None:
    decision = admit(
        _prop("single-family representation across gate_proj / up_proj / down_proj"),
        science=[
            {
                "id": "NS-012",
                "mechanism": "Single-family representation across gate_proj / up_proj / down_proj",
                "class": "INSUFFICIENT",
                "models": ["q80"],
                "retry_when": "reopen with a per-organ measurement",
            }
        ],
    )
    # q80-only entry does not apply to qwen38; either way INSUFFICIENT is not terminal.
    assert decision.verdict == Verdict.ALLOW


def test_q80_is_refused_at_launch() -> None:
    decision = admit(_prop("assign codecs per layer and per head", model="q80"))
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "sealed_model"


def test_inflight_same_mechanism_is_refused() -> None:
    decision = admit(
        _prop("assign codecs per layer and per head"),
        running=[
            Attempt(
                id="live-assign",
                mechanism="per-layer and per-head assignment",
                bottleneck="weight_addressing",
                status="running",
            )
        ],
    )
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "inflight_duplicate"
    assert decision.matched_prior is not None
    assert decision.matched_prior.id == "live-assign"


def test_nine_exhausted_then_tenth_is_new() -> None:
    """The actual night-wasting shape: 9 failures, dry queue, 21 ms still there."""
    decision = decide_next(
        pending=[],
        measured=_ledger(),
        attempts=list(CAMPAIGN_EXHAUSTED),
        include_campaign_exhausted=True,
    )
    assert decision.verdict == Verdict.SYNTHESIZE
    assert decision.synthesized is not None
    assert decision.synthesized.mechanism_id == SYNTHESIS_CATALOG[0].mechanism_id
    # And that tenth mechanism is itself admissible.
    allowed = admit(
        decision.synthesized.as_proposal(),
        attempts=list(CAMPAIGN_EXHAUSTED),
        include_campaign_exhausted=True,
    )
    assert allowed.verdict == Verdict.ALLOW


def test_pending_duplicate_is_treated_as_dry_and_synthesizes() -> None:
    pending = [
        _prop("fusing tiny kernels into the following GEMV", id="auto-qwen38-weight-addressing-try2")
    ]
    decision = decide_next(
        pending=pending,
        measured=_ledger(),
        attempts=list(CAMPAIGN_EXHAUSTED),
        include_campaign_exhausted=True,
    )
    assert decision.verdict == Verdict.SYNTHESIZE
    assert decision.synthesized is not None
    assert decision.synthesized.mechanism_id != "fuse_tiny_kernels_into_gemv"
    refusals = decision.details.get("pending_refusals") or []
    assert refusals
    assert refusals[0]["verdict"] == "REFUSE"


def test_admissible_pending_is_launched_not_synthesized() -> None:
    pending = [_prop("assign codecs per layer and per head", id="pending-assign")]
    decision = decide_next(
        pending=pending,
        measured=_ledger(),
        attempts=list(CAMPAIGN_EXHAUSTED),
        include_campaign_exhausted=True,
    )
    assert decision.verdict == Verdict.LAUNCH
    assert decision.proposal is not None
    assert decision.proposal.id == "pending-assign"


def test_hold_when_nothing_measurable_remains() -> None:
    decision = synthesize(
        measured=[
            {"component": "command_submission", "ns_per_token": 12_084, "triage": "noise"}
        ],
        include_campaign_exhausted=True,
    )
    assert decision.verdict == Verdict.HOLD
    assert decision.gate == "synthesis_needs_bottleneck"


def test_will_not_replay_when_catalog_is_only_exhausted() -> None:
    decision = synthesize(
        measured=_ledger(),
        attempts=list(CAMPAIGN_EXHAUSTED),
        include_campaign_exhausted=True,
        catalog=(
            CatalogItem(
                mechanism_id="fuse_tiny_kernels_into_gemv",
                mechanism="fusing tiny kernels into the following GEMV",
                bottleneck="weight_addressing",
                axis="fusion",
                assumption_attacked="tiny kernels are free",
                experiment="do not",
            ),
        ),
    )
    assert decision.verdict == Verdict.ERROR
    assert decision.gate == "synthesis_not_replay"
    assert decision.red


# ---------------------------------------------------------------- corrupt each gate


def test_every_gate_goes_red_on_corrupted_input() -> None:
    cases = corrupt_gate_cases()
    assert cases, "no gate corruptions registered"
    failed = [c for c in cases if not c["red"]]
    assert not failed, failed
    gates = {c["gate"] for c in cases}
    required = {
        "named_mechanism_required",
        "semantic_duplicate",
        "retry_delta",
        "invalid_prior",
        "inflight_duplicate",
        "negative_science",
        "sealed_model",
        "synthesis_needs_bottleneck",
        "synthesis_not_replay",
    }
    assert required <= gates


def test_empty_mechanism_is_error() -> None:
    decision = admit({"mechanism": "", "bottleneck": "weight_addressing", "model": "qwen38"})
    assert decision.verdict == Verdict.ERROR
    assert decision.gate == "named_mechanism_required"
    assert decision.red


def test_bottleneck_name_as_mechanism_is_refused() -> None:
    decision = admit(
        {"mechanism": "weight_addressing", "bottleneck": "weight_addressing", "model": "qwen38"}
    )
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "named_mechanism_required"
    assert decision.red


def test_identical_values_do_not_count_as_a_delta() -> None:
    """Corrupt the retry gate: claim a retry, change nothing."""
    prior = CAMPAIGN_EXHAUSTED[0]
    decision = admit(
        _prop("fusing tiny kernels into the following GEMV"),
        attempts=[prior],
        include_campaign_exhausted=False,
    )
    assert decision.verdict == Verdict.REFUSE
    assert decision.gate == "semantic_duplicate"
    assert decision.changed == []


def test_malformed_science_goes_red() -> None:
    decision = admit(
        _prop("gaussian activations"),
        science=[{"id": "NS-009", "mechanism": "gaussian activations"}],
    )
    assert decision.verdict == Verdict.ERROR
    assert decision.gate == "negative_science"
    assert decision.red


def test_unknown_science_class_goes_red() -> None:
    decision = admit(
        _prop("gaussian activations"),
        science=[
            {
                "id": "NS-X",
                "mechanism": "gaussian activations",
                "class": "HAPPY",
                "models": ["qwen38"],
            }
        ],
    )
    assert decision.verdict == Verdict.ERROR
    assert decision.red


def test_hollow_running_record_goes_red() -> None:
    decision = admit(
        _prop("assign codecs per layer and per head"),
        running=[Attempt(id="live-1", mechanism="", status="running")],
    )
    assert decision.verdict == Verdict.ERROR
    assert decision.gate == "inflight_duplicate"
    assert decision.red


def test_corrupt_ledger_goes_red() -> None:
    decision = synthesize(measured={"schema": "broken"}, include_campaign_exhausted=True)
    assert decision.verdict == Verdict.ERROR
    assert decision.gate == "synthesis_needs_bottleneck"
    assert decision.red


def test_missing_ns_on_a_component_goes_red() -> None:
    decision = synthesize(
        measured=[{"component": "weight_addressing"}],
        include_campaign_exhausted=True,
    )
    assert decision.verdict == Verdict.ERROR
    assert decision.red


# ---------------------------------------------------------------- receipt


def test_write_required_evidence_receipt() -> None:
    receipt = emit_evidence(RECEIPT_DIR / "MECHANISM_DEDUP_EVIDENCE.json")
    path = Path(receipt["_path"])
    assert path.is_file()
    body = json.loads(path.read_text())
    demo = body["demonstrations"]
    assert demo["semantic_duplicate_refused"]["verdict"] == "REFUSE"
    assert demo["semantic_duplicate_refused"]["matched_prior_id"] == "wa-exhausted-01"
    assert demo["legitimate_retry_allowed"]["verdict"] == "ALLOW"
    assert "precondition" in demo["legitimate_retry_allowed"]["changed"]
    assert demo["synthesis_from_dry_queue"]["verdict"] == "SYNTHESIZE"
    assert demo["synthesis_from_dry_queue"]["new_mechanism_id"] == "per_layer_per_head_assignment"
    assert body["all_corrupt_gates_red"] is True
    assert DEFAULT_RECEIPT == path or path.name == "MECHANISM_DEDUP_EVIDENCE.json"


def test_science_from_mapping_rejects_a_hollow_entry() -> None:
    with pytest.raises(ValueError):
        coerce_science({"id": "NS-X", "mechanism": "x"})


def test_campaign_exhausted_is_exactly_nine() -> None:
    assert len(CAMPAIGN_EXHAUSTED) == 9
    # Mechanism 1 must be the GEMV fusion regression — synthesis must not replay it.
    assert "gemv" in CAMPAIGN_EXHAUSTED[0].mechanism.lower()
