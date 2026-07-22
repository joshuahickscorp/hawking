from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_state as gs  # noqa: E402
import glm52_window_execution as wx  # noqa: E402
from glm52_common import seal  # noqa: E402
from glm52_grounding import (  # noqa: E402
    ProducerAuthenticator,
    ResourceReservePolicy,
    sample_resources,
)
from glm52_state import EvidenceAuthConfig  # noqa: E402


CAMPAIGN = "glm52-window-execution-test"
REVISION = "a" * 40
ANCHOR = "b" * 64
HASH = "c" * 64
AUTH = wx.WindowExecutionAuthenticators(
    grounding=ProducerAuthenticator(
        b"window-execution-test-grounding-key-minimum-32-bytes"
    ),
    evidence=EvidenceAuthConfig(
        hmac_key=b"window-execution-test-evidence-key-minimum-32-bytes",
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
    ),
)
OTHER_AUTH = wx.WindowExecutionAuthenticators(
    grounding=ProducerAuthenticator(
        b"window-execution-other-grounding-key-minimum-32-bytes"
    ),
    evidence=EvidenceAuthConfig(
        hmac_key=b"window-execution-other-evidence-key-minimum-32-bytes",
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
    ),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gate_policy(path: str) -> dict[str, object]:
    return {
        "path": path,
        "expected_seal_sha256": HASH,
        "expected_schema": "test.window.execution.evidence.v1",
        "allowed_statuses": ["PASS"],
        "validator_id": "sealed_exact_v1",
        "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            "sealed_exact_v1"
        ],
        "require_producer_hmac": False,
    }


def _state_gates() -> dict[str, dict[str, object]]:
    return {
        "ASSEMBLE_ARTIFACT": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": {"source": _gate_policy("evidence/source.json")},
            "required_checklist": {},
        },
        "COMPLETE": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": True,
            "required_phone_status_path": "evidence/phone.json",
            "required_artifacts": {"final": _gate_policy("evidence/final.json")},
            "required_checklist": {"closed": _gate_policy("evidence/closed.json")},
        },
    }


def _contract(payloads: dict[str, bytes]) -> dict[str, object]:
    names = list(payloads)
    shards = [
        {
            "path": path,
            "logical_bytes": len(payloads[path]),
            "xet_hash": hashlib.sha256(f"xet:{path}".encode()).hexdigest(),
            "lfs_sha256": _sha(payloads[path]),
        }
        for path in names
    ]
    return gs.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest="d" * 64,
        source_shards=shards,
        expected_tensors=["tensor.a", "tensor.b"],
        window_schedule=[
            {
                "schedule_index": 0,
                "window_id": "W000",
                "source_shards": names[:2],
                "carry_in_shards": [],
                "new_fetch_shards": names[:2],
                "refetch_shards": [],
                "carry_out_shards": [names[1]],
                "evict_shards": [names[0]],
                "tensor_set": ["tensor.a"],
            },
            {
                "schedule_index": 1,
                "window_id": "W001",
                "source_shards": names[1:],
                "carry_in_shards": [names[1]],
                "new_fetch_shards": [names[2]],
                "refetch_shards": [],
                "carry_out_shards": [],
                "evict_shards": names[1:],
                "tensor_set": ["tensor.b"],
            },
        ],
        state_gates=_state_gates(),
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at="2026-07-21T00:00:00Z",
    )


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, object]:
    payloads = {
        "model-00001-of-00003.safetensors": b"source-one-exact",
        "model-00002-of-00003.safetensors": b"source-two-carry",
        "model-00003-of-00003.safetensors": b"source-three-prefetch",
    }
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    source_root.mkdir()
    artifact_root.mkdir()
    for name, payload in payloads.items():
        (source_root / name).write_bytes(payload)
    contract = _contract(payloads)
    policy = ResourceReservePolicy(emergency_floor_bytes=0)
    resource = sample_resources(
        source_root,
        root_id=wx.source_root_id(contract),
        policy=policy,
        authenticator=AUTH.grounding,
    )
    terminal_policy = wx.make_terminal_policy(
        contract,
        0,
        entries=[
            {
                "tensor_name": "tensor.a",
                "disposition": wx.COMPACT_DISPOSITION,
                "payload_class": "CORE",
            }
        ],
        authenticator=AUTH,
    )
    fetch_intent = wx.make_fetch_intent_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        resource_policy_receipt=resource,
        expected_resource_policy=policy,
        terminal_policy=terminal_policy,
        authenticator=AUTH,
    )
    return {
        "tmp": tmp_path,
        "payloads": payloads,
        "source_root": source_root,
        "artifact_root": artifact_root,
        "contract": contract,
        "policy": policy,
        "resource": resource,
        "terminal_policy": terminal_policy,
        "fetch_intent": fetch_intent,
    }


def _artifact_manifest(
    c: dict[str, object], phase: str, directory: str, filename: str, data: bytes, kind: str
) -> dict[str, object]:
    root = c["artifact_root"]
    assert isinstance(root, Path)
    target_dir = root / directory
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / filename).write_bytes(data)
    return wx.make_artifact_manifest(
        c["contract"],
        0,
        phase=phase,
        artifact_directory=directory,
        files=[
            {
                "relative_path": filename,
                "logical_bytes": len(data),
                "sha256": _sha(data),
                "artifact_kind": kind,
            }
        ],
        authenticator=AUTH,
    )


def _through_forward(c: dict[str, object]) -> dict[str, object]:
    contract = c["contract"]
    source_root = c["source_root"]
    artifact_root = c["artifact_root"]
    fetched = wx.make_fetch_committed_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["fetch_intent"],
        source_root=source_root,
        authenticator=AUTH,
    )
    verified = wx.make_sources_verified_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=fetched,
        source_root=source_root,
        authenticator=AUTH,
    )
    teacher_manifest = _artifact_manifest(
        c, "TEACHER_CAPTURED", "teacher/W000", "teacher.bin", b"teacher", "TEACHER_EVIDENCE"
    )
    teacher = wx.make_teacher_captured_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=verified,
        artifact_root=artifact_root,
        artifact_manifest=teacher_manifest,
        authenticator=AUTH,
    )
    fit_manifest = _artifact_manifest(
        c, "CANDIDATES_FIT", "fit/W000", "fit.bin", b"fit-result", "FIT_RESULT"
    )
    fit = wx.make_candidates_fit_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=teacher,
        artifact_root=artifact_root,
        artifact_manifest=fit_manifest,
        authenticator=AUTH,
    )
    packed_manifest = _artifact_manifest(
        c,
        "CANDIDATES_PACKED",
        "packed/W000",
        "compact.bin",
        b"0123456789abcdef",
        "COMPACT_PAYLOAD",
    )
    packed = wx.make_candidates_packed_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=fit,
        artifact_root=artifact_root,
        artifact_manifest=packed_manifest,
        authenticator=AUTH,
    )
    forward_manifest = _artifact_manifest(
        c,
        "FORWARD_COMPLETE",
        "forward/W000",
        "metrics.json",
        b'{"loss":1.25}',
        "FORWARD_METRICS",
    )
    forward = wx.make_forward_complete_receipt(
        contract,
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=packed,
        artifact_root=artifact_root,
        artifact_manifest=forward_manifest,
        authenticator=AUTH,
    )
    c.update(
        {
            "fetched": fetched,
            "verified": verified,
            "teacher": teacher,
            "fit": fit,
            "packed_manifest": packed_manifest,
            "packed": packed,
            "forward_manifest": forward_manifest,
            "forward": forward,
        }
    )
    return c


def _sealed(c: dict[str, object]) -> dict[str, object]:
    if "forward" not in c:
        _through_forward(c)
    coverage = wx.make_terminal_coverage_manifest(
        c["contract"],
        0,
        terminal_policy=c["terminal_policy"],
        packed_manifest=c["packed_manifest"],
        entries=[
            {
                "tensor_name": "tensor.a",
                "disposition": wx.COMPACT_DISPOSITION,
                "payload_class": "CORE",
                "payload_relative_path": "compact.bin",
                "byte_offset": 0,
                "byte_length": 8,
            }
        ],
        authenticator=AUTH,
    )
    sealed_receipt = wx.make_window_sealed_receipt(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["forward"],
        receipt_chain=[
            c["fetch_intent"],
            c["fetched"],
            c["verified"],
            c["teacher"],
            c["fit"],
            c["packed"],
            c["forward"],
        ],
        expected_resource_policy=c["policy"],
        terminal_policy=c["terminal_policy"],
        packed_manifest=c["packed_manifest"],
        terminal_coverage_manifest=coverage,
        source_root=c["source_root"],
        artifact_root=c["artifact_root"],
        authenticator=AUTH,
    )
    c.update({"coverage": coverage, "sealed": sealed_receipt})
    return c


def test_declared_window_is_derived_only_from_validated_contract(campaign) -> None:
    declared = wx.derive_declared_window(campaign["contract"], 0)
    assert declared["window_id"] == "W000"
    assert [item["path"] for item in declared["source_shards"]] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
    ]
    tampered = copy.deepcopy(declared)
    tampered["evict_shards"] = tampered["carry_out_shards"]
    tampered = seal(tampered)
    with pytest.raises(wx.WindowExecutionError, match="authoritative"):
        wx.validate_declared_window(tampered, campaign["contract"], 0)


def test_full_phase_chain_and_transactional_eviction_preserve_other_paths(campaign) -> None:
    c = _sealed(campaign)
    journal = c["tmp"] / "eviction.jsonl"
    resource = sample_resources(
        c["source_root"],
        root_id=wx.source_root_id(c["contract"]),
        policy=c["policy"],
        authenticator=AUTH.grounding,
    )
    committed = wx._execute_eviction_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    assert committed["phase"] == "EVICTION_COMMITTED"
    assert not (c["source_root"] / "model-00001-of-00003.safetensors").exists()
    assert (c["source_root"] / "model-00002-of-00003.safetensors").exists()
    assert (c["source_root"] / "model-00003-of-00003.safetensors").exists()
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [item["event_kind"] for item in events] == [
        "EVICTION_INTENT",
        "EVICTION_QUARANTINED",
        "EVICTION_COMMITTED",
    ]
    again = wx._execute_eviction_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    assert again == committed
    assert len(journal.read_text().splitlines()) == 3


def test_phase_receipts_are_explicitly_non_authoritative_and_public_eviction_refuses(
    campaign,
) -> None:
    c = _sealed(campaign)
    for receipt in (
        c["fetch_intent"],
        c["fetched"],
        c["verified"],
        c["teacher"],
        c["fit"],
        c["packed"],
        c["forward"],
        c["sealed"],
    ):
        assert receipt["status"] == "EVIDENCE_ONLY"
        assert receipt["authority_status"] == wx.NON_AUTHORITATIVE_STATUS
        assert receipt["campaign_stop_condition_eligible"] is False
        assert receipt["destructive_authority_eligible"] is False
    with pytest.raises(wx.WindowExecutionError, match="production eviction is disabled"):
        wx.execute_eviction()
    with pytest.raises(wx.WindowExecutionError, match="production eviction is disabled"):
        wx.prepare_eviction_intent()
    with pytest.raises(wx.WindowExecutionError, match="production eviction is disabled"):
        wx.reconcile_eviction()


def test_public_validation_has_no_orphan_bypass(campaign) -> None:
    fetched = wx.make_fetch_committed_receipt(
        campaign["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=campaign["fetch_intent"],
        source_root=campaign["source_root"],
        authenticator=AUTH,
    )
    with pytest.raises(wx.WindowExecutionError, match="requires its previous"):
        wx.validate_fetch_committed_receipt(
            fetched, campaign["contract"], None, AUTH
        )
    with pytest.raises(TypeError, match="allow_unlinked_previous"):
        wx.validate_fetch_committed_receipt(
            fetched,
            campaign["contract"],
            None,
            AUTH,
            allow_unlinked_previous=True,
        )


def test_authenticator_roles_reject_same_key_material() -> None:
    reused = b"same-window-execution-key-material-minimum-32-bytes"
    with pytest.raises(wx.WindowExecutionError, match="must be different"):
        wx.WindowExecutionAuthenticators(
            grounding=ProducerAuthenticator(reused),
            evidence=EvidenceAuthConfig(
                hmac_key=reused,
                campaign_id=CAMPAIGN,
                source_revision=REVISION,
            ),
        )


def test_window_seal_regrounds_every_prerequisite_artifact(campaign) -> None:
    c = _through_forward(campaign)
    (c["artifact_root"] / "teacher/W000/teacher.bin").unlink()
    with pytest.raises(Exception, match="manifest|inspect|directory"):
        _sealed(c)


def test_eviction_journal_read_is_bounded(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "oversized.jsonl"
    path.write_bytes(b"x" * 65)
    fd = os.open(path, os.O_RDONLY)
    try:
        monkeypatch.setattr(wx, "MAX_EVICTION_JOURNAL_BYTES", 64)
        with pytest.raises(wx.WindowExecutionError, match="bounded read limit"):
            wx._read_eviction_journal_fd(fd)
    finally:
        os.close(fd)


def test_source_grounding_rejects_missing_wrong_hash_and_same_size_corruption(campaign) -> None:
    target = campaign["source_root"] / "model-00001-of-00003.safetensors"
    original = target.read_bytes()
    target.unlink()
    with pytest.raises(Exception, match="cannot inspect|grounded file"):
        wx.make_fetch_committed_receipt(
            campaign["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=campaign["fetch_intent"],
            source_root=campaign["source_root"],
            authenticator=AUTH,
        )
    target.write_bytes(b"X" * len(original))
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        wx.make_fetch_committed_receipt(
            campaign["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=campaign["fetch_intent"],
            source_root=campaign["source_root"],
            authenticator=AUTH,
        )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_source_grounding_rejects_symlink_and_hardlink(campaign, kind: str) -> None:
    root = campaign["source_root"]
    target = root / "model-00001-of-00003.safetensors"
    original = target.read_bytes()
    if kind == "symlink":
        outside = campaign["tmp"] / "outside.bin"
        outside.write_bytes(original)
        target.unlink()
        target.symlink_to(outside)
        match = "symlink"
    else:
        os.link(target, root / "hardlink-alias")
        match = "hard link"
    with pytest.raises(Exception, match=match):
        wx.make_fetch_committed_receipt(
            campaign["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=campaign["fetch_intent"],
            source_root=root,
            authenticator=AUTH,
        )


def test_traversal_is_rejected_for_contract_and_artifact_manifests(campaign) -> None:
    bad = copy.deepcopy(campaign["contract"])
    old = bad["source"]["shards"][0]["path"]
    bad["source"]["shards"][0]["path"] = "../escape.safetensors"
    for field in ("source_shards", "new_fetch_shards", "evict_shards"):
        bad["window_schedule"][0][field] = [
            "../escape.safetensors" if item == old else item
            for item in bad["window_schedule"][0][field]
        ]
    bad = seal(bad)
    with pytest.raises(wx.WindowExecutionError, match="traverse|relative"):
        wx.derive_declared_window(bad, 0)
    with pytest.raises(wx.WindowExecutionError, match="artifact_directory"):
        wx.make_artifact_manifest(
            campaign["contract"],
            0,
            phase="TEACHER_CAPTURED",
            artifact_directory="../escape",
            files=[
                {
                    "relative_path": "file.bin",
                    "logical_bytes": 1,
                    "sha256": _sha(b"x"),
                    "artifact_kind": "TEACHER_EVIDENCE",
                }
            ],
            authenticator=AUTH,
        )


def test_resource_receipt_must_match_exact_required_policy(campaign) -> None:
    stronger = ResourceReservePolicy(
        emergency_floor_bytes=0,
        largest_atomic_source_write_bytes=1,
    )
    with pytest.raises(wx.WindowExecutionError, match="differs from the required policy"):
        wx.make_fetch_intent_receipt(
            campaign["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            resource_policy_receipt=campaign["resource"],
            expected_resource_policy=stronger,
            terminal_policy=campaign["terminal_policy"],
            authenticator=AUTH,
        )


def test_phase_chain_rejects_anchor_predecessor_and_producer_replay(campaign) -> None:
    fetched = wx.make_fetch_committed_receipt(
        campaign["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=campaign["fetch_intent"],
        source_root=campaign["source_root"],
        authenticator=AUTH,
    )
    with pytest.raises(wx.WindowExecutionError, match="controller.*anchor"):
        wx.make_sources_verified_receipt(
            campaign["contract"],
            0,
            controller_anchor_sha256="e" * 64,
            previous_receipt=fetched,
            source_root=campaign["source_root"],
            authenticator=AUTH,
        )
    with pytest.raises(wx.WindowExecutionError, match="predecessor phase"):
        wx.validate_phase_receipt(
            fetched,
            campaign["contract"],
            authenticator=AUTH,
            previous_receipt=fetched,
        )
    with pytest.raises(wx.WindowExecutionError, match="producer identity"):
        wx.validate_fetch_intent_receipt(
            campaign["fetch_intent"], campaign["contract"], OTHER_AUTH
        )


def test_artifact_manifest_rejects_fake_metrics_extra_files_and_corruption(campaign) -> None:
    c = _through_forward(campaign)
    forward_dir = c["artifact_root"] / "forward/W000"
    (forward_dir / "unlisted-metrics.json").write_text('{"loss":0}')
    with pytest.raises(wx.WindowExecutionError, match="exact manifest"):
        wx.make_forward_complete_receipt(
            c["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=c["packed"],
            artifact_root=c["artifact_root"],
            artifact_manifest=c["forward_manifest"],
            authenticator=AUTH,
        )
    (forward_dir / "unlisted-metrics.json").unlink()
    target = forward_dir / "metrics.json"
    original = target.read_bytes()
    target.write_bytes(b"X" * len(original))
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        wx.make_forward_complete_receipt(
            c["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=c["packed"],
            artifact_root=c["artifact_root"],
            artifact_manifest=c["forward_manifest"],
            authenticator=AUTH,
        )


def test_terminal_coverage_rejects_gaps_fake_billing_and_unjustified_omission(campaign) -> None:
    packed = _artifact_manifest(
        campaign,
        "CANDIDATES_PACKED",
        "packed-only/W000",
        "compact.bin",
        b"0123456789abcdef",
        "COMPACT_PAYLOAD",
    )
    with pytest.raises(wx.WindowExecutionError, match="every scheduled tensor"):
        wx.make_terminal_policy(
            campaign["contract"], 0, entries=[], authenticator=AUTH
        )

    source_path = "model-00001-of-00003.safetensors"
    protected = wx.make_terminal_policy(
        campaign["contract"],
        0,
        entries=[
            {
                "tensor_name": "tensor.a",
                "disposition": wx.PROTECTED_DISPOSITION,
                "native_source_path": source_path,
                "billed_bytes": len(campaign["payloads"][source_path]) + 1,
                "protection_justification": "native protection preregistration",
            }
        ],
        authenticator=AUTH,
    )
    with pytest.raises(wx.WindowExecutionError, match="exceed"):
        protected_coverage = copy.deepcopy(protected["entries"])
        protected_coverage[0].update(
            {
                "payload_relative_path": "compact.bin",
                "byte_offset": 0,
                "byte_length": protected_coverage[0]["billed_bytes"],
            }
        )
        wx.make_terminal_coverage_manifest(
            campaign["contract"],
            0,
            terminal_policy=protected,
            packed_manifest=packed,
            entries=protected_coverage,
            authenticator=AUTH,
        )

    omitted = wx.make_terminal_policy(
        campaign["contract"],
        0,
        entries=[
            {
                "tensor_name": "tensor.a",
                "disposition": wx.OMITTED_DISPOSITION,
                "capability_justification": "preregistered unsupported operator",
                "justification_evidence_sha256": "f" * 64,
            }
        ],
        authenticator=AUTH,
    )
    unjustified = copy.deepcopy(omitted["entries"])
    unjustified[0]["capability_justification"] = "invented after execution"
    with pytest.raises(wx.WindowExecutionError, match="preregistered"):
        wx.make_terminal_coverage_manifest(
            campaign["contract"],
            0,
            terminal_policy=omitted,
            packed_manifest=packed,
            entries=unjustified,
            authenticator=AUTH,
        )


def test_partial_eviction_reconciles_but_identity_swap_fails_closed(campaign) -> None:
    c = _sealed(campaign)
    journal = c["tmp"] / "partial.jsonl"
    resource = sample_resources(
        c["source_root"],
        root_id=wx.source_root_id(c["contract"]),
        policy=c["policy"],
        authenticator=AUTH.grounding,
    )
    intent = wx._prepare_eviction_intent_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    target = c["source_root"] / "model-00001-of-00003.safetensors"
    quarantine = c["source_root"] / wx._quarantine_path(
        target.name, intent["seal_sha256"]
    )
    target.rename(quarantine)
    committed = wx._reconcile_eviction_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    assert committed["evidence"]["reconciled_previously_absent_shards"] == []


def test_reconcile_recovers_absence_after_durable_quarantine_event(campaign) -> None:
    c = _sealed(campaign)
    journal = c["tmp"] / "durable-quarantine-crash.jsonl"
    resource = sample_resources(
        c["source_root"],
        root_id=wx.source_root_id(c["contract"]),
        policy=c["policy"],
        authenticator=AUTH.grounding,
    )
    intent = wx._prepare_eviction_intent_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    target = c["source_root"] / "model-00001-of-00003.safetensors"
    quarantine = c["source_root"] / wx._quarantine_path(
        target.name, intent["seal_sha256"]
    )
    target.rename(quarantine)

    quarantine_entries = wx._quarantine_entries(intent)
    quarantine_event = wx._make_eviction_event(
        "EVICTION_QUARANTINED",
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_event_sha256=intent["seal_sha256"],
        payload={"quarantined_sources": quarantine_entries},
        authenticator=AUTH,
    )
    (
        parent_fds,
        parent_fd,
        journal_fd,
        journal_leaf,
        parent_links,
        parent_root_stat,
    ) = wx._open_eviction_journal(journal)
    try:
        wx._append_eviction_event(
            journal_fd,
            parent_fd,
            journal_leaf,
            quarantine_event,
            parent_fds,
            parent_links,
            parent_root_stat,
        )
    finally:
        os.close(journal_fd)
        for item in reversed(parent_fds):
            os.close(item)

    # Simulate a crash after the durable quarantine record and unlink, but
    # before the EVICTION_COMMITTED record is written.
    quarantine.unlink()
    committed = wx._reconcile_eviction_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    assert committed["evidence"]["reconciled_previously_absent_shards"] == [
        target.name
    ]


def test_eviction_rejects_same_content_identity_swap_and_journal_replay(campaign) -> None:
    c = _sealed(campaign)
    journal = c["tmp"] / "swap.jsonl"
    resource = sample_resources(
        c["source_root"],
        root_id=wx.source_root_id(c["contract"]),
        policy=c["policy"],
        authenticator=AUTH.grounding,
    )
    wx._prepare_eviction_intent_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    target = c["source_root"] / "model-00001-of-00003.safetensors"
    data = target.read_bytes()
    target.unlink()
    target.write_bytes(data)
    with pytest.raises(wx.WindowExecutionError, match="identity changed"):
        wx._reconcile_eviction_test_only(
            c["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=c["sealed"],
            source_root=c["source_root"],
            journal_path=journal,
            resource_policy_receipt=resource,
            expected_resource_policy=c["policy"],
            authenticator=AUTH,
        )


def test_eviction_rejects_target_renamed_to_unlisted_stash(campaign) -> None:
    c = _sealed(campaign)
    journal = c["tmp"] / "rename-stash.jsonl"
    resource = sample_resources(
        c["source_root"],
        root_id=wx.source_root_id(c["contract"]),
        policy=c["policy"],
        authenticator=AUTH.grounding,
    )
    wx._prepare_eviction_intent_test_only(
        c["contract"],
        0,
        controller_anchor_sha256=ANCHOR,
        previous_receipt=c["sealed"],
        source_root=c["source_root"],
        journal_path=journal,
        resource_policy_receipt=resource,
        expected_resource_policy=c["policy"],
        authenticator=AUTH,
    )
    target = c["source_root"] / "model-00001-of-00003.safetensors"
    target.rename(c["source_root"] / "unlisted-stash.bin")
    with pytest.raises(wx.WindowExecutionError, match="vanished"):
        wx._reconcile_eviction_test_only(
            c["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=c["sealed"],
            source_root=c["source_root"],
            journal_path=journal,
            resource_policy_receipt=resource,
            expected_resource_policy=c["policy"],
            authenticator=AUTH,
        )
    first_line = journal.read_bytes()
    journal.write_bytes(first_line + first_line)
    with pytest.raises(wx.WindowExecutionError, match="chain|begin|replay|order"):
        wx._reconcile_eviction_test_only(
            c["contract"],
            0,
            controller_anchor_sha256=ANCHOR,
            previous_receipt=c["sealed"],
            source_root=c["source_root"],
            journal_path=journal,
            resource_policy_receipt=resource,
            expected_resource_policy=c["policy"],
            authenticator=AUTH,
        )
