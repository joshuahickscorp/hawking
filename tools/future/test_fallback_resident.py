"""Fallback resident: restorable-now must be able to return false.

Negative controls that must actually fire:
- a missing artifact is NOT_RESTORABLE and names the artifact
- a present file whose digest does not match the seal is not the fallback
- rollback_state names durable science it will not revert

Live-host assertions cope with either restorable or not; they never skip.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.future import fallback_resident as fb
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.resident_identity import RESIDENCY_STATUS
from tools.future.super_resident import QWEN_ROLE, REL_QWEN_IDENTITY


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _matching_world(tmp_path: Path) -> dict:
    """A complete overlay whose digests match the injected seal prefixes."""
    art = tmp_path / "NOETIC_PARENT_A"
    art.mkdir(exist_ok=True)
    tok = b"sealed-tokenizer-bytes\n"
    chat = b"{% chat template %}\n"
    greedy = b"hybrid-greedy-binary\n"
    resident = b"resident-protocol-binary\n"
    catalog = art / "catalog.hq38m20"
    catalog.write_text("catalog-identity\n")
    (art / "tokenizer.json").write_bytes(tok)
    (art / "chat_template.jinja").write_bytes(chat)
    greedy_p = tmp_path / "ascension_qwen38_hybrid_greedy"
    resident_p = tmp_path / "ascension_qwen38_resident"
    greedy_p.write_bytes(greedy)
    resident_p.write_bytes(resident)
    parent_params = 26895998464
    mix = {
        "mix_id": "mix_all_mlp_affine_g64_ls",
        "parent_params": parent_params,
        "did_not_load_second_27b": True,
        "recipe": {"id": "mix_all_mlp_affine_g64_ls"},
        "catalog": str(catalog),
        "artifact_root": str(art),
    }
    (art / "MIX_REPORT.json").write_text(json.dumps(mix))
    ident = {
        "model_id": "qwen3.8-27b-sealed-3.14",
        "resident_identity": "sealed-3.14",
        "family": "qwen3.8",
        "protocol": "hawking.qwen38.resident.v1",
        "runtime": "hawking-native",
        "provider": "native",
        "mode": "auto",
        "artifact_root": str(art),
        "tokenizer": str(art / "tokenizer.json"),
        "binary": str(greedy_p),
        "resident_binary": str(resident_p),
        "prompt_contract": {"renderer": "qwen-chat-template-or-closed-think-fallback"},
    }
    seal = {
        "status": "SEALED",
        "resident": "sealed-3.14",
        "fields": {
            "tokenizer_sha256_16": {"value": _sha(tok)[:16]},
            "runtime_binary_sha256_16": {"value": _sha(greedy)[:16]},
            "chat_template_sha256_16": {"value": _sha(chat)[:16]},
            "physical_closure": {"value": {"parent_params": parent_params}},
            "runtime_commit": {"value": "5e365a5c08c3497d8ed7553d9fb00eeb2cdbc07f"},
            "artifact_root": {"value": str(art)},
            "runtime_binary": {"value": str(greedy_p)},
        },
    }
    return {
        "identity_document": {"exists": True, "source": "fixture", "path": str(tmp_path / "ident.json"), "doc": ident},
        "seal": {"exists": True, "source": "fixture", "path": str(tmp_path / "seal.json"), "doc": seal},
        "artifact_root": {"path": str(art), "exists": True, "is_dir": True, "source": "fixture"},
        "tokenizer": {"path": str(art / "tokenizer.json")},
        "chat_template": {"path": str(art / "chat_template.jinja")},
        "runtime_binary": {"path": str(greedy_p)},
        "resident_binary": {"path": str(resident_p)},
        "mix_report": {"path": str(art / "MIX_REPORT.json"), "exists": True, "doc": mix, "source": "fixture"},
        "catalog": {"path": str(catalog)},
        "resident_gate": {"exists": True, "in_git_head": True, "source": "fixture"},
        "connector": {"exists": True, "in_git_head": True, "source": "fixture"},
        "recovery": {"exists": True, "in_git_head": True, "source": "fixture"},
    }


def test_build_emits_sealed_receipt():
    out = fb.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FALLBACK_RESIDENT.json"
    assert doc["schema"] == fb.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["performed_restore"] is False
    assert doc["started_model_process"] is False
    assert doc["took_gpu_lease"] is False
    assert doc["flock"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"] == "tools.future.fallback_resident.verify_restorable()"
    assert doc["resident_callable"]["receipt"] == "receipts/future/FALLBACK_RESIDENT.json"
    assert doc["resident_callable"]["frontier"] == "FT.CHILD_RESIDENT.install-dry-run"
    assert doc["resident_callable"]["fails_closed"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)


def test_selftest_emits_sealed_receipt_and_fires_missing_control():
    out = fb.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["negative_control"]["missing_artifact_must_be_not_restorable"] is True


def test_fallback_identity_is_qwen27_current_nonfinal(tmp_path):
    ident = fb.fallback_identity(_matching_world(tmp_path))
    assert ident["status"] == "SEALED"
    assert ident["role"] == QWEN_ROLE
    assert ident["residency_status"] == RESIDENCY_STATUS
    assert ident["not_a_singularity"] is True
    assert ident["id"] == "qwen3.8-27b-sealed-3.14"
    assert ident["artifact_path"]
    assert ident["specimen_identity"]["mix_id"] == "mix_all_mlp_affine_g64_ls"
    assert isinstance(ident["config_digest"], str) and len(ident["config_digest"]) == 64
    assert ident["copied_runtime_tps"] is False
    assert ident["copied_physical_ebpw"] is False
    assert ident["performed_restore"] is False
    assert ident["gpu_authority"] is False
    assert "complete_tps" not in json.dumps(ident)
    assert ident["tokenizer"]["prefix_verdict"] == "MATCH"
    assert ident["runtime_binary"]["prefix_verdict"] == "MATCH"


def test_live_identity_copes_without_skipping():
    """Sparse checkout or full host: the collector records which path it took."""
    ident = fb.fallback_identity()
    assert ident["role"] == QWEN_ROLE
    assert ident["residency_status"] == RESIDENCY_STATUS
    assert ident["gpu_authority"] is False
    assert ident["status"] in {"SEALED", "IDENTITY_UNRESOLVED"}
    presence = ident["identity_presence"]
    assert presence["rel"] == REL_QWEN_IDENTITY or presence.get("recovery")
    if ident["status"] == "SEALED":
        assert ident["config_digest"]
        assert ident["copied_runtime_tps"] is False


def test_missing_artifact_is_not_restorable_and_names_it(tmp_path):
    world = _matching_world(tmp_path)
    world["tokenizer"] = {
        "path": str(tmp_path / "gone" / "tokenizer.json"),
        "exists": False,
        "sha256": fb.UNKNOWN,
        "source": "fixture",
    }
    verdict = fb.verify_restorable(world)
    assert verdict["verdict"] == fb.VERDICT_NOT
    assert verdict["restorable"] is False
    assert verdict["unmet_precondition"]
    assert "tokenizer" in str(verdict["unmet_precondition"])
    assert verdict["performed_restore"] is False
    named = [r for r in verdict["preconditions"] if r["id"] == "tokenizer"][0]
    assert named["state"] == "UNMET"
    assert named["kind"] == "MISSING"
    assert named["names"] == world["tokenizer"]["path"]


def test_missing_artifact_root_is_not_restorable(tmp_path):
    world = _matching_world(tmp_path)
    world["artifact_root"] = {
        "path": str(tmp_path / "missing-root"),
        "exists": False,
        "is_dir": False,
        "source": "fixture",
    }
    world["mix_report"] = {"path": str(tmp_path / "missing-root" / "MIX_REPORT.json"), "exists": False, "doc": None}
    verdict = fb.verify_restorable(world)
    assert verdict["verdict"] == fb.VERDICT_NOT
    assert verdict["unmet_precondition"]
    blob = json.dumps(verdict)
    assert "missing-root" in blob or "artifact" in str(verdict["reason"]).lower() or "MIX_REPORT" in blob


def test_present_wrong_digest_is_not_the_fallback(tmp_path):
    world = _matching_world(tmp_path)
    ident_doc = world["identity_document"]["doc"]
    tok_path = Path(ident_doc["tokenizer"])
    tok_path.write_bytes(b"I am a different model pretending to sit at the sealed path\n")
    # Drop overlay sha so the observer hashes the real (now wrong) file.
    world["tokenizer"] = {"path": str(tok_path)}
    ident = fb.fallback_identity(world)
    assert ident["tokenizer"]["exists"] is True
    assert ident["tokenizer"]["prefix_verdict"] == "MISMATCH"
    verdict = fb.verify_restorable(world, identity=ident)
    assert verdict["verdict"] == fb.VERDICT_NOT
    assert verdict["restorable"] is False
    tok_row = [r for r in verdict["preconditions"] if r["id"] == "tokenizer"][0]
    assert tok_row["kind"] == "DIGEST_MISMATCH"
    assert str(tok_path) in tok_row["names"]
    assert "not the fallback" in tok_row["why"]
    # WITH_ACTION would be the silent-health failure mode. Mismatch must not take it.
    assert verdict["verdict"] != fb.VERDICT_ACTION
    assert verdict["action"] is None


def test_wrong_mix_at_artifact_root_is_not_restorable(tmp_path):
    world = _matching_world(tmp_path)
    mix = dict(world["mix_report"]["doc"])
    mix["parent_params"] = 1
    world["mix_report"] = {**world["mix_report"], "doc": mix}
    ident = fb.fallback_identity(world)
    assert ident["specimen_identity"]["parent_params_matches_seal"] is False
    verdict = fb.verify_restorable(world, identity=ident)
    assert verdict["verdict"] == fb.VERDICT_NOT
    spec = [r for r in verdict["preconditions"] if r["id"] == "specimen_mix"][0]
    assert spec["kind"] == "DIGEST_MISMATCH"
    assert "not the fallback" in spec["why"]


def test_restorable_now_on_matching_world(tmp_path):
    world = _matching_world(tmp_path)
    verdict = fb.verify_restorable(world)
    assert verdict["verdict"] == fb.VERDICT_NOW
    assert verdict["restorable"] is True
    assert verdict["unmet_precondition"] is None
    assert verdict["n_unmet"] == 0
    assert all(r["state"] == "MET" for r in verdict["preconditions"])
    assert verdict["performed_restore"] is False
    assert verdict["took_gpu_lease"] is False
    assert verdict["flock"] is False


def test_restorable_with_action_names_the_rebuild(tmp_path):
    world = _matching_world(tmp_path)
    world["runtime_binary"] = {
        "path": str(tmp_path / "missing-greedy"),
        "exists": False,
        "sha256": fb.UNKNOWN,
        "source": "fixture",
    }
    ident = fb.fallback_identity(world)
    verdict = fb.verify_restorable(world, identity=ident)
    assert verdict["verdict"] == fb.VERDICT_ACTION
    assert verdict["restorable"] is False
    assert verdict["action"]
    assert "rebuild" in verdict["action"]
    assert "greedy" in verdict["action"] or "hybrid" in verdict["action"]
    run = [r for r in verdict["preconditions"] if r["id"] == "runtime_binary"][0]
    assert run["state"] == "UNMET"
    assert run["kind"] == "MISSING"
    assert run["action"]


def test_verdicts_are_the_three_named_states(tmp_path):
    now_world = _matching_world(tmp_path)
    now = fb.verify_restorable(now_world)
    missing = fb.verify_restorable(fb._selftest_overlay_missing())
    now_world["runtime_binary"] = {"path": "/nonexistent/greedy", "exists": False, "sha256": fb.UNKNOWN}
    action = fb.verify_restorable(now_world)
    seen = {now["verdict"], missing["verdict"], action["verdict"]}
    assert seen == {fb.VERDICT_NOW, fb.VERDICT_NOT, fb.VERDICT_ACTION}
    assert set(fb.VERDICTS) == {fb.VERDICT_NOW, fb.VERDICT_NOT, fb.VERDICT_ACTION}


def test_restore_path_steps_are_independently_checkable(tmp_path):
    world = _matching_world(tmp_path)
    path = fb.restore_path(world)
    assert path["step_ids"] == list(fb.RESTORE_STEPS)
    assert path["n_steps"] == len(fb.RESTORE_STEPS)
    assert path["performed_restore"] is False
    assert path["started_model_process"] is False
    checkable = [s for s in path["steps"] if s["checkable_without_restore"]]
    executed = [s for s in path["steps"] if s["executes_restore"]]
    assert checkable
    assert executed
    for step in checkable:
        assert step["current_state"] in {"MET", "UNMET", "UNKNOWN"}
        assert step["executes_restore"] is False
    for step in executed:
        assert step["current_state"] == "NOT_EXECUTED"
        assert step["checkable_without_restore"] is False
    ids = [s["id"] for s in path["steps"]]
    assert ids == list(fb.RESTORE_STEPS)


def test_rollback_names_what_it_does_not_revert():
    rb = fb.rollback_state()
    assert rb["durable_science_survives"] is True
    assert rb["does_not_revert"]
    blob = " ".join(rb["does_not_revert"]).lower()
    assert "receipts/future" in blob
    assert "scar" in blob or "negative_science" in blob
    assert "git" in blob
    assert rb["reverts"]
    assert any("bound" in x.lower() or "resident" in x.lower() for x in rb["reverts"])
    assert rb["performed_restore"] is False
    assert rb["rollback_is_not_a_restore_until_executed"] is True
    # Explicit: durable science is not in the revert set.
    revert_blob = " ".join(rb["reverts"]).lower()
    assert "receipts/future" not in revert_blob
    assert "negative_science" not in revert_blob


def test_live_verify_copes_either_way_and_never_claims_a_restore():
    verdict = fb.verify_restorable()
    assert verdict["verdict"] in fb.VERDICTS
    assert verdict["performed_restore"] is False
    assert verdict["started_model_process"] is False
    assert verdict["took_gpu_lease"] is False
    assert verdict["flock"] is False
    assert verdict["gpu_authority"] is False
    assert verdict["evidence_class"] == "STATIC_ONLY"
    if verdict["verdict"] == fb.VERDICT_NOT:
        assert verdict["unmet_precondition"]
        assert verdict["restorable"] is False
    elif verdict["verdict"] == fb.VERDICT_ACTION:
        assert verdict["action"]
        assert verdict["restorable"] is False
    else:
        assert verdict["restorable"] is True
        assert verdict["n_unmet"] == 0


def test_receipt_contains_identity_restore_verify_rollback():
    doc = json.loads(fb.build().read_text())
    assert doc["identity"]["role"] == QWEN_ROLE
    assert doc["restore_path"]["steps"]
    assert doc["verify_restorable"]["verdict"] in fb.VERDICTS
    assert doc["rollback_state"]["does_not_revert"]
    assert doc["work_units"]
    assert doc["work_units"][0]["id"] == "future.fallback_resident.verify"
    assert doc["work_units"][0]["resource_class"] in {"STATIC_ANALYSIS", "static_analysis"}
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        # A numeric hardware key anywhere in the receipt is a campaign failure.
        hits = fb._hardware_numeric_keys(doc)
        assert hits == [], hits
        assert not isinstance(doc.get(key), (int, float))


def test_unresolved_identity_is_not_restorable():
    overlay = {
        "identity_document": {"exists": False, "path": "hcli/hawking-native.sealed-3.14.json", "doc": None},
        "seal": {"exists": False, "doc": None},
        "artifact_root": {"path": None, "exists": False, "is_dir": False},
        "tokenizer": {"path": None, "exists": False},
        "chat_template": {"path": None, "exists": False},
        "runtime_binary": {"path": None, "exists": False},
        "resident_binary": {"path": None, "exists": False},
        "mix_report": {"exists": False, "doc": None},
        "resident_gate": {"exists": False, "in_git_head": False},
        "connector": {"exists": False, "in_git_head": False},
        "recovery": {"exists": False, "in_git_head": False},
    }
    ident = fb.fallback_identity(overlay)
    assert ident["status"] == "IDENTITY_UNRESOLVED"
    verdict = fb.verify_restorable(overlay, identity=ident)
    assert verdict["verdict"] == fb.VERDICT_NOT
    assert verdict["unmet_precondition"]
    assert ident["performed_restore"] is False
