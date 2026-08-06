"""PROTO_FRANKENSTEIN_V0 SEAL: assembler + independent verify + cloud/restore."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_ablation as ablation  # noqa: E402
from lab.operators import frankenstein_bridges as bridges  # noqa: E402
from lab.operators import frankenstein_gates as gates  # noqa: E402
from lab.operators import frankenstein_promotion_gate as promo  # noqa: E402
from lab.operators import frankenstein_v0_seal as seal_v0  # noqa: E402
from lab.operators.frankenstein_gates import REQUIRES_TRAINED_MODULES  # noqa: E402
from lab.receipts import verify  # noqa: E402


def _module_pack(*, trained: bool = True, drop: str | None = None) -> dict:
    modules = {}
    for name in bridges.V0_MODULE_NAMES:
        if drop and name == drop:
            continue
        modules[name] = {
            "content_hash": "a" * 64,
            "parameter_bytes": 1024,
            "trained": trained,
            "reversible": True,
            "ablatable": True,
            "bypassable": True,
        }
    return {
        "trained": trained,
        "modules": modules,
        "schema": "test.module_pack",
    }


def _write_pack(tmp: Path, pack: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "MODULE_PACK.json"
    path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    return tmp


# ---------------------------------------------------------------------------
# Fail closed without trained modules
# ---------------------------------------------------------------------------


def test_assemble_fails_closed_without_modules(tmp_path: Path) -> None:
    result = seal_v0.assemble_v0_artifact(
        modules_path=None,
        out_dir=tmp_path,
        write=True,
    )
    assert result["status"] == "FAIL_CLOSED"
    assert result["gate"] == REQUIRES_TRAINED_MODULES
    assert result["loadable"] is False
    assert result["capability_claim"] is False
    receipt = result["receipt"]
    verify(receipt, label="assemble receipt")
    assert receipt["loadable_artifact_written"] is False
    assert not (tmp_path / seal_v0.ARTIFACT_NAME).exists()
    assert (tmp_path / seal_v0.ASSEMBLE_RECEIPT_NAME).exists()


def test_assemble_fails_closed_on_untrained_pack(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "mods", _module_pack(trained=False))
    result = seal_v0.assemble_v0_artifact(modules_path=pack_dir, out_dir=tmp_path / "out")
    assert result["status"] == "FAIL_CLOSED"
    assert result["loadable"] is False
    assert result["module_pack"]["trained"] is False


def test_assemble_fails_closed_on_incomplete_pack(tmp_path: Path) -> None:
    pack = _module_pack(trained=True, drop="GLM_VALUE_HEAD")
    pack_dir = _write_pack(tmp_path / "mods", pack)
    result = seal_v0.assemble_v0_artifact(modules_path=pack_dir, out_dir=tmp_path / "out")
    assert result["status"] == "FAIL_CLOSED"
    assert "GLM_VALUE_HEAD" in (result["module_pack"].get("missing_modules") or [])


def test_assemble_refuses_raw_pt_without_pack_json(tmp_path: Path) -> None:
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "BEST_BALANCED.pt").write_bytes(b"not-a-real-checkpoint")
    result = seal_v0.assemble_v0_artifact(modules_path=d, out_dir=tmp_path / "out")
    assert result["status"] == "FAIL_CLOSED"
    assert result["loadable"] is False


# ---------------------------------------------------------------------------
# Honest assemble with trained pack
# ---------------------------------------------------------------------------


def test_assemble_with_trained_pack(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "mods", _module_pack(trained=True))
    out = tmp_path / "out"
    result = seal_v0.assemble_v0_artifact(
        modules_path=pack_dir,
        out_dir=out,
        base_resident_bytes=8_000_000_000,
        tps=12.5,
        p99=80.0,
        active_bytes_per_token=4096,
    )
    assert result["status"] == "ASSEMBLED"
    assert result["loadable"] is True
    artifact = result["artifact"]
    verify(artifact, label="v0 artifact")
    assert artifact["schema"] == seal_v0.V0_ARTIFACT_SCHEMA
    assert artifact["trained"] is True
    assert artifact["reversible"] is True
    assert artifact["student_body"]["read_only"] is True
    assert artifact["composition"]["body_rewritten"] is False
    assert len(artifact["v0_modules"]) == len(bridges.V0_MODULE_NAMES)
    assert artifact["capability_claim"] is False
    assert artifact["proto_frankenstein_complete"] is False
    grav = artifact["gravity_accounting"]
    verify(grav, label="gravity")
    assert grav["parameter_bytes"] == 1024 * len(bridges.V0_MODULE_NAMES)
    assert grav["tps"] == 12.5
    assert grav["p99"] == 80.0
    assert grav["fabricated"] is False
    assert (out / seal_v0.ARTIFACT_NAME).is_file()
    assert artifact["bridges"]["kimi_stage1_status"] == "PRESERVED_UNTOUCHED"


def test_gravity_accounting_pending_tps_not_fabricated() -> None:
    pack = seal_v0.admit_trained_modules(None)
    # Use a synthetic admitted pack structure
    admitted = {
        "trained": True,
        "complete": True,
        "modules": {
            "GLM_METHOD_BRIDGE": {
                "content_hash": "b" * 64,
                "parameter_bytes": 100,
            }
        },
    }
    grav = seal_v0.build_gravity_accounting(module_pack=admitted)
    verify(grav, label="grav pending tps")
    assert grav["tps"] is None
    assert grav["tps_status"] == "PENDING"
    assert grav["parameter_bytes"] == 100
    assert grav["fabricated"] is False
    assert pack["trained"] is False  # none path still fail-closed


# ---------------------------------------------------------------------------
# Independent verify
# ---------------------------------------------------------------------------


def test_independent_verify_pending_without_scores(tmp_path: Path) -> None:
    result = seal_v0.independent_verify(out_dir=tmp_path, write=True)
    assert result["verdict"] == "PENDING"
    assert result["pass"] is False
    doc = result["document"]
    verify(doc, label="verify doc")
    assert doc["independent_of_training_lane"] is True
    assert doc["promotion_gate"]["verdict"] == "PENDING"
    assert (tmp_path / seal_v0.VERIFY_RECEIPT_NAME).is_file()


def test_independent_verify_rejects_contamination(tmp_path: Path) -> None:
    base = ablation.default_score_template(0.70)
    good = ablation.default_score_template(0.70)
    for d in ablation.MATH_DOMAINS:
        good["math"][d] = 0.85
    # Need full latent arm set for matrix latent_v0
    from lab.operators import frankenstein_latent_v0 as lv

    arm_scores = {
        lv.ARM_A: {
            "math": base["math"],
            "secondary": base["secondary"],
            "bench_scope": "FIXTURE",
        },
        lv.ARM_G: {
            "math": good["math"],
            "secondary": good["secondary"],
            "bench_scope": "FIXTURE",
        },
    }
    result = seal_v0.independent_verify(
        arm_scores=arm_scores,
        contamination_overlap_ids=["ex-leak-1"],
        out_dir=tmp_path,
        matrix="latent_v0",
    )
    assert result["verdict"] == "REJECT"
    assert result["challenger"]["contamination"]["contamination_detected"] is True
    assert result["challenger"]["verdict"] == "REJECT"


def test_independent_verify_rejects_teacher_memorization(tmp_path: Path) -> None:
    result = seal_v0.independent_verify(
        memorization_flags={"memorization_detected": True},
        out_dir=tmp_path,
        write=False,
    )
    # Without arm scores challenger may still REJECT on memorization alone
    assert result["challenger"]["teacher_memorization"]["memorization_detected"] is True
    assert result["challenger"]["verdict"] == "REJECT"
    assert result["verdict"] == "REJECT"


def test_ablation_audit_contamination_pass_with_membership() -> None:
    contam = ablation.audit_contamination(
        membership={"disjoint": True, "counts": {"train": 10, "hidden_test": 5}},
        barrier_pass=True,
    )
    verify(contam, label="contam")
    assert contam["status"] == "PASS"
    assert contam["clean"] is True


def test_ablation_audit_memorization_threshold() -> None:
    ok = ablation.audit_teacher_memorization(
        hidden_exact_match_rate=0.01,
        teacher_answer_replay_rate=0.02,
    )
    assert ok["status"] == "PASS"
    bad = ablation.audit_teacher_memorization(hidden_exact_match_rate=0.50)
    assert bad["status"] == "FAIL"
    assert bad["memorization_detected"] is True


# ---------------------------------------------------------------------------
# Promotion gate V0 extensions
# ---------------------------------------------------------------------------


def test_promotion_gate_pending_honest_by_default() -> None:
    doc = promo.evaluate_promotion()
    verify(doc, label="promo")
    assert doc["verdict"] == "PENDING"
    names = {c["name"] for c in doc["checks"]}
    assert "trained_modules" in names
    assert "contamination_barrier" in names
    assert "retention_gate" in names
    assert REQUIRES_TRAINED_MODULES in doc["infra_gates"]


def test_promotion_gate_rejects_untrained_modules() -> None:
    doc = promo.evaluate_promotion(trained_modules={"trained": False})
    assert any(
        c["name"] == "trained_modules" and c["status"] == "FAIL" for c in doc["checks"]
    )
    assert doc["verdict"] == "REJECT"


def test_promotion_gate_rejects_contamination() -> None:
    doc = promo.evaluate_promotion(
        contamination={"pass": False, "contamination_detected": True}
    )
    assert doc["verdict"] == "REJECT"


def test_gates_requires_trained_modules_constant() -> None:
    assert gates.REQUIRES_TRAINED_MODULES == "REQUIRES_TRAINED_MODULES"
    inv = gates.inventory_built_vs_gated()
    assert any("V0 HCLI-loadable" in x for x in inv["built_now"])
    assert REQUIRES_TRAINED_MODULES in inv["runtime_gated"]


# ---------------------------------------------------------------------------
# Cloud package + restore + confirm hook
# ---------------------------------------------------------------------------


def test_cloud_package_restore_and_confirm(tmp_path: Path) -> None:
    # Assemble trained artifact into seal root
    pack_dir = _write_pack(tmp_path / "mods", _module_pack(trained=True))
    seal_root = tmp_path / "proto-frankenstein"
    assembled = seal_v0.assemble_v0_artifact(modules_path=pack_dir, out_dir=seal_root)
    assert assembled["loadable"] is True

    # Independent verify receipt (pending is fine for packaging)
    seal_v0.independent_verify(out_dir=seal_root, write=True)

    pkg = seal_v0.build_cloud_package(artifact_dir=seal_root, out_dir=seal_root)
    assert pkg["status"] == "PACKAGE_READY_AWAITING_MANUAL_UPLOAD"
    assert pkg["upload_performed"] is False
    assert pkg["cloud_sealed"]["confirmed"] is False
    assert pkg["cloud_sealed"]["reclaim_may_evict_superseded"] is False
    man_path = Path(pkg["paths"]["manifest"])
    verify(json.loads(man_path.read_text()), label="cloud manifest")
    assert (Path(pkg["package_dir"]) / seal_v0.RESTORE_SCRIPT_NAME).is_file()
    assert (Path(pkg["package_dir"]) / seal_v0.UPLOAD_INSTRUCTIONS_NAME).is_file()

    # Restore to a fresh dir
    restore_out = tmp_path / "restored"
    restored = seal_v0.restore_from_package(
        package_dir=pkg["package_dir"], out_dir=restore_out
    )
    assert restored["status"] == "RESTORED"
    assert (restore_out / seal_v0.ARTIFACT_NAME).is_file()
    assert (restore_out / "PROTO_FRANKENSTEIN_V0_RESTORE_RECEIPT.json").is_file()

    # Confirm cloud sealed (manual human step simulated)
    remote = "c" * 64
    conf = seal_v0.confirm_cloud_sealed(
        package_dir=pkg["package_dir"],
        remote_hash=remote,
        remote_uri="s3://example/proto-v0.tar",
    )
    assert conf["confirmed"] is True
    assert conf["reclaim_may_evict_superseded"] is True
    sealed_path = Path(conf["path"])
    sealed_doc = json.loads(sealed_path.read_text())
    verify(sealed_doc, label="PROTO_CLOUD_SEALED")
    assert sealed_doc["remote_hash"] == remote
    assert sealed_doc["reclaim_contract"]["require_confirmed_true"] is True

    status = seal_v0.read_cloud_sealed_for_reclaim([pkg["package_dir"]])
    assert status["found"] is True
    assert status["reclaim_may_evict_superseded"] is True


def test_confirm_cloud_rejects_empty_hash(tmp_path: Path) -> None:
    seal_root = tmp_path / "proto"
    seal_root.mkdir()
    pkg = seal_v0.build_cloud_package(artifact_dir=seal_root, out_dir=seal_root)
    with pytest.raises(seal_v0.V0SealError, match="remote_hash"):
        seal_v0.confirm_cloud_sealed(package_dir=pkg["package_dir"], remote_hash="  ")


def test_restore_detects_tampered_payload(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "mods", _module_pack(trained=True))
    seal_root = tmp_path / "proto"
    seal_v0.assemble_v0_artifact(modules_path=pack_dir, out_dir=seal_root)
    pkg = seal_v0.build_cloud_package(artifact_dir=seal_root, out_dir=seal_root)
    payload = Path(pkg["package_dir"]) / "payload" / seal_v0.ARTIFACT_NAME
    if payload.is_file():
        payload.write_text(payload.read_text() + "\n# tampered\n")
        with pytest.raises(seal_v0.V0SealError, match="hash mismatch"):
            seal_v0.restore_from_package(
                package_dir=pkg["package_dir"], out_dir=tmp_path / "bad"
            )


def test_cli_assemble_exit_code(tmp_path: Path) -> None:
    code = seal_v0.main(["assemble", "--out", str(tmp_path), "--modules", str(tmp_path / "missing")])
    assert code == 2
