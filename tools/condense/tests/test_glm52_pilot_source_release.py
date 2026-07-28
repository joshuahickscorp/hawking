#!/usr/bin/env python3.12
"""Fake-only tests for the sealed five-shard pilot source release controller.

Never touches the real pilot_source under Application Support. All worlds are
tiny temporary directories with synthetic five-byte shard bodies.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_pilot_source_release as rel  # noqa: E402

CONFIRM = rel.CONFIRM_PHRASE

SHARD_SPECS = [
    ("model-00108-of-00282.safetensors", b"shard-a-payload-00108"),
    ("model-00156-of-00282.safetensors", b"shard-b-payload-00156"),
    ("model-00157-of-00282.safetensors", b"shard-c-payload-00157"),
    ("model-00112-of-00282.safetensors", b"shard-d-payload-00112"),
    ("model-00256-of-00282.safetensors", b"shard-e-payload-00256"),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def _clean_scan(pilot_root: Path) -> dict:
    return {
        "lsof": {"available": True, "open_references": 0},
        "argv": {"available": True, "referencing_processes": 0},
        "matches": [],
        "self_pid": os.getpid(),
        "any_probe_ran": True,
        "both_probes_unavailable": False,
        "clean": True,
    }


def _consumer_scan(pilot_root: Path) -> dict:
    return {
        "lsof": {"available": True, "open_references": 1},
        "argv": {"available": True, "referencing_processes": 1},
        "matches": [
            {"probe": "lsof", "line": f"python 99999 {pilot_root}/model-00108-of-00282.safetensors"},
            {"probe": "argv", "line": f"99999 python -c open({pilot_root})"},
        ],
        "self_pid": os.getpid(),
        "any_probe_ran": True,
        "both_probes_unavailable": False,
        "clean": False,
    }


def _no_probe_scan(pilot_root: Path) -> dict:
    return {
        "lsof": {"available": False, "error": "lsof missing"},
        "argv": {"available": False, "error": "ps failed"},
        "matches": [],
        "self_pid": os.getpid(),
        "any_probe_ran": False,
        "both_probes_unavailable": True,
        "clean": False,
    }


def _build_world(tmp_path: Path, *, mutate=None) -> dict:
    """Construct a fully green fake world. mutate(world) may break it."""
    support = tmp_path / "GLM52Gravity"
    pilot = support / "pilot_source"
    repo = tmp_path / "repo"
    capsules = support / "source_fetch" / "teacher" / "capsules"
    compact = support / "compact"
    mop = tmp_path / "mop"
    for d in (pilot, capsules, compact, mop, repo / "odyssey" / "launch", repo / "tools" / "condense" / "tests"):
        d.mkdir(parents=True, exist_ok=True)

    # Retained evidence beside shards.
    _write(pilot / "REHYDRATE_LEDGER.jsonl", '{"event":"rehydrated"}\n')
    _write(pilot / "final_ascent_rehydrate.stdout.log", "ok\n")
    _write(pilot / "final_ascent_rehydrate.stderr.log", "")
    (pilot / "hf_home").mkdir()
    (pilot / ".cache").mkdir()

    shards_meta = []
    for name, payload in SHARD_SPECS:
        _write(pilot / name, payload)
        shards_meta.append(
            {
                "name": name,
                "role": f"fake {name}",
                "bytes": len(payload),
                "sha256": _sha(payload),
            }
        )

    # Pilot code/test stubs whose hashes the reseal binds.
    basis_py = repo / "tools" / "condense" / "glm52_basis_pilot.py"
    pack_py = repo / "tools" / "condense" / "glm52_activation_aware_pack.py"
    test_py = repo / "tools" / "condense" / "tests" / "test_glm52_basis_pilot.py"
    _write(basis_py, b"# fake basis pilot\n")
    _write(pack_py, b"# fake pack\n")
    _write(test_py, b"# fake test\n")
    code_hashes = {
        "glm52_basis_pilot_py_sha256": _sha(basis_py.read_bytes()),
        "glm52_activation_aware_pack_py_sha256": _sha(pack_py.read_bytes()),
        "test_glm52_basis_pilot_py_sha256": _sha(test_py.read_bytes()),
    }

    rev0_content_sha = _sha(b"revision-0-tensor-results-fake")
    rev0 = {
        "label": "revision_0",
        "sha256": rev0_content_sha,
        "bytes": 100,
        "schema": "hawking.glm52.basis_pilot.v1",
        "safety": {
            "gaussian_proxy_used_for_selection": False,
            "full_parent_traversal_started": False,
            "ODYSSEY_LAUNCH_AUTHORIZED": False,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "HIDE_KERNEL_TURN": False,
        },
    }
    rev0_path = repo / "GLM52_BASIS_PILOT_REVISION_0_EVIDENCE.json"
    _write(rev0_path, json.dumps(rev0, indent=2, sort_keys=True))

    measurement = {
        "schema": "hawking.glm52.basis_pilot.v1",
        "revision": 1,
        "revision_0_evidence": {"label": "revision_0", "sha256": rev0_content_sha, "bytes": 100},
        "inputs": {
            "pilot_source": str(pilot),
            "verified_shards": [
                {
                    "name": m["name"],
                    "path": str(pilot / m["name"]),
                    "bytes": m["bytes"],
                    "sha256": m["sha256"],
                    "verified": True,
                }
                for m in shards_meta
            ],
        },
        "safety": {
            "full_parent_traversal_started": False,
            "gaussian_proxy_used_for_selection": False,
            "ODYSSEY_LAUNCH_AUTHORIZED": False,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "HIDE_KERNEL_TURN": False,
            "full_traversal_authorized": False,
        },
    }
    measurement_path = repo / "GLM52_BASIS_PILOT_RECEIPT.json"
    _write(measurement_path, json.dumps(measurement, indent=2, sort_keys=True))
    measurement_hash = _sha(measurement_path.read_bytes())

    reseal = {
        "schema": "hawking.glm52.basis_pilot.controller_reseal.v1",
        "status": "SEALED_WITH_NON_MEASUREMENT_CONTROLLER_FIX",
        "measurement_receipt": {
            "path": "GLM52_BASIS_PILOT_RECEIPT.json",
            "sha256": measurement_hash,
        },
        "revision_0_evidence": {"sha256": rev0_content_sha},
        "reviewed_current_code": code_hashes,
        "post_measurement_fix": {"measurement_math_changed": False},
        "scientific_disposition": {"full_traversal_authorized": False},
        "fences": {
            "ODYSSEY_LAUNCH_AUTHORIZED": False,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "HIDE_KERNEL_TURN": False,
        },
    }
    reseal_path = repo / "GLM52_BASIS_PILOT_CONTROLLER_RESEAL.json"
    _write(reseal_path, json.dumps(reseal, indent=2, sort_keys=True))

    rehydration = {
        "schema": "hawking.final_ascent.source_rehydration_receipt.v1",
        "source": {
            "repo": "zai-org/GLM-5.2",
            "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
            "destination": str(pilot),
            "resident_now_shards": 5,
        },
        "shards": shards_meta,
        "safety": {
            "ODYSSEY_LAUNCH_AUTHORIZED": False,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "HIDE_KERNEL_TURN": False,
        },
    }
    rehydration_path = repo / "HAWKING_FINAL_ASCENT_SOURCE_REHYDRATION_RECEIPT.json"
    _write(rehydration_path, json.dumps(rehydration, indent=2, sort_keys=True))

    status = {
        "fences": {
            "ODYSSEY_LAUNCH_AUTHORIZED": False,
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "HIDE_KERNEL_TURN": False,
        }
    }
    status_path = repo / "HAWKING_FINAL_ASCENT_STATUS.json"
    _write(status_path, json.dumps(status, indent=2, sort_keys=True))
    _write(repo / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED", "false\n")

    release_receipt = repo / "HAWKING_FINAL_ASCENT_PILOT_SOURCE_RELEASE_RECEIPT.json"

    world = {
        "support": support,
        "pilot": pilot,
        "repo": repo,
        "capsules": capsules,
        "compact": compact,
        "mop": mop,
        "shards_meta": shards_meta,
        "paths": rel.Paths(
            support_root=support,
            pilot_root=pilot,
            repo_root=repo,
            measurement_receipt=measurement_path,
            controller_reseal=reseal_path,
            revision_0_evidence=rev0_path,
            rehydration_receipt=rehydration_path,
            final_ascent_status=status_path,
            odyssey_authorized=repo / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED",
            release_receipt=release_receipt,
            basis_pilot_py=basis_py,
            activation_aware_pack_py=pack_py,
            test_basis_pilot_py=test_py,
            capsules=capsules,
            compact=compact,
            mop=mop,
        ),
        "basis_py": basis_py,
        "pack_py": pack_py,
        "test_py": test_py,
        "measurement_path": measurement_path,
        "reseal_path": reseal_path,
        "rev0_path": rev0_path,
        "rehydration_path": rehydration_path,
        "status_path": status_path,
        "release_receipt": release_receipt,
        "code_hashes": code_hashes,
        "rev0_content_sha": rev0_content_sha,
    }
    if mutate is not None:
        mutate(world)
    return world


def _activate(world: dict, process_scan=None, free_bytes=None) -> None:
    rel.configure_for_tests(
        world["paths"],
        rel.Runtime(
            process_scan=process_scan or _clean_scan,
            free_bytes=free_bytes or (lambda _p: 10**12),
        ),
    )


@pytest.fixture(autouse=True)
def _reset_defaults():
    yield
    rel.reset_to_defaults()


# --------------------------------------------------------------------------- green path


def test_green_gate_and_confirmed_exact_deletion(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    report = rel.gate()
    assert report["all_green"], json.dumps(report["gates"], indent=2)
    assert report["green"] == report["total"] == 10

    receipt = rel.release(CONFIRM)
    assert receipt["success"] is True
    assert receipt["deletion"]["all_deleted"] is True
    assert receipt["deletion"]["deleted_bytes"] == sum(m["bytes"] for m in world["shards_meta"])
    assert rel.verify_receipt_seal(receipt)

    for m in world["shards_meta"]:
        assert not (world["pilot"] / m["name"]).exists()
    # Retained evidence and directory.
    assert world["pilot"].is_dir()
    assert (world["pilot"] / "REHYDRATE_LEDGER.jsonl").is_file()
    assert (world["pilot"] / "final_ascent_rehydrate.stdout.log").is_file()
    assert (world["pilot"] / "hf_home").is_dir()
    assert world["release_receipt"].is_file()
    # Protected trees untouched.
    assert world["capsules"].is_dir()
    assert world["compact"].is_dir()
    assert world["mop"].is_dir()


def test_wrong_confirmation(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    with pytest.raises(SystemExit, match="confirmation phrase"):
        rel.release("YES_DELETE")
    for m in world["shards_meta"]:
        assert (world["pilot"] / m["name"]).is_file()


def test_missing_confirm_flag_cli(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    rc = rel.main(["release"])
    assert rc == 1
    for m in world["shards_meta"]:
        assert (world["pilot"] / m["name"]).is_file()


# --------------------------------------------------------------------------- shard identity


def test_missing_shard(tmp_path: Path):
    def mutate(w):
        (w["pilot"] / SHARD_SPECS[0][0]).unlink()

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g03_exact_five_regular_bodies"]["status"] == "red"
    assert not report["all_green"]


def test_extra_shard(tmp_path: Path):
    def mutate(w):
        _write(w["pilot"] / "model-00001-of-00282.safetensors", b"extra")

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g03_exact_five_regular_bodies"]["status"] == "red"
    assert "extra" in report["gates"]["g03_exact_five_regular_bodies"]["reason"]


def test_size_mismatch_shard(tmp_path: Path):
    def mutate(w):
        path = w["pilot"] / SHARD_SPECS[1][0]
        path.write_bytes(path.read_bytes() + b"X")

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g03_exact_five_regular_bodies"]["status"] == "red"


def test_hash_mismatch_shard(tmp_path: Path):
    def mutate(w):
        # Same size, different bytes — size gate may still pass depending on order;
        # g04 full-hash must fail. Keep size identical.
        name, original = SHARD_SPECS[2]
        path = w["pilot"] / name
        flipped = bytes((b ^ 0xFF) for b in original)
        assert len(flipped) == len(original)
        path.write_bytes(flipped)

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g03_exact_five_regular_bodies"]["status"] == "green"
    assert report["gates"]["g04_full_hash_match"]["status"] == "red"


# --------------------------------------------------------------------------- symlinks


def test_symlinked_root_refusal(tmp_path: Path):
    world = _build_world(tmp_path)
    real = world["support"] / "pilot_source_real"
    world["pilot"].rename(real)
    world["pilot"].symlink_to(real)
    # Rebind paths object pilot_root still points at the symlink path.
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g01_pilot_root_exact"]["status"] == "red"
    assert "symlink" in report["gates"]["g01_pilot_root_exact"]["reason"]


def test_symlinked_shard_refusal(tmp_path: Path):
    def mutate(w):
        name = SHARD_SPECS[0][0]
        real = w["pilot"] / (name + ".real")
        path = w["pilot"] / name
        path.rename(real)
        path.symlink_to(real)

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g03_exact_five_regular_bodies"]["status"] == "red"


# --------------------------------------------------------------------------- path isolation


def test_path_escape_protected_target_refusal(tmp_path: Path):
    """A sealed name that is a directory / outside pilot must never be deletable.

    We simulate escape by pointing a sealed shard path at the capsules directory
    via a symlink — which g03 already refuses. Also assert g09 isolation language
    by forcing pilot_root to equal support_root (misconfiguration).
    """
    world = _build_world(tmp_path)
    # Misconfigure pilot_root to the support root itself.
    world["paths"] = rel.Paths(
        **{
            **world["paths"].__dict__,
            "pilot_root": world["support"],
        }
    )
    _activate(world)
    report = rel.gate()
    assert not report["all_green"]
    # Either root name check or isolation must catch this.
    assert (
        report["gates"]["g01_pilot_root_exact"]["status"] == "red"
        or report["gates"]["g09_path_isolation"]["status"] == "red"
    )


def test_deletion_never_touches_protected_trees(tmp_path: Path):
    world = _build_world(tmp_path)
    sentinel_cap = world["capsules"] / "KEEP.npz"
    sentinel_compact = world["compact"] / "KEEP.gravity"
    sentinel_mop = world["mop"] / "KEEP.bin"
    _write(sentinel_cap, b"capsule")
    _write(sentinel_compact, b"compact")
    _write(sentinel_mop, b"mop")
    _activate(world)
    rel.release(CONFIRM)
    assert sentinel_cap.read_bytes() == b"capsule"
    assert sentinel_compact.read_bytes() == b"compact"
    assert sentinel_mop.read_bytes() == b"mop"


# --------------------------------------------------------------------------- bindings


def test_stale_measurement_receipt_binding(tmp_path: Path):
    def mutate(w):
        data = json.loads(w["measurement_path"].read_text())
        data["tamper"] = True
        w["measurement_path"].write_text(json.dumps(data, indent=2, sort_keys=True))

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g05_controller_reseal"]["status"] == "red"


def test_stale_reseal_binding(tmp_path: Path):
    def mutate(w):
        data = json.loads(w["reseal_path"].read_text())
        data["measurement_receipt"]["sha256"] = "0" * 64
        w["reseal_path"].write_text(json.dumps(data, indent=2, sort_keys=True))

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g05_controller_reseal"]["status"] == "red"


def test_revision_0_hash_mismatch(tmp_path: Path):
    def mutate(w):
        data = json.loads(w["rev0_path"].read_text())
        data["sha256"] = "f" * 64
        w["rev0_path"].write_text(json.dumps(data, indent=2, sort_keys=True))

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g05_controller_reseal"]["status"] == "red"


def test_current_code_hash_mismatch(tmp_path: Path):
    def mutate(w):
        w["basis_py"].write_bytes(w["basis_py"].read_bytes() + b"\n# changed\n")

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g05_controller_reseal"]["status"] == "red"
    assert "code" in report["gates"]["g05_controller_reseal"]["reason"]


def test_measurement_math_changed_blocks(tmp_path: Path):
    def mutate(w):
        data = json.loads(w["reseal_path"].read_text())
        data["post_measurement_fix"]["measurement_math_changed"] = True
        w["reseal_path"].write_text(json.dumps(data, indent=2, sort_keys=True))

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    assert rel.gate()["gates"]["g05_controller_reseal"]["status"] == "red"


def test_fence_not_false_blocks(tmp_path: Path):
    def mutate(w):
        data = json.loads(w["status_path"].read_text())
        data["fences"]["ODYSSEY_LAUNCH_AUTHORIZED"] = True
        w["status_path"].write_text(json.dumps(data, indent=2, sort_keys=True))

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    assert rel.gate()["gates"]["g07_final_ascent_fences"]["status"] == "red"


def test_gaussian_selection_blocks(tmp_path: Path):
    def mutate(w):
        data = json.loads(w["measurement_path"].read_text())
        data["safety"]["gaussian_proxy_used_for_selection"] = True
        # Re-bind reseal to the new measurement hash so g05 stays green and g06 fires.
        w["measurement_path"].write_text(json.dumps(data, indent=2, sort_keys=True))
        reseal = json.loads(w["reseal_path"].read_text())
        reseal["measurement_receipt"]["sha256"] = _sha(w["measurement_path"].read_bytes())
        w["reseal_path"].write_text(json.dumps(reseal, indent=2, sort_keys=True))

    world = _build_world(tmp_path, mutate=mutate)
    _activate(world)
    report = rel.gate()
    assert report["gates"]["g06_measurement_receipt_safety"]["status"] == "red"


# --------------------------------------------------------------------------- process probes


def test_simulated_live_consumer(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world, process_scan=_consumer_scan)
    report = rel.gate()
    assert report["gates"]["g08_process_quiescence"]["status"] == "red"
    with pytest.raises(SystemExit, match="gates not green"):
        rel.release(CONFIRM)
    for m in world["shards_meta"]:
        assert (world["pilot"] / m["name"]).is_file()


def test_no_process_probe_fail_closed(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world, process_scan=_no_probe_scan)
    report = rel.gate()
    assert report["gates"]["g08_process_quiescence"]["status"] == "red"
    assert "fail closed" in report["gates"]["g08_process_quiescence"]["reason"]


# --------------------------------------------------------------------------- release behavior


def test_release_reruns_the_gate(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    assert rel.gate_run_count() == 0
    rel.gate()
    assert rel.gate_run_count() == 1
    rel.release(CONFIRM)
    # release must invoke gate once more in-process
    assert rel.gate_run_count() == 2


def test_retained_ledger_log_cache_survive(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    rel.release(CONFIRM)
    assert (world["pilot"] / "REHYDRATE_LEDGER.jsonl").read_text().startswith("{")
    assert (world["pilot"] / "final_ascent_rehydrate.stdout.log").exists()
    assert (world["pilot"] / "final_ascent_rehydrate.stderr.log").exists()
    assert (world["pilot"] / "hf_home").is_dir()
    assert (world["pilot"] / ".cache").is_dir()
    assert world["pilot"].is_dir()
    # Only the five model bodies are gone — no recursive wipe.
    leftovers = [p.name for p in world["pilot"].iterdir()]
    assert not any(n.startswith("model-") and n.endswith(".safetensors") for n in leftovers)
    assert "REHYDRATE_LEDGER.jsonl" in leftovers


def test_sealed_receipt_verifies_and_blocks_replay(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    receipt = rel.release(CONFIRM)
    assert rel.verify_receipt_seal(receipt)
    on_disk = json.loads(world["release_receipt"].read_text())
    assert rel.verify_receipt_seal(on_disk)
    assert on_disk["seal_sha256"] == receipt["seal_sha256"]

    # Tamper breaks seal.
    on_disk["deletion"]["deleted_bytes"] = 0
    assert not rel.verify_receipt_seal(on_disk)

    # Second successful release is refused (bodies already gone + sealed receipt).
    with pytest.raises(SystemExit) as exc:
        rel.release(CONFIRM)
    msg = str(exc.value)
    assert "gates not green" in msg or "already" in msg.lower()


def test_status_is_read_only(tmp_path: Path):
    world = _build_world(tmp_path)
    _activate(world)
    before = {m["name"]: (world["pilot"] / m["name"]).read_bytes() for m in world["shards_meta"]}
    summary = rel.status()
    assert summary["read_only"] is True
    assert summary["resident_count"] == 5
    assert not world["release_receipt"].exists()
    for name, payload in before.items():
        assert (world["pilot"] / name).read_bytes() == payload


def test_cli_gate_status_release(tmp_path: Path, capsys):
    world = _build_world(tmp_path)
    _activate(world)
    assert rel.main(["gate"]) == 0
    assert rel.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "resident_count" in out or '"resident_count"' in out
    assert rel.main(["release", "--confirm", CONFIRM]) == 0
    assert not any((world["pilot"] / m["name"]).exists() for m in world["shards_meta"])


def test_release_records_free_bytes(tmp_path: Path):
    world = _build_world(tmp_path)
    calls = {"n": 0}

    def free(_p):
        calls["n"] += 1
        return 1_000_000 + calls["n"] * 100

    _activate(world, free_bytes=free)
    receipt = rel.release(CONFIRM)
    assert receipt["disk"]["free_bytes_before"] is not None
    assert receipt["disk"]["free_bytes_after"] is not None
    assert "free_delta_bytes" in receipt["disk"]


def test_partial_hash_drift_at_delete_reports_state(tmp_path: Path, monkeypatch):
    """If a body mutates between gate and unlink, release fails nonzero without full success."""
    world = _build_world(tmp_path)
    _activate(world)
    original_unlink = os.unlink
    flipped = False

    def flaky_unlink(path):
        nonlocal flipped
        path = Path(path)
        if not flipped and path.name == SHARD_SPECS[0][0]:
            # Mutate a *different* remaining shard before continuing — gate already passed.
            other = world["pilot"] / SHARD_SPECS[1][0]
            other.write_bytes(b"!" * other.stat().st_size)
            flipped = True
        return original_unlink(path)

    monkeypatch.setattr(os, "unlink", flaky_unlink)
    with pytest.raises(SystemExit) as exc:
        rel.release(CONFIRM)
    payload = str(exc.value)
    assert "partial" in payload.lower() or "remaining" in payload.lower() or "drift" in payload.lower()
    # At least one body may remain; controller must not claim full success.
    assert not (
        world["release_receipt"].exists()
        and json.loads(world["release_receipt"].read_text()).get("success") is True
        and json.loads(world["release_receipt"].read_text()).get("deletion", {}).get("all_deleted")
        is True
        and all(not (world["pilot"] / m["name"]).exists() for m in world["shards_meta"])
    )


def test_confirm_phrase_constant():
    assert CONFIRM == "RELEASE_EXACT_SEALED_FIVE_SHARD_PILOT"
