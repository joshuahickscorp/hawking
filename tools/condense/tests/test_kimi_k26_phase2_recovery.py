from __future__ import annotations

import ast
import copy
import contextlib
import hashlib
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import kimi_k26_phase2_recovery as phase2  # noqa: E402
import kimi_k26_release_cycle as phase1  # noqa: E402


def _private(path: Path) -> None:
    path.mkdir(mode=0o700, parents=False, exist_ok=False)


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            assert written > 0
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sealed_bytes(value: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    sealed = phase1.seal_document(value)
    return sealed, phase1.canonical_json(sealed) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324


@dataclass
class World:
    layout: phase1.SessionLayout
    mop: Path
    shared: Path
    historical: dict[str, bytes]
    frozen_raw: dict[str, bytes]
    bracket_raw: dict[str, bytes]
    doctor_raw: dict[str, bytes]
    events: list[str]
    calls: list[dict[str, Any]]
    hooks: phase2.Phase2Hooks
    runner: "FakeRunner"


class FakeRunner:
    def __init__(
        self,
        *,
        bracket_raw: dict[str, bytes],
        doctor_raw: dict[str, bytes],
        events: list[str],
        calls: list[dict[str, Any]],
    ) -> None:
        self.bracket_raw = bracket_raw
        self.doctor_raw = doctor_raw
        self.events = events
        self.calls = calls
        self.corrupt_bracket: str | None = None
        self.corrupt_doctor: str | None = None
        self.returncode_on_call: int | None = None

    def __call__(
        self,
        argv: tuple[str, ...] | list[str],
        *,
        env: dict[str, str],
        cwd: Path,
        pass_fds: tuple[int, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        call_number = len(self.calls) + 1
        self.calls.append(
            {
                "argv": list(argv),
                "env": dict(env),
                "cwd": cwd,
                "pass_fds": tuple(pass_fds),
            }
        )
        self.events.append(f"process-{call_number}")
        if self.returncode_on_call == call_number:
            return subprocess.CompletedProcess(list(argv), 19, b"", b"fake failure")
        output = Path(argv[list(argv).index("--output-dir") + 1])
        is_bracket = "--source" in argv
        source = self.bracket_raw if is_bracket else self.doctor_raw
        corrupt = self.corrupt_bracket if is_bracket else self.corrupt_doctor
        for name, raw in source.items():
            value = raw + b"tampered" if name == corrupt else raw
            target = output / name
            if target.exists():
                assert target.read_bytes() == value
            else:
                _write_private(target, value)
        return subprocess.CompletedProcess(list(argv), 0, b'{"status":"PASS"}\n', b"")


def _source_document() -> dict[str, Any]:
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.source_verification.v1",
            "status": "PASS_EXACT_IMMUTABLE_SOURCE",
            "repo": phase1.KIMI_REPO,
            "revision": phase1.KIMI_REVISION,
            "manifest_seal_sha256": phase1.KIMI_MANIFEST_SEAL_SHA256,
            "file_count": phase1.KIMI_FILE_COUNT,
            "weight_shards": phase1.KIMI_WEIGHT_SHARDS,
            "logical_bytes": phase1.KIMI_TOTAL_BYTES,
            "weight_bytes": phase1.KIMI_WEIGHT_BYTES,
            "shared_xet_used": False,
            "mop_touched": False,
            "network_accessed_by_verifier": False,
            "files": [],
        }
    )


def _archive_document() -> dict[str, Any]:
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.recovery_archive.v1",
            "status": "PASS_SANITIZED_EXACT_ARCHIVE",
            "archive_sha256": phase1.ACCEPTED_ARCHIVE_SHA256,
            "archive_bytes": phase1.ACCEPTED_ARCHIVE_BYTES,
            "entry_count": phase1.ACCEPTED_ARCHIVE_ENTRIES,
            "credential_entry_present": False,
            "old_898063_byte_archive_rejected": True,
        }
    )


def _runtime_document() -> dict[str, Any]:
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.generation_runtime.v1",
            "status": "PASS_EXACT_PINNED_RUNTIME",
            "network_accessed": False,
            "fake_offline_test": True,
        }
    )


def _candidate_result(
    candidate: str,
    spec: phase2.BinarySpec,
) -> bytes:
    p1 = candidate == "P1"
    _, raw = _sealed_bytes(
        {
            "schema": "hawking.kimi_k26.f1_candidate_result.v1",
            "status": "PASS",
            "candidate": candidate,
            "source": {"repo": phase1.KIMI_REPO, "revision": phase1.KIMI_REVISION},
            "layer": 1,
            "sentinel_expert": 0,
            "candidate_verdict": "DEGRADED_F1" if p1 else "COLLAPSE_F1",
            "physical_budget": {
                "target_complete_bpw": "49/50" if p1 else "1/2",
                "complete_ceiling_bytes": 5_394_923 if p1 else 2_752_512,
                "base_ceiling_bytes": 4_046_192 if p1 else 1_431_306,
                "doctor_ceiling_bytes": 1_240_832 if p1 else 1_238_630,
                "overhead_ceiling_bytes": 107_899 if p1 else 82_576,
                "logical_weights_represented": 44_040_192,
                "all_payload_bytes_counted": True,
            },
            "payload": {
                "bytes": spec.logical_bytes,
                "sha256": spec.sha256,
                "base_component_bytes": 4_022_298 if p1 else 1_404_937,
                "doctor_component_bytes": 1_220_627,
                "header_overhead_bytes": 5_831 if p1 else 4_962,
            },
            "metrics": {"held_out_score": 0.75 if p1 else 0.25},
        }
    )
    return raw


def _doctor_result(spec: phase2.BinarySpec) -> bytes:
    candidate = "P1" if spec.filename.startswith("P1_") else "P5"
    architecture = spec.filename.removesuffix(".k26f1").removeprefix(candidate + "_")
    p1 = candidate == "P1"
    dual = architecture == "DUAL_PATH_RECOVERY_R16X2"
    _, raw = _sealed_bytes(
        {
            "schema": "hawking.kimi_k26.f1_hidden_doctor_result.v1",
            "status": "PASS",
            "candidate": candidate,
            "architecture": architecture,
            "source": {"repo": phase1.KIMI_REPO, "revision": phase1.KIMI_REVISION},
            "layer": 1,
            "sentinel_expert": 0,
            "candidate_verdict": "SURVIVES_F1" if p1 and dual else "DEGRADED_F1",
            "physical_budget": {
                "target_complete_bpw": "49/50" if p1 else "1/2",
                "complete_ceiling_bytes": 5_394_923 if p1 else 2_752_512,
                "base_ceiling_bytes": 4_046_192 if p1 else 1_431_306,
                "doctor_ceiling_bytes": 1_240_832 if p1 else 1_238_630,
                "overhead_ceiling_bytes": 107_899 if p1 else 82_576,
                "logical_weights_represented": 44_040_192,
                "all_payload_bytes_counted": True,
            },
            "payload": {
                "bytes": spec.logical_bytes,
                "sha256": spec.sha256,
                "base_component_bytes": 4_022_298 if p1 else 1_404_937,
                "doctor_component_bytes": 974_848 if dual else 931_478,
                "header_overhead_bytes": 4_669,
            },
            "metrics": {"held_out_score": 0.9 if p1 and dual else 0.4},
        }
    )
    return raw


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    parent = tmp_path / "sessions"
    mop = tmp_path / "mop"
    shared = tmp_path / "shared-xet"
    for path in (parent, mop, shared):
        _private(path)
    layout = phase1.layout_for(parent / "case", parent=parent)
    _private(layout.session)
    for path in (layout.hub, layout.xet, layout.build, layout.recovery, layout.evidence):
        _private(path)
    _private(layout.tmp)
    _private(layout.hf_home)

    corpus, corpus_raw = _sealed_bytes(
        {
            "schema": "hawking.test.corpus.v1",
            "status": "PASS",
            "source": {
                "repo": phase1.KIMI_REPO,
                "revision": phase1.KIMI_REVISION,
                "tokenizer_sha256": phase2.CORPUS_TOKENIZER_SHA256,
            },
        }
    )
    monkeypatch.setattr(phase2, "CORPUS_SEAL_SHA256", corpus["seal_sha256"])
    historical: dict[str, bytes] = {
        "KIMI_K26_CORPUS_INTEGRITY.json": corpus_raw,
        "tools/condense/kimi_k26_f1_bracket.py": b"# fake bracket\n",
        "tools/condense/kimi_k26_f1_doctor_auction.py": b"# fake doctor\n",
        "tools/condense/gravity_forge.py": b"# fake forge\n",
        "tools/condense/kimi_k26_adapter.py": b"# fake adapter\n",
        "tools/condense/kimi_k26_reference.py": b"# fake reference\n",
    }
    blobs = tuple(
        phase2.BlobSpec(name, _git_blob(raw), len(raw), _sha(raw))
        for name, raw in historical.items()
    )
    monkeypatch.setattr(phase2, "HISTORICAL_BLOBS", blobs)

    teacher = b"tiny-teacher-capture"
    base_p1 = b"tiny-bracket-p1"
    base_p5 = b"tiny-bracket-p5"
    doctor_p1_r31 = b"tiny-doctor-p1-r31"
    best = b"tiny-doctor-p1-r16x2"
    doctor_p5_r31 = b"tiny-doctor-p5-r31"
    doctor_p5_r16 = b"tiny-doctor-p5-r16x2"
    bracket_specs = (
        phase2.BinarySpec("teacher_capture.npz", len(teacher), _sha(teacher)),
        phase2.BinarySpec("P1_sentinel_expert.k26f1", len(base_p1), _sha(base_p1)),
        phase2.BinarySpec("P5_sentinel_expert.k26f1", len(base_p5), _sha(base_p5)),
    )
    doctor_specs = (
        phase2.BinarySpec("P1_BASE_OUTPUT_RECOVERY_R31.k26f1", len(doctor_p1_r31), _sha(doctor_p1_r31)),
        phase2.BinarySpec(phase1.BEST_PAYLOAD_BASENAME, len(best), _sha(best)),
        phase2.BinarySpec("P5_BASE_OUTPUT_RECOVERY_R31.k26f1", len(doctor_p5_r31), _sha(doctor_p5_r31)),
        phase2.BinarySpec("P5_DUAL_PATH_RECOVERY_R16X2.k26f1", len(doctor_p5_r16), _sha(doctor_p5_r16)),
    )
    capsule_specs = (
        doctor_specs[1],
        phase2.BinarySpec(phase1.TEACHER_CAPTURE_BASENAME, len(teacher), _sha(teacher)),
    )
    monkeypatch.setattr(phase2, "BRACKET_BINARIES", bracket_specs)
    monkeypatch.setattr(phase2, "DOCTOR_BINARIES", doctor_specs)
    monkeypatch.setattr(phase2, "CAPSULE_BINARIES", capsule_specs)
    monkeypatch.setattr(phase1, "TEACHER_CAPTURE_BYTES", len(teacher))
    monkeypatch.setattr(phase1, "TEACHER_CAPTURE_SHA256", _sha(teacher))
    monkeypatch.setattr(phase1, "BEST_PAYLOAD_BYTES", len(best))
    monkeypatch.setattr(phase1, "BEST_PAYLOAD_SHA256", _sha(best))

    frozen_raw: dict[str, bytes] = {}
    frozen_specs: list[phase2.FrozenRecordSpec] = []
    for relative in (
        phase1.BEST_RESULT_RELATIVE,
        phase1.TEACHER_CAPTURE_RECORD_RELATIVE,
        phase1.GRAVITY_FINAL_RELATIVE,
    ):
        sealed, raw = _sealed_bytes(
            {"schema": "hawking.test.frozen.v1", "status": "PASS", "relative": relative}
        )
        frozen_raw[relative] = raw
        frozen_specs.append(
            phase2.FrozenRecordSpec(relative, len(raw), _sha(raw), sealed["seal_sha256"])
        )
    monkeypatch.setattr(phase2, "FROZEN_RECORDS", tuple(frozen_specs))

    capture, capture_json = _sealed_bytes(
        {
            "schema": "hawking.kimi_k26.f1_teacher_capture.v1",
            "status": "PASS",
            "revision": phase1.KIMI_REVISION,
            "layer": 1,
            "token_overlap": 0,
            "sentinel_expert": 0,
            "captured_at": "2026-01-01T00:00:00Z",
            "claim_boundary": "ONE_REAL_LAYER_ONE_SENTINEL_EXPERT; teacher captured once",
            "fit_token_ids": [1],
            "score_token_ids": [2],
            "token_ids": [1, 2],
            "sentinel_fit_route_slots": 1,
            "sentinel_score_route_slots": 1,
            "capture_bytes": len(teacher),
            "capture_sha256": _sha(teacher),
        }
    )
    assert capture["status"] == "PASS"
    monkeypatch.setattr(
        phase2, "PINNED_SEED_RELATIVE", phase2.PurePosixPath("run-failed/thread1.frozen")
    )
    monkeypatch.setattr(phase2, "PINNED_SEED_RECEIPT", capture)
    monkeypatch.setattr(phase2, "PINNED_SEED_RECEIPT_BYTES", len(capture_json))
    monkeypatch.setattr(phase2, "PINNED_SEED_RECEIPT_SHA256", _sha(capture_json))
    generic = _sealed_bytes({"schema": "hawking.test.output.v1", "status": "PASS"})[1]
    bracket_raw = {
        "teacher_capture.npz": teacher,
        "teacher_capture.json": capture_json,
        "P1_sentinel_expert.k26f1": base_p1,
        "P1_F1_RESULT.json": _candidate_result("P1", bracket_specs[1]),
        "P5_sentinel_expert.k26f1": base_p5,
        "P5_F1_RESULT.json": _candidate_result("P5", bracket_specs[2]),
        "KIMI_K26_F1_REPRESENTATION_BRACKET.json": generic,
        "KIMI_K26_SCIENTIFIC_STATUS.json": generic,
        "KIMI_K26_F1_PROGRESS.json": generic,
    }
    doctor_raw = {
        spec.filename: raw
        for spec, raw in zip(
            doctor_specs,
            (doctor_p1_r31, best, doctor_p5_r31, doctor_p5_r16),
            strict=True,
        )
    }
    for spec in doctor_specs:
        result_name = spec.filename.removesuffix(".k26f1") + "_RESULT.json"
        doctor_raw[result_name] = _doctor_result(spec)
    doctor_raw["KIMI_K26_F1_DOCTOR_AUCTION.json"] = generic

    frozen_semantic_raw = {
        name: raw
        for name, raw in {**bracket_raw, **doctor_raw}.items()
        if name in {
            "P1_F1_RESULT.json",
            "P5_F1_RESULT.json",
            "P1_BASE_OUTPUT_RECOVERY_R31_RESULT.json",
            "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json",
            "P5_BASE_OUTPUT_RECOVERY_R31_RESULT.json",
            "P5_DUAL_PATH_RECOVERY_R16X2_RESULT.json",
        }
    }
    frozen_semantics = {
        name: phase1.strict_json_bytes(raw, label=f"fake frozen semantic {name}")
        for name, raw in frozen_semantic_raw.items()
    }
    monkeypatch.setattr(
        phase2,
        "FROZEN_SEMANTIC_RECORDS",
        tuple(
            phase2.FrozenSemanticSpec(
                f"fake/{name}",
                name,
                len(raw),
                _sha(raw),
                frozen_semantics[name]["seal_sha256"],
            )
            for name, raw in sorted(frozen_semantic_raw.items())
        ),
    )

    events: list[str] = []
    calls: list[dict[str, Any]] = []
    runner = FakeRunner(
        bracket_raw=bracket_raw,
        doctor_raw=doctor_raw,
        events=events,
        calls=calls,
    )

    def extract_records(
        selected_layout: phase1.SessionLayout, names: set[str]
    ) -> dict[str, Any]:
        events.append("extract")
        assert names <= set(frozen_raw)
        root_fd = phase1._open_absolute_directory(selected_layout.recovery)  # noqa: SLF001
        try:
            for name in sorted(names):
                phase1._write_new_private_file(  # noqa: SLF001
                    root_fd, phase2.PurePosixPath(name), frozen_raw[name]
                )
        finally:
            os.close(root_fd)
        return phase1.seal_document(
            {
                "schema": "hawking.test.extraction.v1",
                "status": "PASS_ALLOWLISTED_TEXT_ONLY",
                "extracted": [{"relative_path": name} for name in sorted(names)],
            }
        )

    def verify_capsule(_layout: phase1.SessionLayout) -> dict[str, Any]:
        events.append("verify-capsule")
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.release_cycle.rollback_capsule.v1",
                "status": "PASS_EXACT_PAYLOAD_RESULT_CAPTURE",
            }
        )

    @contextlib.contextmanager
    def lease(_layout: phase1.SessionLayout) -> Iterator[int]:
        events.append("lease-enter")
        descriptor = os.open("/dev/null", os.O_RDONLY)
        try:
            yield descriptor
        finally:
            os.close(descriptor)
            events.append("lease-exit")

    @contextlib.contextmanager
    def source_guard(
        _layout: phase1.SessionLayout, _verification: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        events.append("source-guard-enter")
        try:
            yield phase1.seal_document({"schema": "hawking.test.guard.v1", "status": "PASS"})
        finally:
            events.append("source-guard-exit")

    hooks = phase2.Phase2Hooks(
        verify_source=lambda _layout: _source_document(),
        verify_archive=_archive_document,
        verify_runtime=_runtime_document,
        load_historical_sources=lambda: dict(historical),
        load_frozen_semantics=lambda: dict(frozen_semantics),
        extract_records=extract_records,
        verify_capsule=verify_capsule,
        exclusive_lease=lease,
        source_guard=source_guard,
        run_process=runner,
    )
    return World(
        layout,
        mop,
        shared,
        historical,
        frozen_raw,
        bracket_raw,
        doctor_raw,
        events,
        calls,
        hooks,
        runner,
    )


def _preflight(world: World) -> dict[str, Any]:
    return phase2.preflight(
        world.layout,
        hooks=world.hooks,
        mop_root=world.mop,
        shared_xet=world.shared,
    )


def _generate(world: World) -> dict[str, Any]:
    return phase2.generate(
        world.layout,
        hooks=world.hooks,
        mop_root=world.mop,
        shared_xet=world.shared,
    )


def test_preflight_is_read_only_and_binds_six_blobs(world: World) -> None:
    before = sorted(str(path.relative_to(world.layout.session)) for path in world.layout.session.rglob("*"))
    result = _preflight(world)
    after = sorted(str(path.relative_to(world.layout.session)) for path in world.layout.session.rglob("*"))
    assert before == after
    assert result["status"] == "PASS_READ_ONLY_READY_FOR_EXPLICIT_GENERATE"
    export = result["historical_export_verification"]
    assert export["status"] == "PASS_EXACT_SIX_BLOB_ALLOWLIST"
    assert {row["relative_path"] for row in export["blobs"]} == set(world.historical)
    assert result["generator_executed"] is False
    assert not world.calls
    assert not world.events


def test_incomplete_transfer_blocks_before_generation(world: World) -> None:
    _write_private(world.layout.hub / "dead-object.incomplete", b"partial")
    with pytest.raises(phase2.Phase2RecoveryError, match="incomplete transfer"):
        _generate(world)
    assert not world.calls
    assert not (world.layout.build / "phase2-recovery").exists()


def test_generate_uses_private_export_sandbox_and_exact_capsule(world: World) -> None:
    result = _generate(world)
    assert result["status"] == "PASS_EXACT_RECOVERED_CAPSULE"
    assert len(world.calls) == 2
    for call in world.calls:
        argv = call["argv"]
        assert argv[:3] == [str(phase2.SANDBOX_EXEC), "-p", phase2.SANDBOX_PROFILE]
        assert argv[3:7] == [str(phase2.PYTHON), "-I", "-S", "-B"]
        assert call["pass_fds"] and len(call["pass_fds"]) == 1
        assert all(
            marker not in key.upper()
            for key in call["env"]
            for marker in phase2._TOKENISH  # noqa: SLF001
        )
        assert "SSH_AUTH_SOCK" not in call["env"]
        assert call["env"]["HF_HUB_OFFLINE"] == "1"
    assert world.events.index("process-1") < world.events.index("process-2")
    assert world.events.index("source-guard-enter") < world.events.index("process-1")
    assert world.events.index("source-guard-exit") > world.events.index("process-2")

    runs = list((world.layout.build / "phase2-recovery").iterdir())
    assert len(runs) == 1
    stage = runs[0]
    assert stat.S_IMODE(stage.stat().st_mode) == 0o700
    exported = {
        str(path.relative_to(stage / "export"))
        for path in (stage / "export").rglob("*")
        if path.is_file()
    }
    assert exported == set(world.historical)
    for path in (stage / "export").rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o600 if path.is_file() else 0o700)

    expected_capsule = {
        phase1.BEST_PAYLOAD_BASENAME: world.doctor_raw[phase1.BEST_PAYLOAD_BASENAME],
        phase1.TEACHER_CAPTURE_BASENAME: world.bracket_raw["teacher_capture.npz"],
    }
    assert {path.name for path in world.layout.capsule.iterdir()} == set(expected_capsule)
    assert stat.S_IMODE(world.layout.capsule.stat().st_mode) == 0o700
    for name, raw in expected_capsule.items():
        path = world.layout.capsule / name
        assert path.read_bytes() == raw
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
    receipt = world.layout.evidence / str(phase2.FINAL_RECEIPT)
    assert receipt.exists()
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600


def test_tampered_historical_blob_blocks_before_export_or_process(world: World) -> None:
    corrupted = dict(world.historical)
    corrupted["tools/condense/gravity_forge.py"] += b"x"
    world.hooks = phase2.Phase2Hooks(
        **{**world.hooks.__dict__, "load_historical_sources": lambda: corrupted}
    )
    with pytest.raises(phase2.Phase2RecoveryError, match="historical blob bytes changed"):
        _generate(world)
    assert not world.calls
    assert not (world.layout.build / "phase2-recovery").exists()


def _replace_fake_corpus_source(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    source: dict[str, str],
) -> dict[str, bytes]:
    corpus, corpus_raw = _sealed_bytes(
        {"schema": "hawking.test.corpus.v1", "status": "PASS", "source": source}
    )
    historical = dict(world.historical)
    historical["KIMI_K26_CORPUS_INTEGRITY.json"] = corpus_raw
    specs = tuple(
        phase2.BlobSpec(spec.relative_path, _git_blob(corpus_raw), len(corpus_raw), _sha(corpus_raw))
        if spec.relative_path == "KIMI_K26_CORPUS_INTEGRITY.json"
        else spec
        for spec in phase2.HISTORICAL_BLOBS
    )
    monkeypatch.setattr(phase2, "HISTORICAL_BLOBS", specs)
    monkeypatch.setattr(phase2, "CORPUS_SEAL_SHA256", corpus["seal_sha256"])
    return historical


def test_live_shaped_frozen_corpus_source_authority_is_accepted(world: World) -> None:
    verification = phase2._verify_blob_mapping(world.historical)  # noqa: SLF001
    assert verification["corpus_seal_sha256"] == phase2.CORPUS_SEAL_SHA256
    corpus = phase1.strict_json_bytes(
        world.historical["KIMI_K26_CORPUS_INTEGRITY.json"], label="test corpus"
    )
    assert corpus["source"] == {
        "repo": phase1.KIMI_REPO,
        "revision": phase1.KIMI_REVISION,
        "tokenizer_sha256": phase2.CORPUS_TOKENIZER_SHA256,
    }


@pytest.mark.parametrize("mutation", ["missing", "wrong", "extra"])
def test_frozen_corpus_rejects_nonexact_source_fields(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = {
        "repo": phase1.KIMI_REPO,
        "revision": phase1.KIMI_REVISION,
        "tokenizer_sha256": phase2.CORPUS_TOKENIZER_SHA256,
    }
    if mutation == "missing":
        source.pop("tokenizer_sha256")
    elif mutation == "wrong":
        source["tokenizer_sha256"] = "0" * 64
    else:
        source["branch"] = "main"
    historical = _replace_fake_corpus_source(world, monkeypatch, source)
    with pytest.raises(phase2.Phase2RecoveryError, match="corpus source authority changed"):
        phase2._verify_blob_mapping(historical)  # noqa: SLF001


def _install_fake_git_reader(
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    *,
    wrong_tree: bool = False,
    wrong_mode: bool = False,
) -> None:
    git_spec = next(spec for spec in phase2.RUNTIME_FILES if spec.path == phase2.GIT)
    monkeypatch.setattr(
        phase2,
        "_hash_system_binary",
        lambda *_args, **_kwargs: {
            "logical_bytes": git_spec.logical_bytes,
            "sha256": git_spec.sha256,
            "mode": f"{git_spec.mode:04o}",
            "uid": git_spec.uid,
            "hard_links": git_spec.hard_links,
        },
    )

    def fake_read(_repo: Path, *args: str) -> bytes:
        if args == ("cat-file", "-t", phase2.HISTORICAL_COMMIT):
            return b"commit\n"
        if args == ("cat-file", "-p", phase2.HISTORICAL_COMMIT):
            tree = "0" * 40 if wrong_tree else phase2.HISTORICAL_TREE
            return (
                f"tree {tree}\nparent {phase2.HISTORICAL_PARENT}\n\nmessage\n".encode()
            )
        for index, spec in enumerate(phase2.HISTORICAL_BLOBS):
            object_name = f"{phase2.HISTORICAL_COMMIT}:{spec.relative_path}"
            if args == ("rev-parse", "--verify", object_name):
                return f"{spec.git_blob_sha1}\n".encode()
            if args == ("ls-tree", "-z", phase2.HISTORICAL_COMMIT, "--", spec.relative_path):
                mode = "100755" if wrong_mode and index == 1 else spec.git_mode
                return (
                    f"{mode} blob {spec.git_blob_sha1}\t{spec.relative_path}".encode()
                    + b"\x00"
                )
            if args == ("cat-file", "-t", spec.git_blob_sha1):
                return b"blob\n"
            if args == ("cat-file", "-s", spec.git_blob_sha1):
                return f"{spec.logical_bytes}\n".encode()
            if args == ("cat-file", "blob", spec.git_blob_sha1):
                return world.historical[spec.relative_path]
        raise AssertionError(f"unexpected fake Git call: {args}")

    monkeypatch.setattr(phase2, "_git_read", fake_read)


def test_local_git_loader_uses_only_exact_cat_file_allowlist(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fake-repo"
    _private(repo)
    _private(repo / ".git")
    _install_fake_git_reader(monkeypatch, world)
    assert phase2.load_historical_sources(repo) == world.historical


def test_real_system_git_binding_accepts_exact_ssv_hardlinks() -> None:
    spec = next(spec for spec in phase2.RUNTIME_FILES if spec.path == phase2.GIT)
    facts = phase2._hash_system_binary(spec, label="test SSV Git")  # noqa: SLF001
    assert facts["sha256"] == spec.sha256
    assert facts["uid"] == 0
    assert facts["mode"] == "0755"
    assert facts["hard_links"] == 78
    assert facts["ssv_multi_link_allowed"] is True


@pytest.mark.parametrize("failure", ["tree", "mode"])
def test_local_git_loader_rejects_wrong_tree_or_mode(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    repo = tmp_path / f"fake-repo-{failure}"
    _private(repo)
    _private(repo / ".git")
    _install_fake_git_reader(
        monkeypatch,
        world,
        wrong_tree=failure == "tree",
        wrong_mode=failure == "mode",
    )
    with pytest.raises(phase2.Phase2RecoveryError):
        phase2.load_historical_sources(repo)


def test_wrong_bracket_binary_prevents_doctor_and_capsule(world: World) -> None:
    world.runner.corrupt_bracket = "P5_sentinel_expert.k26f1"
    with pytest.raises((phase1.ReleaseCycleError, phase2.Phase2RecoveryError)):
        _generate(world)
    assert len(world.calls) == 1
    assert not world.layout.capsule.exists()
    assert not (world.layout.evidence / str(phase2.FINAL_RECEIPT)).exists()


def test_wrong_doctor_binary_leaves_capsule_untouched(world: World) -> None:
    world.runner.corrupt_doctor = "P5_BASE_OUTPUT_RECOVERY_R31.k26f1"
    with pytest.raises((phase1.ReleaseCycleError, phase2.Phase2RecoveryError)):
        _generate(world)
    assert len(world.calls) == 2
    assert not world.layout.capsule.exists()
    assert not (world.layout.evidence / str(phase2.FINAL_RECEIPT)).exists()


def test_wrong_existing_capsule_is_never_overwritten(world: World) -> None:
    _private(world.layout.capsule)
    wrong = b"wrong-existing-data"
    path = world.layout.capsule / phase1.BEST_PAYLOAD_BASENAME
    _write_private(path, wrong)
    inode = path.stat().st_ino
    with pytest.raises((phase1.ReleaseCycleError, phase2.Phase2RecoveryError)):
        _generate(world)
    assert path.read_bytes() == wrong
    assert path.stat().st_ino == inode
    assert not world.calls


def test_exact_partial_capsule_resumes_without_overwrite(world: World) -> None:
    _private(world.layout.capsule)
    payload = world.doctor_raw[phase1.BEST_PAYLOAD_BASENAME]
    path = world.layout.capsule / phase1.BEST_PAYLOAD_BASENAME
    _write_private(path, payload)
    inode = path.stat().st_ino
    result = _generate(world)
    assert result["status"] == "PASS_EXACT_RECOVERED_CAPSULE"
    assert path.stat().st_ino == inode
    assert path.read_bytes() == payload
    assert (world.layout.capsule / phase1.TEACHER_CAPTURE_BASENAME).read_bytes() == world.bracket_raw[
        "teacher_capture.npz"
    ]


def test_completed_generate_is_idempotent_and_runs_no_second_child(world: World) -> None:
    first = _generate(world)
    receipt_path = world.layout.evidence / str(phase2.FINAL_RECEIPT)
    receipt_inode = receipt_path.stat().st_ino
    receipt_raw = receipt_path.read_bytes()
    world.calls.clear()
    second = _generate(world)
    assert second == first
    assert not world.calls
    assert receipt_path.stat().st_ino == receipt_inode
    assert receipt_path.read_bytes() == receipt_raw


def test_extracts_only_three_frozen_records_and_verify_is_read_only(world: World) -> None:
    _generate(world)
    recovery_files = {
        str(path.relative_to(world.layout.recovery))
        for path in world.layout.recovery.rglob("*")
        if path.is_file()
    }
    assert recovery_files == set(world.frozen_raw)
    world.events.clear()
    world.calls.clear()
    result = phase2.verify(
        world.layout,
        hooks=world.hooks,
        mop_root=world.mop,
        shared_xet=world.shared,
    )
    assert result["status"] == "PASS_EXACT_RECOVERED_CAPSULE"
    assert not world.calls
    assert "extract" not in world.events
    assert "lease-enter" not in world.events


def _different_same_size(raw: bytes) -> bytes:
    assert raw
    return bytes([raw[0] ^ 1]) + raw[1:]


def test_auto_discovered_seed_builds_honest_replacement_without_mutating_seed(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = world.layout.build / phase2.PHASE2_ROOT
    _private(root)
    failed_run = root / "run-failed"
    _private(failed_run)
    seed = failed_run / "thread1.frozen"
    _private(seed)
    _write_private(seed / "teacher_capture.npz", world.bracket_raw["teacher_capture.npz"])
    _write_private(seed / "teacher_capture.json", world.bracket_raw["teacher_capture.json"])
    seed_before = {
        path.name: (path.read_bytes(), path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
        for path in seed.iterdir()
    }

    p1 = _different_same_size(world.bracket_raw["P1_sentinel_expert.k26f1"])
    p5 = _different_same_size(world.bracket_raw["P5_sentinel_expert.k26f1"])
    world.runner.bracket_raw["P1_sentinel_expert.k26f1"] = p1
    world.runner.bracket_raw["P5_sentinel_expert.k26f1"] = p5
    world.runner.bracket_raw["P1_F1_RESULT.json"] = _candidate_result(
        "P1", phase2.BinarySpec("P1_sentinel_expert.k26f1", len(p1), _sha(p1))
    )
    world.runner.bracket_raw["P5_F1_RESULT.json"] = _candidate_result(
        "P5", phase2.BinarySpec("P5_sentinel_expert.k26f1", len(p5), _sha(p5))
    )
    for spec in phase2.DOCTOR_BINARIES:
        replacement = _different_same_size(world.runner.doctor_raw[spec.filename])
        world.runner.doctor_raw[spec.filename] = replacement
        replacement_spec = phase2.BinarySpec(spec.filename, len(replacement), _sha(replacement))
        world.runner.doctor_raw[
            spec.filename.removesuffix(".k26f1") + "_RESULT.json"
        ] = _doctor_result(replacement_spec)

    pinned_binaries = {
        name: _sha(raw)
        for name, raw in {**world.runner.bracket_raw, **world.runner.doctor_raw}.items()
        if name.endswith(".k26f1")
    }
    pinned_results = {
        name: phase2._result_projection_sha256(  # noqa: SLF001
            phase1.strict_json_bytes(raw, label=f"fake replacement {name}")
        )
        for name, raw in {**world.runner.bracket_raw, **world.runner.doctor_raw}.items()
        if name in world.hooks.load_frozen_semantics()
    }
    monkeypatch.setattr(phase2, "PINNED_REPLACEMENT_BINARY_SHA256", pinned_binaries)
    monkeypatch.setattr(
        phase2, "PINNED_REPLACEMENT_RESULT_PROJECTION_SHA256", pinned_results
    )

    world.hooks = phase2.Phase2Hooks(
        **{
            **world.hooks.__dict__,
            "verify_capsule": lambda layout: phase2.verify_capsule_for_release(
                layout,
                mop_root=world.mop,
                shared_xet=world.shared,
                frozen_semantics=world.hooks.load_frozen_semantics(),
            ),
        }
    )
    result = _generate(world)
    assert result["status"] == "PASS_DETERMINISTIC_SEMANTIC_REPLACEMENT_CAPSULE"
    assert result["original_bytes_status"] == "UNRECOVERABLE_ORIGINAL_BYTES"
    assert result["exact_original_stop_condition_met"] is False
    assert len(world.calls) == 4
    for call in world.calls:
        assert {
            key: call["env"][key] for key in phase2.DETERMINISTIC_THREAD_ENVIRONMENT
        } == phase2.DETERMINISTIC_THREAD_ENVIRONMENT
        assert any(
            "torch.use_deterministic_algorithms(True)" in argument
            for argument in call["argv"]
        )

    seed_after = {
        path.name: (path.read_bytes(), path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
        for path in seed.iterdir()
    }
    assert seed_after == seed_before
    capsule_names = {path.name for path in world.layout.capsule.iterdir()}
    assert capsule_names == phase2._replacement_capsule_names()  # noqa: SLF001
    for path in world.layout.capsule.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
    verification = phase2.verify_capsule_for_release(
        world.layout,
        mop_root=world.mop,
        shared_xet=world.shared,
        frozen_semantics=world.hooks.load_frozen_semantics(),
    )
    assert verification["semantic_equivalence_verified"] is True
    assert verification["retained_replays_verified"] is True
    assert verification["pinned_replacement_hash_verified"] is True
    assert verification["payload"]["sha256"] != phase1.BEST_PAYLOAD_SHA256
    lineage = phase1.strict_json_bytes(
        (world.layout.capsule / phase2.REPLACEMENT_LINEAGE).read_bytes(),
        label="test replacement lineage",
    )
    assert lineage["original_bytes_status"] == "UNRECOVERABLE_ORIGINAL_BYTES"
    assert all(
        row["status"] == "UNRECOVERABLE_ORIGINAL_BYTES"
        for row in lineage["original_binaries"]
    )
    assert lineage["residual_gap"]["byte_identity"] == "NOT_REPRODUCED"

    selected_path = world.layout.capsule / phase1.BEST_PAYLOAD_BASENAME
    forged = _different_same_size(selected_path.read_bytes())
    selected_path.write_bytes(forged)
    os.chmod(selected_path, 0o600)
    forged_lineage = dict(lineage)
    forged_lineage.pop("seal_sha256")
    forged_lineage["selected_replacement"] = {
        **forged_lineage["selected_replacement"],
        "sha256": _sha(forged),
    }
    resealed = phase1.seal_document(forged_lineage)
    lineage_path = world.layout.capsule / phase2.REPLACEMENT_LINEAGE
    lineage_path.write_bytes(phase1.canonical_json(resealed) + b"\n")
    os.chmod(lineage_path, 0o600)
    with pytest.raises(phase2.Phase2RecoveryError, match="pinned object"):
        phase2.verify_capsule_for_release(
            world.layout,
            mop_root=world.mop,
            shared_xet=world.shared,
            frozen_semantics=world.hooks.load_frozen_semantics(),
        )


@pytest.mark.parametrize("mutation", ["numeric_residual", "geometry"])
def test_frozen_semantic_verifier_rejects_material_change(
    world: World, mutation: str
) -> None:
    frozen = world.hooks.load_frozen_semantics()["P1_F1_RESULT.json"]
    actual = copy.deepcopy(frozen)
    if mutation == "numeric_residual":
        actual["metrics"]["held_out_score"] += 0.01
        match = "numeric residual"
    else:
        actual["physical_budget"]["complete_ceiling_bytes"] += 1
        match = "frozen semantic value"
    with pytest.raises(phase2.Phase2RecoveryError, match=match):
        phase2._compare_frozen_semantics(  # noqa: SLF001
            actual, frozen, label="test frozen semantic"
        )


def test_generate_parser_supports_optional_seed_without_controller_change() -> None:
    parser = phase2.build_parser()
    plain = parser.parse_args(["generate", "--session", "/private/example"])
    explicit = parser.parse_args(
        [
            "generate",
            "--session",
            "/private/example",
            "--seed-stage",
            "/private/example/build/phase2-recovery/run-old/thread1.frozen",
        ]
    )
    assert plain.seed_stage is None
    assert explicit.seed_stage.name == "thread1.frozen"


def test_no_delete_api_and_cli_has_only_three_commands() -> None:
    source = Path(phase2.__file__).read_text()
    tree = ast.parse(source)
    forbidden = {"unlink", "rmdir", "remove", "rmtree", "rename", "replace"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (calls & forbidden)
    parser = phase2.build_parser()
    choices: set[str] = set()
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, phase2.argparse._SubParsersAction):  # noqa: SLF001
            choices = set(action.choices)
    assert choices == {"preflight", "generate", "verify"}
