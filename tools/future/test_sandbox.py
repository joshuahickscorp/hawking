"""Resident sandbox: provision, bounded FS, named authority refusals.

Negative control: filesystem escape is refused three ways (absolute, ..,
symlink) and each of the higher-authority actions is refused separately
with the authority named. A guard nobody has watched fail is not a guard.

Never asserts that a sparse-checkout path is absent. The source index and
canonical-write trials record which path they took.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hcli.workunit import DEFAULT_RETRY_BUDGET, WorkUnit
from tools.future import mutation_surface as ms
from tools.future import sandbox as sb
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims
from tools.future.resident_optimizer import BoundViolation, OptimizerBound


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    return sb.init_fixture_repo(tmp_path / "canonical")


@pytest.fixture
def box(canonical: Path) -> sb.ResidentSandbox:
    return sb.provision("lab", canonical_root=canonical)


def test_build_and_selftest_emit_sealed_receipt():
    out = sb.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_SANDBOX.json"
    assert doc["schema"] == "hawking.future.sandbox.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["measurement_state"] == "STATIC_ONLY"
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["built"] is True
    _assert_no_hardware_claims(doc)
    assert doc["bounded_filesystem"]["all_refused"] is True
    assert set(doc["bounded_filesystem"]["kinds_proven"]) == {"absolute", "dotdot", "symlink"}
    assert doc["authority"]["all_higher_refused"] is True
    assert {r["authority"] for r in doc["authority"]["refusals_proven"]} == set(sb.REFUSED_HIGHER)
    assert doc["no_model_start"]["starts_model"] is False
    assert doc["longevity"]["second_provision_creates_second_sandbox"] is False
    assert doc["canonical_mutation"]["disjoint"]["disjoint"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    callable_ = doc["resident_callable"]
    assert callable_["can_hcli_invoke"] is True
    assert callable_["entry_point"]
    assert callable_["workunit"]["id"]
    assert callable_["receipt"] == "receipts/future/RESIDENT_SANDBOX.json"
    assert callable_["frontier_fed"]["name"]
    assert callable_["fail_closed"]
    assert "VI" not in "".join(doc["vocabulary"]["eras"])
    assert "IV" not in "".join(doc["vocabulary"]["odysseys"])
    assert doc["counts"]["permitted_autonomous"] == len(sb.PERMITTED_AUTONOMOUS)
    assert doc["counts"]["refused_higher"] == len(sb.REFUSED_HIGHER)
    for row in doc["recovered_implementation"]:
        assert "present" in row
        assert row["what"]
        # Cope with either checkout state; do not require a file to be missing.
        assert row["present"] in {True, False}


def test_provision_creates_worktree_layout_and_is_idempotent(canonical: Path):
    first = sb.provision("lab", canonical_root=canonical)
    assert first.newly_created is True
    assert first.worktree.is_dir()
    assert (first.worktree / ".git").exists()
    assert (first.worktree / sb.STATE_NAME).is_file()
    layout = first.layout()
    for name in sb.LAYOUT_DIRS:
        assert Path(layout[name]).is_dir()
    first.attempt("local_reversible_experiment", name="keep.txt", content="stay\n")

    second = sb.provision("lab", canonical_root=canonical)
    assert second.newly_created is False
    assert os.path.realpath(str(second.worktree)) == os.path.realpath(str(first.worktree))
    assert second.state()["identity_sha256"] == first.state()["identity_sha256"]
    assert second.fs.read_text("experiment/keep.txt") == "stay\n"
    listed = sb._worktree_paths(canonical)
    assert os.path.realpath(str(first.worktree)) in listed


def test_teardown_preserves_artifacts_and_refuses_destroy(box: sb.ResidentSandbox, canonical: Path):
    box.attempt("isolated_build", name="artifact.txt", content="keep-me\n")
    artifact = box.writable_root / "build" / "artifact.txt"
    assert artifact.read_text(encoding="utf-8") == "keep-me\n"
    with pytest.raises(sb.AuthorityRefused, match="destructive_action") as excinfo:
        box.teardown(preserve_artifacts=False)
    assert excinfo.value.authority == "destructive_action"
    report = box.teardown(preserve_artifacts=True)
    assert report["artifacts_preserved"] is True
    assert report["worktree_removed"] is False
    assert artifact.is_file()
    assert artifact.read_text(encoding="utf-8") == "keep-me\n"
    reentered = sb.provision("lab", canonical_root=canonical)
    assert reentered.newly_created is False
    assert reentered.fs.read_text("build/artifact.txt") == "keep-me\n"


def test_longevity_reentry_recovers_same_sandbox_not_a_second(canonical: Path):
    box = sb.provision("lab", canonical_root=canonical)
    box.attempt("receipt_creation", name="one.json", payload={"k": 1})
    identity = box.state()["identity_sha256"]
    worktree = os.path.realpath(str(box.worktree))
    del box

    recovered = sb.provision("lab", canonical_root=canonical)
    assert recovered.newly_created is False
    assert os.path.realpath(str(recovered.worktree)) == worktree
    assert recovered.state()["identity_sha256"] == identity
    payload = json.loads(recovered.fs.read_text("receipts/one.json"))
    assert payload["k"] == 1
    assert payload["gpu_authority"] is False


def test_filesystem_escape_absolute_refused(box: sb.ResidentSandbox, tmp_path: Path):
    victim = tmp_path / "absolute-victim.txt"
    victim.write_text("safe\n")
    with pytest.raises(sb.SandboxEscapeError) as excinfo:
        box.fs.write_text(str(victim.resolve()), "pwned\n")
    assert excinfo.value.kind == "absolute"
    assert "absolute" in str(excinfo.value)
    assert victim.read_text() == "safe\n"


def test_filesystem_escape_dotdot_refused(box: sb.ResidentSandbox):
    sentinel = box.worktree / "SANDBOX.json"
    before = sentinel.read_text(encoding="utf-8") if sentinel.is_file() else None
    with pytest.raises(sb.SandboxEscapeError) as excinfo:
        box.fs.write_text("../SANDBOX.json", "pwned\n")
    assert excinfo.value.kind == "dotdot"
    assert "dotdot" in str(excinfo.value)
    if before is not None:
        assert sentinel.read_text(encoding="utf-8") == before
    else:
        assert not sentinel.exists() or sentinel.read_text(encoding="utf-8") != "pwned\n"


def test_filesystem_escape_symlink_refused(box: sb.ResidentSandbox, tmp_path: Path):
    victim = tmp_path / "symlink-victim.txt"
    victim.write_text("safe\n")
    link = box.writable_root / "escape_link"
    os.symlink(str(victim.resolve()), str(link))
    with pytest.raises(sb.SandboxEscapeError) as excinfo:
        box.fs.write_text("escape_link", "pwned\n")
    assert excinfo.value.kind == "symlink"
    assert "symlink" in str(excinfo.value)
    assert victim.read_text() == "safe\n"


def test_prove_filesystem_refusals_watches_all_three_kinds(box: sb.ResidentSandbox, tmp_path: Path):
    rows = sb.prove_filesystem_refusals(box, outside=tmp_path / "outside.txt")
    assert {r["kind"] for r in rows} == {"absolute", "dotdot", "symlink"}
    assert all(r["refused"] is True for r in rows)
    assert (tmp_path / "outside.txt").read_text() == "untouched\n"


@pytest.mark.parametrize("authority", sorted(sb.REFUSED_HIGHER))
def test_each_higher_authority_refused_by_name(box: sb.ResidentSandbox, authority: str):
    with pytest.raises(sb.AuthorityRefused) as excinfo:
        box.attempt(authority)
    assert excinfo.value.authority == authority
    assert authority in str(excinfo.value)


def test_higher_authority_refusals_are_separate_not_blanket(box: sb.ResidentSandbox):
    rows = sb.prove_higher_authority_refusals(box)
    assert len(rows) == len(sb.REFUSED_HIGHER)
    names = [r["authority"] for r in rows]
    assert names == sorted(sb.REFUSED_HIGHER)
    assert len(set(names)) == len(names)
    for row in rows:
        assert row["refused"] is True
        assert row["authority"] in row["error"]


@pytest.mark.parametrize("action", sorted(sb.PERMITTED_AUTONOMOUS))
def test_each_permitted_autonomous_action_is_authorized(box: sb.ResidentSandbox, action: str):
    decision = box.authorize(action)
    assert decision.allowed is True
    assert decision.authority == action


def test_optimizer_bound_is_reused_and_cannot_widen(box: sb.ResidentSandbox):
    assert isinstance(box.bound.optimizer, OptimizerBound)
    with pytest.raises(BoundViolation, match="cannot grant promotion|authority widening"):
        OptimizerBound(may_widen_authority=True)
    with pytest.raises(BoundViolation, match="cannot grant promotion"):
        OptimizerBound(may_modify_verifier=True)
    with pytest.raises(BoundViolation, match="forbidden authority"):
        OptimizerBound(allowed_authority=frozenset({"read_receipts", "self_promotion"}))
    with pytest.raises(sb.AuthorityRefused, match="authority_widening") as excinfo:
        box.bound.grant("canonical_merge")
    assert excinfo.value.authority == "authority_widening"
    with pytest.raises(sb.AuthorityRefused, match="authority_widening"):
        box.bound.permitted = frozenset(sb.PERMITTED_AUTONOMOUS | {"canonical_merge"})


def test_optimizer_aliases_refuse_under_sandbox_authority_names(box: sb.ResidentSandbox):
    with pytest.raises(sb.AuthorityRefused) as widen:
        box.attempt("widen_authority")
    assert widen.value.authority == "authority_widening"
    with pytest.raises(sb.AuthorityRefused) as verify:
        box.attempt("modify_verifier")
    assert verify.value.authority == "verifier_modification"
    with pytest.raises(sb.AuthorityRefused) as gpu:
        box.attempt("acquire_gpu_lease")
    assert gpu.value.authority == "hardware_risk_action"
    with pytest.raises(sb.AuthorityRefused) as merge:
        box.attempt("mutate_codex_surface")
    assert merge.value.authority == "canonical_merge"


def test_does_not_start_a_model_and_promote_does_not_exist(box: sb.ResidentSandbox):
    with pytest.raises(sb.AuthorityRefused, match="hardware_risk_action") as excinfo:
        box.start_model()
    assert excinfo.value.authority == "hardware_risk_action"
    assert box.state()["model_started"] is False
    assert box.state()["starts_model"] is False
    assert not hasattr(sb.ResidentSandbox, "promote")
    with pytest.raises(AttributeError):
        box.promote()  # type: ignore[attr-defined]


def test_canonical_and_codex_writes_are_refused(box: sb.ResidentSandbox):
    target = REPO / "crates" / "hawking-core" / "Cargo.toml"
    before = target.read_bytes() if target.is_file() else None
    with pytest.raises((sb.SandboxEscapeError, sb.CanonicalMutationError)):
        box.fs.write_text(str(target), "mutated-by-sandbox\n")
    if before is not None:
        assert target.read_bytes() == before
    hcli_target = REPO / "hcli" / "workspace.py"
    hcli_before = hcli_target.read_bytes() if hcli_target.is_file() else None
    with pytest.raises((sb.SandboxEscapeError, sb.CanonicalMutationError)):
        box.fs.write_text(str(hcli_target), "mutated-by-sandbox\n")
    if hcli_before is not None:
        assert hcli_target.read_bytes() == hcli_before


def test_mutation_surface_check_disjoint_on_this_module():
    here = Path(sb.__file__).resolve()
    test = Path(__file__).resolve()
    assert ms.owner(os.path.relpath(here, REPO).replace(os.sep, "/")) == "SIDECAR"
    assert ms.owner(os.path.relpath(test, REPO).replace(os.sep, "/")) == "SIDECAR"
    assert ms.check_disjoint([str(here), str(test)]) == 0
    proof = sb.prove_module_disjoint()
    assert proof["disjoint"] is True
    assert proof["exit_code"] == 0


def test_workunit_roundtrips_hcli_constructor(box: sb.ResidentSandbox):
    row = box.emit_workunit()
    unit = WorkUnit.from_dict(row)
    assert unit.id == row["id"]
    assert unit.id.startswith("future.resident-sandbox.provision.")
    assert unit.verifier == "future.sandbox.provision_contract"
    assert unit.classification == "STATIC_ONLY"
    assert unit.effect_class == "REVERSIBLE"
    assert row["starts_model"] is False
    assert row["may_promote"] is False
    assert row["may_modify_verifier"] is False
    assert row["gpu_windows_held"] == 0
    assert row["budget"]["attempts"] == DEFAULT_RETRY_BUDGET


def test_source_index_copes_with_either_checkout_state(canonical: Path):
    live = sb.build_source_index(REPO)
    assert live["copes_with_sparse_checkout"] is True
    for entry in live["entries"]:
        assert entry["present"] in {True, False}
        if entry["present"]:
            assert entry["file_count"] >= 0
        else:
            assert entry["file_count"] == 0
    fixture = sb.build_source_index(canonical)
    # Fixture repo has none of the declared Hawking roots; that is a recorded
    # path, not a missing-file assertion about the live checkout.
    assert set(fixture["missing_roots"]) | set(fixture["present_roots"]) == set(
        sb.SOURCE_INDEX_ROOTS
    )


def test_interior_write_and_read_roundtrip(box: sb.ResidentSandbox):
    path = box.fs.write_text("experiment/nested/ok.txt", "hello\n")
    assert path.is_file()
    assert box.fs.read_text("experiment/nested/ok.txt") == "hello\n"
    assert box.fs.exists("experiment/nested/ok.txt") is True


def test_unknown_action_denied_by_default(box: sb.ResidentSandbox):
    with pytest.raises(sb.AuthorityRefused) as excinfo:
        box.attempt("choose_the_weather")
    assert excinfo.value.authority == "choose_the_weather"
    assert "deny-by-default" in str(excinfo.value)
