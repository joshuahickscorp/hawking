"""G021 promotion controls: the mutation caused the win, or the harness is lying.

No GPU, no 27B, no live llama-server. The score is admitted_n through
Controller.ensure_runtime_pool on a git-archive scratch copy of HEAD hcli.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "SELF_OPT_CANDIDATE_PROMOTED.json"

spec = importlib.util.spec_from_file_location(
    "hcli_self_optimize_2", HERE / "hcli_self_optimize_2.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_compute_decision_refuses_noop_identical_admission():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=False,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=1,
        mut_med=1,
        spread=0,
        admission_differs=False,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "REFUSED"
    assert d["would_refuse"] is True
    assert d["refuse_if"]["admission_did_not_differ"]["triggered"] is True
    assert d["refuse_if"]["throughput_did_not_improve_beyond_spread"]["triggered"] is True


def test_compute_decision_refuses_bad_worse_score():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=False,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=1,
        mut_med=0,
        spread=0,
        admission_differs=True,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "REFUSED"
    assert d["decision"] == "reject"
    assert "did not improve" in d["reason"]


def test_compute_decision_refuses_no_evidence_even_if_score_moved():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=False,
        validation_reason="NO_EVIDENCE",
        orig_med=1,
        mut_med=2,
        spread=0,
        admission_differs=True,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "REFUSED"
    assert "NO_EVIDENCE" in d["reason"]


def test_compute_decision_promotes_only_when_every_gate_is_green():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=1,
        mut_med=2,
        spread=0,
        admission_differs=True,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "PROMOTE"
    assert d["would_refuse"] is False


def test_failing_gate_trial_computes_would_refuse_from_verdict():
    ws = Path(tempfile.mkdtemp(prefix="g021-failing-gate-"))
    trial = mod.run_failing_gate_trial(ws)
    assert trial["hardcoded"] is False
    assert trial["pytest_exit_code"] != 0
    assert trial["pytest_passed_gate"] is False
    failing = trial["decision_on_failing_correctness"]
    assert failing["verdict"] == "REFUSED"
    assert trial["would_refuse_on_failing_gate"] == (failing["verdict"] == "REFUSED")
    assert trial["would_refuse_on_failing_gate"] is True
    assert trial["would_refuse_on_no_evidence"] is True
    assert trial["evidenced"] is True


def test_would_refuse_on_failing_gate_is_not_hardcoded_true():
    src = inspect.getsource(mod.run_failing_gate_trial)
    assert 'failing.get("verdict") == "REFUSED"' in src
    assert "would_refuse_on_failing_gate\": True" not in src
    assert "would_refuse_on_failing_gate'] = True" not in src


def test_four_controls_physically_ran():
    receipt = mod.run_promotion_controls(REPO)
    assert receipt["schema"] == "hawking.headless.hcli_self_opt.candidate_promoted.v1"
    controls = receipt["controls"]
    for name in ("noop", "bad", "paired_interleaved", "failing_gate"):
        assert controls[name]["ran"] is True, name

    noop = controls["noop"]
    assert noop["is_win"] is False, (
        "NO-OP scored as a win; the harness is measuring something other "
        f"than the mutation: {noop}"
    )
    assert noop["through_mutated_mechanism"] is True
    assert noop["admission_differs"] is False
    assert (noop.get("decision") or {}).get("verdict") == "REFUSED"

    bad = controls["bad"]
    assert bad["is_win"] is False
    assert (bad.get("decision") or {}).get("verdict") == "REFUSED"
    assert bad["through_mutated_mechanism"] is True
    assert set(bad["candidate_admitted_n"]) == {0}
    assert set(bad["baseline_admitted_n"]) == {1}

    paired = controls["paired_interleaved"]
    assert paired["block_design"] is False
    order = paired["h1_order"]
    assert order == ["candidate", "baseline", "candidate", "baseline"]
    h1 = paired["h1"]
    assert h1["admission_differs"] is True
    assert set(h1["candidate_admitted_n"]) == {2}
    assert set(h1["baseline_admitted_n"]) == {1}
    assert h1["through_mutated_mechanism"] is True

    fail = controls["failing_gate"]
    assert fail["hardcoded"] is False
    assert fail["would_refuse_on_failing_gate"] is True
    assert fail["pytest_exit_code"] != 0
    assert receipt["would_refuse_on_failing_gate"] == fail["would_refuse_on_failing_gate"]

    validation = (receipt.get("mutation_receipt") or {}).get("validation") or {}
    assert validation.get("ok") is True
    assert validation.get("reason") != "NO_EVIDENCE"

    cause = receipt.get("root_cause") or {}
    assert cause.get("id") == "G021_SCRATCH_IMPORT_SHADOW"
    measured = cause.get("measured_this_run") or {}
    assert measured.get("all_trials_imported_scratch") is True
    assert cause.get("h1_at_head") is True
    assert cause.get("four_controls_unchanged") == {
        "noop_must_not_win": True,
        "bad_must_be_refused": True,
        "paired_interleaved_not_blocks": True,
        "failing_gate_physically_exercised": True,
    }
    for trial in (receipt.get("trials") or {}).get("h1") or []:
        assert trial.get("import_root_is_scratch") is True, trial
        if trial.get("condition") == "baseline":
            assert trial.get("controller_has_h1_wiring") is False, trial
            assert trial.get("observed_overlap_ctor") is None, trial
            assert trial.get("admitted_n") == 1, trial
        if trial.get("condition") == "candidate":
            assert trial.get("controller_has_h1_wiring") is True, trial
            assert trial.get("admitted_n") == 2, trial

    # H1 is already HEAD. The 1→2 delta is against a synthetic original,
    # not a tree change. Tree-level verdict is REFUSED, not PROMOTE.
    assert receipt["decision"] == "reject"
    assert receipt["decision_verdict"] == "REFUSED"
    assert "h1_equals_head" in (receipt.get("decision_reason") or "").lower() or (
        "already HEAD" in (receipt.get("decision_reason") or "")
    )

    assert RECEIPT.is_file()
    disk = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert disk["schema"] == receipt["schema"]
    assert disk["controls"]["noop"]["is_win"] is False
    assert disk["controls"]["bad"]["decision"]["verdict"] == "REFUSED"
    assert disk["would_refuse_on_failing_gate"] is True
    assert disk["decision_verdict"] == "REFUSED"
    assert (disk.get("root_cause") or {}).get("id") == "G021_SCRATCH_IMPORT_SHADOW"


def _materialized_variants(repo: Path):
    scratch = Path(tempfile.mkdtemp(prefix="g021-scratch-"))
    mod.materialize_hcli_from_head(repo, scratch)
    variants = mod.controller_variants(
        (scratch / "hcli" / "controller.py").read_text(encoding="utf-8")
    )
    return scratch, variants


def _probe_from_script_parent(
    *,
    scratch: Path,
    controller_text: str,
    with_hcli: bool,
    repo: Path,
) -> dict:
    """Run the admit child with a copy of the harness living under `parent`.

    When with_hcli is True, parent/hcli is a git-archive of HEAD (H1 present).
    That is the main-checkout case that used to shadow scratch. The child's
    Path(__file__).parents[2] is parent, so ensure_hcli_path used to insert
    HEAD in front of scratch.
    """
    (scratch / "hcli" / "controller.py").write_text(controller_text, encoding="utf-8")
    mod.clear_controller_pyc(scratch)
    parent = Path(tempfile.mkdtemp(prefix="g021-script-parent-"))
    dest = parent / "tools" / "headless"
    dest.mkdir(parents=True)
    shutil.copy2(HERE / "hcli_self_optimize_2.py", dest / "hcli_self_optimize_2.py")
    if with_hcli:
        mod.materialize_hcli_from_head(repo, parent)
        head = (parent / "hcli" / "controller.py").read_text(encoding="utf-8")
        assert "observed_overlap=load_observed_overlap(self.workspace_root)" in head
    out = parent / "out.json"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(scratch)
    env["HCLI_SELFOPT_HCLI_PARENT"] = str(scratch)
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            str(dest / "hcli_self_optimize_2.py"),
            "--probe-controller-admit",
            "--out",
            str(out),
            "--repo",
            str(scratch),
        ],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert out.is_file(), (proc.returncode, proc.stderr[-800:], proc.stdout[-400:])
    data = json.loads(out.read_text(encoding="utf-8"))
    data["child_exit"] = proc.returncode
    data["_script_parent_has_hcli"] = with_hcli
    data["_script_parent"] = str(parent)
    return data


def test_pin_drops_materialized_hcli_trees_on_sys_path():
    scratch, variants = _materialized_variants(REPO)
    (scratch / "hcli" / "controller.py").write_text(
        variants["original"], encoding="utf-8"
    )
    shadow = Path(tempfile.mkdtemp(prefix="g021-shadow-"))
    mod.materialize_hcli_from_head(REPO, shadow)
    assert "observed_overlap=load_observed_overlap" in (
        shadow / "hcli" / "controller.py"
    ).read_text(encoding="utf-8")

    saved_path = list(sys.path)
    saved_env = os.environ.get("HCLI_SELFOPT_HCLI_PARENT")
    try:
        for key in list(sys.modules):
            if key == "hcli" or key.startswith("hcli."):
                del sys.modules[key]
        sys.path.insert(0, str(shadow))
        sys.path.insert(0, str(scratch))
        pinned = mod.pin_hcli_import_root(scratch)
        assert pinned == scratch.resolve()
        import hcli.controller as ctrl_mod

        assert Path(ctrl_mod.__file__).resolve().is_relative_to(scratch.resolve())
        src = Path(ctrl_mod.__file__).read_text(encoding="utf-8")
        assert "observed_overlap=load_observed_overlap(self.workspace_root)" not in src
        identity = mod._hcli_loaded_from(scratch)
        assert identity["import_root_is_scratch"] is True
    finally:
        for key in list(sys.modules):
            if key == "hcli" or key.startswith("hcli."):
                del sys.modules[key]
        sys.path[:] = saved_path
        if saved_env is None:
            os.environ.pop("HCLI_SELFOPT_HCLI_PARENT", None)
        else:
            os.environ["HCLI_SELFOPT_HCLI_PARENT"] = saved_env


def test_stripped_baseline_same_verdict_with_and_without_hcli_on_script_parent():
    """The G021 acceptance: hcli/ present vs absent must not change the probe.

    Pre-fix, a script-parent with hcli/ imported HEAD (ctor=2, admit=2) while
    the disk variant was stripped. A script-parent without hcli/ imported
    scratch (ctor=None, admit=1). After pin_hcli_import_root both admit 1.
    """
    scratch, variants = _materialized_variants(REPO)
    absent = _probe_from_script_parent(
        scratch=scratch,
        controller_text=variants["original"],
        with_hcli=False,
        repo=REPO,
    )
    present = _probe_from_script_parent(
        scratch=scratch,
        controller_text=variants["original"],
        with_hcli=True,
        repo=REPO,
    )
    for row, label in ((absent, "absent"), (present, "present")):
        assert row.get("ok") is True, (label, row.get("error"), row.get("child_stderr"))
        assert row.get("import_root_is_scratch") is True, (label, row)
        assert row.get("controller_has_h1_wiring") is False, label
        assert row.get("observed_overlap_ctor") is None, (label, row)
        assert row.get("admitted_n") == 1, (label, row)
        # Admit-time snapshot: original does not pass workspace, so the
        # pool's workspace is the empty isolate. Complete may later store
        # a high-water mark there; that is not the source of admitted_n.
        assert row.get("loaded_overlap_on_isolate") is None, (label, row)
        assert row.get("stored") == 2, (label, row)
        assert row.get("loaded") == 2, (label, row)
    assert absent["admitted_n"] == present["admitted_n"] == 1
    assert absent["observed_overlap_ctor"] == present["observed_overlap_ctor"]

    h1_absent = _probe_from_script_parent(
        scratch=scratch,
        controller_text=variants["h1"],
        with_hcli=False,
        repo=REPO,
    )
    h1_present = _probe_from_script_parent(
        scratch=scratch,
        controller_text=variants["h1"],
        with_hcli=True,
        repo=REPO,
    )
    for row, label in ((h1_absent, "h1-absent"), (h1_present, "h1-present")):
        assert row.get("ok") is True, (label, row.get("error"))
        assert row.get("admitted_n") == 2, (label, row)
        assert row.get("controller_has_h1_wiring") is True, label
        assert row.get("import_root_is_scratch") is True, (label, row)
    assert h1_absent["admitted_n"] == h1_present["admitted_n"] == 2

    bad_absent = _probe_from_script_parent(
        scratch=scratch,
        controller_text=variants["bad"],
        with_hcli=False,
        repo=REPO,
    )
    bad_present = _probe_from_script_parent(
        scratch=scratch,
        controller_text=variants["bad"],
        with_hcli=True,
        repo=REPO,
    )
    for row, label in ((bad_absent, "bad-absent"), (bad_present, "bad-present")):
        assert row.get("ok") is True, (label, row.get("error"))
        assert row.get("admitted_n") == 0, (label, row)
        assert row.get("requested_n") == 0, (label, row)
        assert row.get("controller_has_requested_n_zero") is True, label
    assert bad_absent["admitted_n"] == bad_present["admitted_n"] == 0
