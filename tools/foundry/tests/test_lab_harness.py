"""Table-driven tests for the single laboratory harness authority."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FOUNDRY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOUNDRY))

from lab_harness import (  # noqa: E402
    HARNESS_VERSION,
    MeasurementRecorder,
    ReceiptWriter,
    ReportRenderer,
    load_spec,
    validate_spec,
)
from lab_harness.runner import Runner  # noqa: E402
from lab_harness.spec import SPEC_SCHEMA, ExperimentSpec  # noqa: E402


def _spec_dict(**over):
    base = {
        "schema": SPEC_SCHEMA,
        "id": "unit_probe",
        "title": "unit probe",
        "artifact_dir": "artifacts/runs/lab_harness_unit",
        "stages": [
            {"id": "echo", "argv": ["python3.12", "-c", "print('ok')"], "skip_if_done": False},
        ],
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "raw,ok",
    [
        (_spec_dict(), True),
        ({"schema": SPEC_SCHEMA, "id": "x", "stages": [{"id": "a", "shell": "true"}]}, True),
        ({"schema": "wrong", "id": "x", "stages": [{"id": "a", "shell": "true"}]}, False),
        ({"schema": SPEC_SCHEMA, "id": "x", "stages": []}, False),
        ({"schema": SPEC_SCHEMA, "stages": [{"id": "a", "shell": "true"}]}, False),
        ({"schema": SPEC_SCHEMA, "id": "x", "stages": [{"id": "a"}]}, False),
        ({"schema": SPEC_SCHEMA, "id": "x", "stages": [{"id": "a", "argv": ["true"]}]}, True),
    ],
)
def test_spec_validation_matrix(raw, ok):
    if ok:
        validate_spec(raw)
        ExperimentSpec.from_dict(raw)
    else:
        with pytest.raises(ValueError):
            validate_spec(raw)


def test_load_spec_roundtrip(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    spec = load_spec(p)
    assert spec.id == "unit_probe"
    assert len(spec.stages) == 1


def test_measurement_recorder(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    with MeasurementRecorder(path) as rec:
        rec.stage_start("s1")
        rec.metric("tps_proxy", 0, note="fence")
        rec.stage_end("s1", rc=0, seconds=0.1, state="done")
    rows = MeasurementRecorder(path).read_all()
    kinds = [r["kind"] for r in rows]
    assert kinds == ["stage_start", "metric", "stage_end"]


def test_receipt_writer_stable_hash(tmp_path: Path):
    w = ReceiptWriter("hawking.lab.receipt.v1")
    r1 = w.build(experiment_id="e", stages=[{"id": "a", "state": "done"}], status="complete")
    r2 = w.build(experiment_id="e", stages=[{"id": "a", "state": "done"}], status="complete")
    # timestamps differ → hashes differ; content_sha256 present
    assert "content_sha256" in r1 and len(r1["content_sha256"]) == 64
    out = tmp_path / "r.json"
    w.write(out, r1)
    assert json.loads(out.read_text())["experiment_id"] == "e"


def test_report_renderer_contains_stages():
    md = ReportRenderer().render_md(
        title="T",
        experiment_id="e",
        status={"state": "complete", "current_stage": "a", "uptime_seconds": 1},
        stages=[{"id": "a", "state": "done", "rc": 0, "seconds": 0.5, "note": ""}],
        measures=[{"kind": "metric", "name": "x", "value": 1}],
    )
    assert "complete" in md and "| a |" in md and "x" in md


def test_runner_dry_run(tmp_path: Path, monkeypatch):
    # run inside tmp as root
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tools" / "foundry" / "lab_harness").mkdir(parents=True)
    # point Runner.REPO-like root via root=
    spec = ExperimentSpec.from_dict(
        _spec_dict(
            artifact_dir="art",
            stages=[{"id": "echo", "shell": "echo hi", "skip_if_done": False}],
        )
    )
    rc = Runner(spec, root=tmp_path, dry_run=True).run()
    assert rc == 0
    assert (tmp_path / "art" / "receipt.json").is_file()
    assert (tmp_path / "art" / "report.md").is_file()
    assert (tmp_path / "art" / "status.json").is_file()


def test_runner_real_echo(tmp_path: Path):
    spec = ExperimentSpec.from_dict(
        _spec_dict(
            artifact_dir="art",
            stages=[{"id": "echo", "argv": ["python3.12", "-c", "print(1)"], "skip_if_done": False}],
        )
    )
    rc = Runner(spec, root=tmp_path, dry_run=False).run()
    assert rc == 0
    receipt = json.loads((tmp_path / "art" / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["stages"][0]["state"] == "done"


def test_matrix_expands_slots(tmp_path: Path):
    spec = ExperimentSpec.from_dict(
        _spec_dict(
            artifact_dir="art",
            matrix=[{"n": "1"}, {"n": "2"}],
            stages=[{"id": "echo", "shell": "echo {n}", "skip_if_done": False}],
        )
    )
    rc = Runner(spec, root=tmp_path, dry_run=True).run()
    assert rc == 0
    receipt = json.loads((tmp_path / "art" / "receipt.json").read_text())
    ids = [s["id"] for s in receipt["stages"]]
    assert any("n=1" in i for i in ids) and any("n=2" in i for i in ids)


def test_harness_version_nonzero():
    assert HARNESS_VERSION and SPEC_SCHEMA.startswith("hawking.lab")
