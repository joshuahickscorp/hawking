from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_scaffold.py"
SPEC = importlib.util.spec_from_file_location("appendix_scaffold", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_is_closed_deterministic_and_nonexecuting() -> None:
    first = MODULE.build_plan()
    assert first == MODULE.build_plan()
    assert first["name"] == "The Appendix"
    assert first["execution_supported"] is False
    assert first["active_run_is_primary_corpus"] is True
    assert first["postrun_bridge_schema"] == "hawking.appendix_postrun.v1"

    ids = [cell["id"] for cell in first["cells"]]
    assert len(ids) == len(set(ids))
    known = set(ids)
    assert all(set(cell["depends_on"]) <= known for cell in first["cells"])
    assert all(cell["mutates_active_corpus"] is False for cell in first["cells"])


def test_corpus_snapshot_does_not_open_artifacts(tmp_path: pathlib.Path) -> None:
    (tmp_path / "cell").mkdir()
    (tmp_path / "cell" / "request.json").write_text("not even json", encoding="utf-8")
    (tmp_path / "cell" / "result.json").write_text("also not json", encoding="utf-8")
    (tmp_path / "cell" / "candidate.tq").write_bytes(b"payload")
    snapshot = MODULE.corpus_snapshot(tmp_path)
    assert snapshot["counts"] == {
        "request_receipts": 1,
        "result_receipts": 1,
        "tq_artifacts": 1,
    }
