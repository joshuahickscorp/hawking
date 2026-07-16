from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "tq_runtime_matrix.py"
SPEC = importlib.util.spec_from_file_location("tq_runtime_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matrix_is_deterministic_closed_and_nonexecuting() -> None:
    matrix = MODULE.build_matrix()
    assert matrix == MODULE.build_matrix()
    assert matrix["execution_supported"] is False
    assert matrix["counts"]["total"] == len(matrix["cells"])
    ids = {cell["id"] for cell in matrix["cells"]}
    assert len(ids) == len(matrix["cells"])
    assert all(set(cell["depends_on"]) <= ids for cell in matrix["cells"])


def test_implemented_candidates_depend_on_stored_and_ragged_is_not_dropped() -> None:
    matrix = MODULE.build_matrix()
    candidate = next(
        cell for cell in matrix["cells"]
        if cell["tensor_family"] == "qwen_3b_ffn_up_gate"
        and cell["k_bits"] == 3
        and cell["runtime_path"] == "compact"
    )
    assert candidate["state"] == "deferred"
    assert len(candidate["depends_on"]) == 1
    baseline = next(cell for cell in matrix["cells"] if cell["id"] == candidate["depends_on"][0])
    assert baseline["runtime_path"] == "stored"

    ragged = next(
        cell for cell in matrix["cells"]
        if cell["tensor_family"] == "qwen_0_5b_attn_q_o"
        and cell["k_bits"] == 3
        and cell["runtime_path"] == "stored"
    )
    assert ragged["state"] == "blocked_geometry"
    assert ragged["blocker"]


def test_future_recipe_is_design_deferred_without_fake_receipt_schema() -> None:
    cell = next(
        cell for cell in MODULE.build_matrix()["cells"]
        if cell["research_recipe"] == "compact_computed"
    )
    assert cell["state"] == "design_deferred"
    assert cell["runtime_path"] is None
    assert cell["receipt_schema"] is None
