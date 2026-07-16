from __future__ import annotations

import importlib.util
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "condense" / "spec_reentry_scaffold.py"
SPEC = importlib.util.spec_from_file_location("spec_reentry_scaffold", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matrix_is_deterministic_unique_and_closed() -> None:
    first = MODULE.build_matrix("7B")
    second = MODULE.build_matrix("7B")
    assert first == second
    assert first["execution_supported"] is False

    ids = [cell["id"] for cell in first["cells"]]
    assert len(ids) == len(set(ids))
    assert all(cell["state"] == "deferred" for cell in first["cells"])

    known = set(ids)
    assert all(set(cell["depends_on"]) <= known for cell in first["cells"])


def test_matrix_covers_both_appendages_and_all_workloads() -> None:
    matrix = MODULE.build_matrix("7B")
    families = {cell["family"] for cell in matrix["cells"]}
    assert "tq_batched_verifier_parity" in families
    assert "verifier_cost_curve" in families
    assert "exact_free_proposer_oracle" in families
    assert "parallel_draft_head" in families
    assert "block_diffusion_draft" in families
    assert "block_iterative_draft" in families
    assert "metal_tree_verify" in families
    assert "target_draft_runtime_composition" in families

    free_cells = [
        cell for cell in matrix["cells"]
        if cell["family"] == "exact_free_proposer_oracle"
    ]
    assert {cell["knobs"]["workload"] for cell in free_cells} == set(MODULE.WORKLOADS)


def test_owner_inventory_includes_doctor_mop_and_cognitive_corpus() -> None:
    output = "\n".join([
        "101 python doctor_v5_ultra_accelerated_queue.py run --nonce x",
        "102 python mop_generation1_campaign.py run --execute",
        "103 python generation1_cognitive_corpus.py --config x",
        "104 python harmless_script.py",
        "105 /bin/ps -axo pid=,command=",
    ])
    owners = MODULE._owners_from_ps_output(output, own_pid=105)
    assert [row["pid"] for row in owners] == [101, 102, 103]
    assert all(row["matched_patterns"] for row in owners)


def test_owner_inventory_fails_closed_when_ps_probe_fails(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=10)

    monkeypatch.setattr(MODULE.subprocess, "run", fail)
    owners = MODULE.active_heavy_owners()
    assert owners == [{
        "pid": 0,
        "command": "owner-probe-unavailable",
        "matched_patterns": [],
        "probe_error": "TimeoutExpired: Command '['"
        + str(MODULE.PS_PATH)
        + "', '-axo', 'pid=,command=']' timed out after 10 seconds",
    }]
