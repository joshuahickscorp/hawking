"""NOETIC_PARENT_A: rebuilt leader is sealed and immutable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from affine2_g64_lsfit import GROUP_AFFINE, NATIVE_KERNEL_GEO  # noqa: E402
from first_noetic_executable import PARENT_PARAMS, Q4_INCUMBENT_EBPW  # noqa: E402
from noetic_parent_a import (  # noqa: E402
    DURABLE,
    RECEIPT,
    RECORDED_DISPATCHES,
    RECORDED_EBPW,
    RECORDED_TEXT,
    RECORDED_TOKEN_IDS,
    SCHEMA,
    catalog_complete,
    durable_root,
    path_is_durable,
    reseal,
)


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        "receipts/headless/NOETIC_PARENT_A.json missing — "
        "run python3 tools/headless/noetic_parent_a.py"
    )
    return json.loads(RECEIPT.read_text())


def test_durable_path_is_outside_worktree_models_and_repo():
    root = durable_root()
    loc = path_is_durable(root)
    assert loc["outside_lane_worktree"], loc
    assert loc["outside_repo"], loc
    assert loc["outside_models"], loc
    assert root == DURABLE.resolve() or str(root).startswith(str(Path.home() / "noetic"))
    assert "/worktrees/" not in str(root)
    assert "/models/" not in str(root)
    assert not str(root).startswith(str(REPO.resolve()))


def test_rebuilt_artifact_has_192_affine_and_catalog():
    root = durable_root()
    assert catalog_complete(root), (
        f"durable artifact incomplete at {root}: need catalog.hq38m20 + 192 .hgrafv01"
    )
    cat = root / "catalog.hq38m20"
    assert cat.read_bytes()[:8] == b"HQ38M20\0"


def test_receipt_schema_genomes_and_byte_split():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["immutable"] is True
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["parent_params"] == PARENT_PARAMS
    loc = doc["artifact"]["durable"]
    assert loc["outside_lane_worktree"]
    assert loc["outside_repo"]
    assert loc["outside_models"]
    assert Path(doc["artifact"]["path"]).resolve() == durable_root()
    for genome in (
        "RepresentationGenome",
        "KernelGenome",
        "RuntimeGenome",
        "MachineGenome",
    ):
        assert genome in doc and doc[genome], genome
    rep = doc["RepresentationGenome"]
    assert rep["group"] == GROUP_AFFINE == 64
    assert rep["fit"] == "least_squares_scale_bias"
    assert "q * scale + bias" in rep["codec"] or "q * scale + bias" in rep["reconstruction"]
    assert rep["n_affine"] == 192
    assert abs(rep["affine_tensor_storage_bpw"] - 2.5) < 1e-12
    kern = doc["KernelGenome"]
    assert kern["production_kernel"] == NATIVE_KERNEL_GEO
    assert "exact_metal_source_hashes" in kern["metal_source_hashes"] or "affine2_group32_matvec.metal" in kern["metal_source_hashes"]
    metal = kern.get("metal_source_hashes") or {}
    exact = metal.get("exact_metal_source_hashes") or metal
    assert exact["affine2_group32_matvec.metal"]["sha256"]
    assert exact["q80_mixed_decode.metal"]["sha256"]
    assert kern["compiler_settings"]["metal"]["api"] == "MTLDevice::newLibraryWithSource"
    assert doc["MachineGenome"]["chipset"]
    assert doc["MachineGenome"]["genome_digest"]
    for key in (
        "MODEL_SPECIFIC_BYTES",
        "SHARED_RUNTIME_BYTES",
        "MACHINE_SPECIFIC_BYTES",
        "GENERATED_CACHE_BYTES",
        "RESIDENT_BYTES",
        "ACTIVE_BYTES",
        "active_bytes_per_token",
    ):
        assert isinstance(doc[key], (int, float)) and doc[key] > 0, key
    assert doc["active_bytes_per_token"] == doc["ACTIVE_BYTES"]
    assert "dispatch_count" in doc
    assert "complete_token_wall" in doc
    assert "capability_evidence" in doc
    cap = doc["capability_evidence"]
    assert cap["gpu_ran"] is True
    assert cap["dense_w_materialized"] == 0
    assert cap["expanded_to_q4"] == 0
    assert cap["expanded_to_float_gemv"] == 0


def test_complete_ebpw_reproduced():
    doc = _receipt()
    ebpw = doc["RepresentationGenome"]["complete_ebpw"]
    assert abs(ebpw - RECORDED_EBPW) < 1e-12, (
        f"complete EBPW drifted: measured {ebpw} recorded {RECORDED_EBPW} "
        f"delta {ebpw - RECORDED_EBPW}"
    )
    assert abs(ebpw - 3.139300850311054) < 1e-12
    compile_ = doc["compile"]
    assert compile_["n_affine"] == 192
    assert compile_["n_tensors"] == 755
    assert abs(compile_["complete_ebpw"] - ebpw) < 1e-15
    assert abs(compile_["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9


def test_reproduction_table_is_honest():
    doc = _receipt()
    repro = doc["reproduction"]
    assert "rows" in repro
    rows = repro["rows"]
    assert "complete_ebpw" in rows
    assert "dispatches_per_token" in rows
    assert "decode_tok_s" in rows
    for name, row in rows.items():
        assert "recorded" in row and "measured" in row and "match" in row, name
        if row["measured"] is not None and isinstance(row["recorded"], (int, float)):
            assert "delta" in row
    # EBPW is a function of bytes and must match exactly.
    assert rows["complete_ebpw"]["match"] is True
    disp = rows["dispatches_per_token"]
    if disp["measured"] is not None:
        if disp["match"] is False:
            # Honest miss is allowed in the receipt; the delta must be named.
            assert disp["delta"] is not None
        else:
            assert disp["measured"] == RECORDED_DISPATCHES
    text = rows["verbatim_text"]["measured"]
    if text:
        assert isinstance(text, str)
        assert len(text) > 0


def test_capability_counters_and_sixteen_tokens():
    doc = _receipt()
    cap = doc["capability_evidence"]
    assert cap["dense_w_materialized"] == 0
    assert cap["expanded_to_q4"] == 0
    assert cap["expanded_to_float_gemv"] == 0
    ids = cap["coherence"]["new_token_ids"] or doc["verbatim"]["new_token_ids"]
    text = cap["coherence"]["text"] or doc["verbatim"]["generated_text"]
    assert isinstance(text, str) and text
    assert isinstance(ids, list) and len(ids) == 16
    par = cap["parity_fused_vs_unfused"]
    assert par.get("max_abs_diff") is not None
    wall = doc["complete_token_wall"]
    assert wall.get("tok_s_mean") is not None or wall.get("mean_decode_wall_s") is not None


def test_sealed_bytes_fail_if_changed():
    """The parent is immutable. A changed file must change the sealed hash."""
    doc = _receipt()
    sealed = doc["executable_closure"]["closure_sha256"]
    assert isinstance(sealed, str) and len(sealed) == 64
    live = reseal(durable_root())
    assert live["closure_sha256"] == sealed, (
        "sealed model-specific bytes changed under ~/noetic/NOETIC_PARENT_A/ — "
        f"receipt {sealed} live {live['closure_sha256']}"
    )
    assert live["n_affine"] == 192
    assert live["n_files"] == doc["executable_closure"]["n_files"]


def test_reseal_reproduces_the_same_closure_hash():
    first = reseal(durable_root())
    second = reseal(durable_root())
    doc = _receipt()
    sealed = doc["executable_closure"]["closure_sha256"]
    assert first["closure_sha256"] == second["closure_sha256"] == sealed


def test_receipt_names_the_commands_that_produced_the_numbers():
    doc = _receipt()
    cmds = doc["commands"]
    assert "rebuild" in cmds
    assert "measure" in cmds
    assert "seal" in cmds
    assert "test" in cmds
    assert cmds["test"][-2:] == ["tools/headless", "-q"] or "pytest" in json.dumps(cmds["test"])
