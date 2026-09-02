"""Representation-family chain: source families, plugin registration, EBPW.

A capability does not exist until something CALLS it. Invoked symbols below
are the pack/execute/constructor names, not module imports.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tools.odyssey import noetic_compiler as nc
from tools.future import complete_ebpw as ce

REPO = Path(__file__).resolve().parents[2]
CORE = list(nc.CORE_MODULE_RELS)

EXECUTING_SOURCE = (
    "grouped_absmax_q4",
    "ternary_group64",
    "binary_sign_codes",
    "low_rank_uv",
    "product_quantization",
    "raw_f32",
)


def _core_text() -> str:
    return "\n".join((REPO / rel).read_text() for rel in CORE)


def test_source_family_invoked_symbols_exist_in_named_files():
    """STATIC: each source family cites a symbol the named file actually defines."""
    inv = nc.family_inventory()
    assert inv["evidence_tier"] == "STATIC"
    assert inv["source_family_ids"]
    for row in inv["source_families"]:
        path = REPO / row["source_path"]
        assert path.is_file(), row["source_path"]
        src = path.read_text()
        for symbol in row["invoked_symbols"]:
            assert f"def {symbol}" in src or f"class {symbol}" in src, (
                f"{row['family_id']}: {symbol} is not defined in {row['source_path']}"
            )


def test_roadmap_destination_list_is_quoted_section_10_2():
    inv = nc.family_inventory()
    assert inv["roadmap"]["section"] == "10.2"
    assert inv["roadmap"]["line"] == 1406
    assert "tensor networks" in inv["roadmap"]["families"]
    assert "clustered bases" in inv["roadmap"]["families"]
    # Destination-only: source does not support these.
    assert "clustered bases" in inv["destination_only"]
    assert "tensor networks" in inv["destination_only"]
    assert "tokenizer/state redesign" in inv["destination_only"]
    assert "expert common backbone + expert deltas" in inv["destination_only"]
    assert "generated experts" in inv["destination_only"]


def test_source_does_not_invent_tensor_networks():
    ids = set(nc.family_inventory()["source_family_ids"])
    assert "tensor_train" not in ids
    assert "tensor_networks" not in ids
    assert "tensor_contraction" not in ids


def test_each_source_family_has_chain_status_with_named_blockers():
    for spec in nc.list_families():
        if nc._is_plugin_path(spec.source_path):
            continue
        status = nc.chain_status(spec)
        assert set(status["links"]) == set(nc.CHAIN_LINKS)
        for name, link in status["links"].items():
            assert link["status"] in {"PRESENT", "BLOCKED"}, (spec.family_id, name)
            if link["status"] == "BLOCKED":
                assert link.get("blocker") or status["blockers"], (
                    f"{spec.family_id}.{name} is BLOCKED with no named blocker"
                )


def test_executing_source_families_round_trip():
    for family_id in EXECUTING_SOURCE:
        result = nc.round_trip(family_id)
        assert result["verified"] is True
        assert result["reconciled"] is True
        assert result["execute"]["match_atol_1e5"] is True
        assert result["accounting"]["stored_bytes"] > 0
        ok, bad = nc._nr_container().validate(result["nr"])
        assert ok, (family_id, bad)
        assert "kernel" not in result["nr"]["representation"]
        assert result["lowering"]["kind"] == "semantic_interpreter"


def test_incomplete_source_families_name_blockers_and_do_not_pretend_to_execute():
    for family_id in (
        "shared_basis",
        "q2_affine",
        "sparse_correction",
        "exact_island",
        "generated_block",
        "recurrent_state_operator",
        "routed_group_execution",
    ):
        spec = nc.get_family(family_id)
        assert spec.executes is False
        assert spec.blockers, family_id
        status = nc.chain_status(spec)
        assert status["complete"] is False
        if spec.demo_payload is None:
            with pytest.raises(nc.FamilyChainBlocked):
                nc.round_trip(family_id)
        else:
            result = nc.round_trip(family_id)
            assert result["execute"] is None
            assert result["reconciled"] is True


def test_toy_family_registers_end_to_end_without_core_edits():
    core_src = _core_text()
    assert "toy_xor_codes" not in core_src
    assert "toy_xor" not in core_src

    nc.ensure_families()
    spec = nc.get_family("toy_xor_codes")
    assert nc._is_plugin_path(spec.source_path)
    assert spec.source_path == "tools/odyssey/families/toy_xor_codes.py"

    result = nc.round_trip("toy_xor_codes")
    assert result["verified"] is True
    assert result["reconciled"] is True
    assert result["execute"]["match_atol_1e5"] is True
    names = {p["name"] for p in result["accounting"]["parts"]}
    assert "toy_xor_codes" in names
    assert "xor_key" in names
    assert "xor_decoder_stub" in names

    diff = subprocess.check_output(
        ["git", "diff", "--name-only", "--", *CORE],
        cwd=REPO,
        text=True,
    )
    # The plugin lives outside core. git diff --name-only of core is shown
    # so a core edit required to register this family is visible.
    toy_rel = spec.source_path
    assert toy_rel not in CORE
    assert Path(toy_rel).name not in {Path(c).name for c in CORE}
    print("git diff --name-only -- core modules:\n" + (diff or "(empty)"))


def test_ephemeral_family_plugin_does_not_modify_core(tmp_path):
    """Load-bearing: a NEW family file round-trips with zero core edits."""
    before = subprocess.check_output(
        ["git", "diff", "--name-only", "--", *CORE],
        cwd=REPO,
        text=True,
    )
    plugin = tmp_path / "ephemeral_shift.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from tools.odyssey.noetic_compiler import (
                STREAM_WEIGHT_CODES,
                FamilySpec,
                register_family,
            )
            import numpy as np

            def pack(W):
                W = np.asarray(W, dtype=np.float32)
                return {
                    "rows": int(W.shape[0]),
                    "cols": int(W.shape[1]),
                    "values": np.asarray(W, dtype="<f4").tobytes(),
                }

            def execute(payload, x):
                W = np.frombuffer(payload["values"], dtype="<f4").reshape(
                    payload["rows"], payload["cols"]
                )
                return W @ np.asarray(x, dtype=np.float32)

            def reconstruct(payload):
                import numpy as np
                return np.frombuffer(payload["values"], dtype="<f4").reshape(
                    payload["rows"], payload["cols"]
                )

            def demo_payload():
                rng = np.random.RandomState(1)
                return pack(rng.randn(3, 4).astype(np.float32))

            def bill_parts(payload):
                return {
                    "representation": [{
                        "name": "ephemeral_values",
                        "bytes": len(payload["values"]),
                        "stream_class": STREAM_WEIGHT_CODES,
                    }],
                }

            register_family(FamilySpec(
                family_id="ephemeral_shift",
                ir_kind="ephemeral_shift",
                source_path=str(__file__),
                invoked_symbols=("pack", "execute"),
                executes=True,
                backend="INTERPRETER",
                backend_kernel=None,
                evidence_tier="FUNCTIONAL_SIM",
                test_rel="tools/odyssey/test_noetic_representation_chain.py",
                pack=pack,
                execute=execute,
                reconstruct=reconstruct,
                demo_payload=demo_payload,
                bill_parts=bill_parts,
                kernel_requirements=({"requires": "ephemeral_shift_decoder"},),
            ))
            """
        )
    )
    nc.load_plugins(extra_paths=[plugin])
    result = nc.round_trip("ephemeral_shift")
    assert result["verified"] is True
    assert result["reconciled"] is True
    assert result["execute"]["match_atol_1e5"] is True
    after = subprocess.check_output(
        ["git", "diff", "--name-only", "--", *CORE],
        cwd=REPO,
        text=True,
    )
    assert after == before
    print("git diff --name-only -- core (after ephemeral register):\n" + (after or "(empty)"))


def test_ebpw_refuses_unbilled_component_from_family_artifact():
    cand = ce.incumbent_candidate()
    cand["hidden_residual"] = [
        {
            "name": "unbilled",
            "bytes": 4096,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.cost(cand)


def test_second_toy_family_registers_end_to_end_without_core_edits():
    """Second plugin family. Core must not name it; round-trip must pass."""
    core_src = _core_text()
    assert "toy_mean_residual" not in core_src
    assert "mean_residual_decoder_stub" not in core_src

    nc.ensure_families()
    spec = nc.get_family("toy_mean_residual")
    assert nc._is_plugin_path(spec.source_path)
    assert spec.source_path == "tools/odyssey/families/toy_mean_residual.py"

    result = nc.round_trip("toy_mean_residual")
    assert result["verified"] is True
    assert result["reconciled"] is True
    assert result["execute"]["match_atol_1e5"] is True
    names = {p["name"] for p in result["accounting"]["parts"]}
    assert "toy_mean_residual_codes" in names
    assert "toy_mean_means" in names
    assert "mean_residual_decoder_stub" in names

    diff = subprocess.check_output(
        ["git", "diff", "--name-only", "--", *CORE],
        cwd=REPO,
        text=True,
    )
    toy_rel = spec.source_path
    assert toy_rel not in CORE
    assert Path(toy_rel).name not in {Path(c).name for c in CORE}
    print("git diff --name-only -- core modules:\n" + (diff or "(empty)"))


def test_toy_mean_residual_execute_matches_reconstruct():
    from tools.odyssey.families import toy_mean_residual as toy

    rng = np.random.RandomState(11)
    W = rng.randn(4, 8).astype(np.float32)
    payload = toy.pack(W)
    x = rng.randn(8).astype(np.float32)
    y_ir = toy.execute(payload, x)
    y_direct = toy.reconstruct(payload) @ x
    assert np.allclose(y_ir, y_direct, rtol=0.0, atol=1e-5)
    # CALL SITE of pack/execute/reconstruct, not an import.
    assert callable(toy.pack) and callable(toy.execute) and callable(toy.reconstruct)


def test_toy_family_execute_matches_reconstruct():
    from tools.odyssey.families import toy_xor_codes as toy

    rng = np.random.RandomState(7)
    W = rng.randn(4, 8).astype(np.float32)
    payload = toy.pack(W)
    x = rng.randn(8).astype(np.float32)
    y_ir = toy.execute(payload, x)
    y_direct = toy.reconstruct(payload) @ x
    assert np.allclose(y_ir, y_direct, rtol=0.0, atol=1e-5)
