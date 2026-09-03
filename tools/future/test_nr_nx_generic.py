"""Negative controls for the generic NR→NX pipeline.

A source-tree runtime read, a renamed source pointer, a skipped stage, a
hardcoded specimen-name gate in adapt(), or a physical_ebpw value would be
the campaign repeating a pass it did not earn. These tests make each of
those refusals fire. An absent specimen is a recorded refusal, never
pytest.skip.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import flash_nx_audit as nx_audit
from tools.future import nr_nx_generic as nng
from tools.future import specimen_verify as sv
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _receipt() -> dict:
    path = nng.build()
    return json.loads(path.read_text())


def _dense_names(n_layers: int = 4) -> list[str]:
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for L in range(n_layers):
        names.extend(
            [
                f"model.layers.{L}.input_layernorm.weight",
                f"model.layers.{L}.mlp.gate_proj.weight",
                f"model.layers.{L}.mlp.up_proj.weight",
                f"model.layers.{L}.mlp.down_proj.weight",
                f"model.layers.{L}.self_attn.q_proj.weight",
                f"model.layers.{L}.self_attn.k_proj.weight",
                f"model.layers.{L}.self_attn.v_proj.weight",
                f"model.layers.{L}.self_attn.o_proj.weight",
            ]
        )
    return names


def _moe_names(n_layers: int = 2, n_experts: int = 4) -> list[str]:
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for L in range(n_layers):
        names.append(f"model.layers.{L}.mlp.gate.weight")
        names.extend(
            [
                f"model.layers.{L}.self_attn.q_proj.weight",
                f"model.layers.{L}.self_attn.k_proj.weight",
                f"model.layers.{L}.self_attn.v_proj.weight",
                f"model.layers.{L}.self_attn.o_proj.weight",
            ]
        )
        for e in range(n_experts):
            names.extend(
                [
                    f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight",
                    f"model.layers.{L}.mlp.experts.{e}.up_proj.weight",
                    f"model.layers.{L}.mlp.experts.{e}.down_proj.weight",
                ]
            )
    return names


def _dense_cfg() -> dict:
    return {
        "architectures": ["WhateverForCausalLM"],
        "model_type": "whatever",
        "num_hidden_layers": 4,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
    }


def _probe_of(sid: str, names: list[str], cfg: dict) -> dict:
    return {
        "ok": True,
        "id": sid,
        "repo": sid.split("@")[0].replace("--", "/", 1),
        "revision": sid.split("@")[-1] if "@" in sid else "",
        "config": cfg,
        "tensor_names": names,
        "names_via": "caller",
        "specimen_path": f"/tmp/{sid}",
        "shard_map": {n: "model.safetensors" for n in names},
    }


def test_build_emits_sealed_static_receipt():
    out = nng.build()
    assert out.name == "NR_NX_GENERIC.json"
    assert out.parent == RECEIPTS
    doc = json.loads(out.read_text())
    assert doc["schema"] == nng.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["physical_ebpw"] is None
    assert doc["physical_ebpw_written"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == "receipts/future/NR_NX_GENERIC.json"
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_EXECUTION.complete-token"
    assert "probe_specimen" in doc["compiler_entry_points"]
    assert "adapt" in doc["compiler_entry_points"]
    assert "callable_on" in doc["compiler_entry_points"]
    assert "run" in doc["compiler_entry_points"]


def test_generic_and_flash_facts_are_separate_and_not_merged():
    doc = _receipt()
    assert "GENERIC_NR_NX_PIPELINE_CALLABLE" in doc
    assert "FLASH_NX_READY" in doc
    assert doc["facts_are_independent"] is True
    assert doc["FLASH_NX_READY"] is False
    assert doc["flash"]["FLASH_NX_READY"] is False
    # generic True does not make FLASH True; they are independent facts.
    if doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is True:
        assert doc.get("first_nx_lower_failure") is None
        nx_stage = next(s for s in doc["stages"] if s["stage"] == "NoeticExecutable")
        assert nx_stage["status"] == nng.PASSED


def test_stages_are_complete_never_skipped():
    doc = _receipt()
    names = [s["stage"] for s in doc["stages"]]
    assert names == list(nng.STAGE_ORDER)
    for row in doc["stages"]:
        assert row["status"] not in nng.FORBIDDEN_STAGE_STATUS
        assert row["status"] != "SKIPPED"
        assert row["status"] in {nng.PASSED, nng.FAILED, nng.REFUSED, nng.BLOCKED}
        assert row["why"]
        assert "invoked" in row


def test_pipeline_callable_is_false_because_a_stage_did_not_pass():
    doc = _receipt()
    live = nng.generic_pipeline_callable(doc["stages"])
    assert doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is live
    failed = [s for s in doc["stages"] if s["status"] != nng.PASSED]
    if live:
        assert not failed
        assert doc["first_failing_stage"] is None
    else:
        assert failed, "callable is False; at least one stage must not have passed"
        first = doc["first_failing_stage"]
        assert first is not None
        assert first["stage"] == failed[0]["stage"]


def test_skipped_stage_cannot_be_declared_callable():
    """NEGATIVE CONTROL: a SKIPPED stage is a constructor error, not a pass."""
    with pytest.raises(nng.StageSkipForbidden):
        nng._stage("Verifier", "SKIPPED", why="nope", invoked=False)
    fake = [
        nng._stage(name, nng.PASSED, why="synthetic all-pass", invoked=True)
        for name in nng.STAGE_ORDER
    ]
    fake[0] = dict(fake[0])
    fake[0]["status"] = "SKIPPED"
    assert nng.generic_pipeline_callable(fake) is False
    with pytest.raises(nng.StageSkipForbidden):
        nng.declare_pipeline_callable(fake, packed_nx_path=None)


def test_all_passed_without_packed_nx_still_refuses_callable(tmp_path):
    """NEGATIVE CONTROL: passing stages with no NX body is not a pipeline."""
    fake = [
        nng._stage(name, nng.PASSED, why="synthetic", invoked=True)
        for name in nng.STAGE_ORDER
    ]
    assert nng.generic_pipeline_callable(fake) is True
    with pytest.raises(nng.PipelineCallableForbidden):
        nng.declare_pipeline_callable(fake, packed_nx_path=None)
    with pytest.raises(nng.PipelineCallableForbidden):
        nng.declare_pipeline_callable(fake, packed_nx_path=tmp_path / "missing.nx")
    nx_path = tmp_path / "packed.nx.json"
    nx_path.write_text("{}")
    assert nng.declare_pipeline_callable(fake, packed_nx_path=nx_path) is True


def test_source_independence_fails_on_runtime_read_into_source_tree():
    """NEGATIVE CONTROL: any runtime read into the specimen must fail."""
    tree = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"
    nx = {
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "source_independent": True,
        "serialized_artifact": {
            "path": "/tmp/packed.nxbin",
            "sha256": "a" * 64,
            "status": "BUILT",
            "self_contained": True,
        },
        "physical_loader": {"status": "BUILT", "source_independent": True},
        "runtime_reads": [f"{tree}/model.safetensors"],
    }
    judged = nng.source_independence(nx, source_trees=[tree])
    assert judged["ok"] is False
    assert any("runtime_reads" in h for h in judged["hits"])
    assert "source tree" in judged["why"]


def test_source_independence_fails_on_renamed_source_pointer():
    """NEGATIVE CONTROL: labeling the parent checkpoint as the NX body is a pointer."""
    tree = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"
    nx = {
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "source_independent": True,
        "serialized_artifact": {
            "path": f"{tree}/model.safetensors",
            "sha256": "b" * 64,
            "status": "BUILT",
            "self_contained": True,
        },
        "physical_loader": {"status": "BUILT", "source_independent": True},
    }
    judged = nng.source_independence(nx, source_trees=[tree])
    assert judged["ok"] is False
    assert any("source tree" in h or "renamed source pointer" in h for h in judged["hits"])


def test_source_independence_can_pass_on_a_self_contained_body():
    """Inverse: the checker must still be able to return ok=True."""
    nx = nx_audit.synthetic_promotable_nx()
    judged = nng.source_independence(
        nx,
        source_trees=["/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"],
    )
    assert judged["ok"] is True
    assert judged["hits"] == []


def test_source_independence_fails_without_an_nx():
    judged = nng.source_independence(None, source_trees=[])
    assert judged["ok"] is False
    assert "no NX" in judged["why"]


def test_flash_v0_fails_source_independence_when_present():
    """The live Flash metadata seal is a real negative, not a synthetic one."""
    path = nx_audit.evidence_path(nx_audit.REL_NX_V0)
    if path is None:
        judged = nng.source_independence(
            {"status": nx_audit.METADATA_ONLY, "source_independent": False},
            source_trees=[],
        )
        assert judged["ok"] is False
        assert "metadata" in judged["why"]
        return
    nx = json.loads(path.read_text())
    judged = nng.source_independence(nx, source_trees=[])
    assert judged["ok"] is False
    assert nx_audit._status_is_metadata_only(nx)


def test_no_code_path_writes_a_physical_ebpw_value():
    """NEGATIVE CONTROL: record_physical_ebpw always raises."""
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.record_physical_ebpw(0.887)
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.record_physical_ebpw(16.0)
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.record_physical_ebpw(None)
    doc = _receipt()
    nng.assert_no_physical_ebpw(doc)
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.assert_no_physical_ebpw({"physical_ebpw": 0.5})
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.assert_no_physical_ebpw({"nested": {"qualified_complete_physical_ebpw": 1.0}})
    src = Path(nng.__file__).read_text()
    tree = ast.parse(src)
    assigned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in nnp_keys():
                    assigned.append(t.id)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in nnp_keys():
                assigned.append(node.target.id)
    assert assigned == []


def nnp_keys():
    return nng.nnp.PHYSICAL_EBPW_KEYS


def test_receipt_refuses_hardware_fields():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "NR_NX_GENERIC_SHOULD_NOT_EXIST.json",
            {"schema": "x", "tps": 1.0},
            "tools/future/test_nr_nx_generic.py",
        )
    for field in HARDWARE_FIELDS:
        assert field not in {"schema", "version", "purpose"}


def test_choose_specimen_refuses_when_nothing_is_verified():
    """NEGATIVE CONTROL: the chooser can return the negative."""
    row = nng.choose_specimen(present=set(), verified={}, lake_mounted=False)
    assert row["ok"] is False
    assert row["id"] is None
    assert "Refusing to invent a specimen" in row["why"]


def test_choose_specimen_records_why_a_later_candidate_was_not_silent():
    """Falcon is tried through adapt(), not swapped in without a recorded reason."""
    verified = {
        nng.FALCON_ID: {
            "status": "WHOLE_TREE_VERIFIED",
            "whole_tree_verified": True,
            "bytes_hashed": 15,
        },
    }
    probes = {
        nng.FALCON_ID: _probe_of(
            nng.FALCON_ID,
            [
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.layers.0.feed_forward.gate_proj.weight",
                "model.layers.0.feed_forward.up_proj.weight",
                "model.layers.0.feed_forward.down_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mamba.out_proj.weight",
                "model.layers.1.feed_forward.gate_proj.weight",
                "model.layers.1.feed_forward.up_proj.weight",
                "model.layers.1.feed_forward.down_proj.weight",
                "model.layers.1.self_attn.q_proj.weight",
                "model.layers.1.mamba.out_proj.weight",
            ],
            {
                "architectures": ["FalconH1ForCausalLM"],
                "model_type": "falcon_h1",
                "num_hidden_layers": 2,
                "hidden_act": "silu",
            },
        ),
    }
    row = nng.choose_specimen(
        present={nng.FALCON_ID},
        verified=verified,
        lake_mounted=True,
        probes=probes,
    )
    attempts = {a["id"]: a for a in row["attempts"]}
    assert nng.QWEN06_ID in attempts
    assert attempts[nng.QWEN06_ID]["not_chosen_because"]
    assert "not present" in attempts[nng.QWEN06_ID]["not_chosen_because"]
    if row["ok"]:
        assert row["id"] == nng.FALCON_ID
        assert row["adaptation"]["mlp_kind"] == "dense"
        assert "feed_forward" in (row["adaptation"]["templates"]["gate"] or "")


def test_choose_specimen_selects_cheapest_adaptable_when_verified():
    verified = {
        nng.QWEN06_ID: {
            "status": "WHOLE_TREE_VERIFIED",
            "whole_tree_verified": True,
            "bytes_hashed": 1519209243,
            "n_files": 10,
            "specimen_path": "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0",
        },
        nng.FALCON_ID: {"status": "WHOLE_TREE_VERIFIED", "whole_tree_verified": True},
        nng.QWEN30_ID: {"status": "WHOLE_TREE_VERIFIED", "whole_tree_verified": True},
    }
    probes = {
        nng.QWEN06_ID: _probe_of(nng.QWEN06_ID, _dense_names(), _dense_cfg()),
        nng.FALCON_ID: _probe_of(
            nng.FALCON_ID,
            [
                "model.layers.0.feed_forward.gate_proj.weight",
                "model.layers.0.feed_forward.up_proj.weight",
                "model.layers.0.feed_forward.down_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
            {"architectures": ["FalconH1ForCausalLM"], "model_type": "falcon_h1",
             "num_hidden_layers": 1, "hidden_act": "silu"},
        ),
        nng.QWEN30_ID: _probe_of(
            nng.QWEN30_ID,
            _moe_names(),
            {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
             "num_hidden_layers": 2, "hidden_act": "silu", "num_experts": 4,
             "num_experts_per_tok": 2},
        ),
    }
    row = nng.choose_specimen(
        present={nng.QWEN06_ID, nng.FALCON_ID, nng.QWEN30_ID},
        verified=verified,
        lake_mounted=True,
        probes=probes,
    )
    assert row["ok"] is True
    assert row["id"] == nng.QWEN06_ID
    assert row["adaptation"]["mlp_kind"] == "dense"
    assert row["adaptation"]["pipeline_can_start"] is True


def test_native_engine_parser_can_accept_and_reject():
    """NEGATIVE CONTROL: the allowlist parser returns both a hit and a miss."""
    native = nng.native_engine_architectures()
    if not native.get("ok"):
        stub = '''
        match arch.as_str() {
            "qwen2" | "qwen2.5" | "qwen" => {}
            "qwen2moe" | "qwen3moe" | "qwen-moe" => {}
            other => Err(Error::Model(format!("unknown architecture")))
        }
        '''
        native = nng.native_engine_architectures(stub)
    assert native["ok"] is True
    assert native["includes_qwen2"] is True
    assert native["includes_qwen3moe"] is True
    assert native["includes_qwen3_dense"] is False
    assert native["includes_falcon_h1"] is False
    assert "qwen3" not in (native.get("architectures") or [])


def test_physical_graph_stage_parameterized_or_named_failure():
    doc = _receipt()
    choice = doc["specimen"]
    pgc = next(s for s in doc["stages"] if s["stage"] == "PhysicalGraphCompiler")
    path = Path(str(choice.get("specimen_path") or ""))
    weights_present = path.is_dir() and (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
    )
    if choice.get("ok") and weights_present:
        assert pgc["invoked"] is True
        ev = pgc["evidence"] or {}
        plan = ev.get("collapse_plan") or []
        if plan and pgc["status"] == nng.PASSED:
            collapse = ev["parameterized_collapse"]
            assert collapse["numerically_equivalent"] is True
            assert collapse["max_abs_diff"] <= nng._SWIGLU_TOLERANCE
            assert "experts." not in collapse["gate"]
            assert ev.get("compiler_main_returncode") != 0 or ev.get("compiler_main_invoked") is False
        else:
            assert pgc["status"] == nng.FAILED
            assert pgc["error"]
    else:
        assert pgc["status"] in {nng.FAILED, nng.REFUSED}
        assert pgc["status"] != "SKIPPED"


def test_sleeping_unit_is_sleeping_never_pending():
    doc = _receipt()
    wu = doc["sleeping_workunit"]
    assert wu["status"] == "sleeping"
    assert wu["classification"] == "SLEEPING"
    assert wu["status"] not in {"pending", "PENDING", "ready", "READY"}
    assert wu["synthetic_result_forbidden"] is True
    assert wu["wake_unmet"]
    assert any(not w["holds"] for w in wu["wake_conditions"])


def test_check_nx_verifier_was_invoked():
    doc = _receipt()
    ver = next(s for s in doc["stages"] if s["stage"] == "Verifier")
    assert ver["invoked"] is True
    assert ver["evidence"]["promotable"] is False
    nx_stage = next(s for s in doc["stages"] if s["stage"] == "NoeticExecutable")
    if nx_stage["status"] == nng.PASSED:
        assert ver["status"] == nng.PASSED
        assert ver["evidence"].get("packer_owned_ok") is True
    else:
        assert ver["status"] != nng.PASSED


def test_architecture_recognizer_ran_or_refused_without_skipping():
    doc = _receipt()
    row = next(s for s in doc["stages"] if s["stage"] == "ArchitectureRecognizer")
    if doc["specimen"].get("ok") and Path(str(doc["specimen"].get("specimen_path") or "")).is_dir():
        assert row["status"] == nng.PASSED
        assert row["invoked"] is True
        assert row["evidence"]["loaded_weights"] is False
        organs = [o["organ"] for o in row["evidence"]["organs"]]
        assert "mlp_gate_up" in organs
        assert "gqa_attention" in organs
        assert row["evidence"]["n_unmatched"] == 0
        assert row["evidence"]["did_not_fetch_network"] is True
    else:
        assert row["status"] == nng.REFUSED
        assert row["status"] != "SKIPPED"


def test_probe_specimen_absent_fails_closed():
    """NEGATIVE CONTROL: a missing body is a refusal, never assumed names."""
    row = nng.probe_specimen("definitely-not-a-specimen-zzzz")
    assert row["ok"] is False
    assert row["tensor_names"] == []
    assert "refus" in row["why"].lower() or "not" in row["why"].lower() or "disk" in row["why"].lower()


def test_probe_specimen_reads_real_header_when_present():
    """Cope either way: lake may be unmounted; synthetic still proves the reader."""
    lake = Path("/Volumes/corpdrive/hawking-modellake/specimens") / nng.QWEN06_ID
    if lake.is_dir() and (lake / "model.safetensors").is_file():
        row = nng.probe_specimen(nng.QWEN06_ID)
        assert row["ok"] is True
        assert row["names_via"] in {"model.safetensors header", "model.safetensors.index.json"}
        names = set(row["tensor_names"])
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in names
        assert row["n_tensors"] == len(row["tensor_names"]) > 0
        assert row["config"]["model_type"] == "qwen3"
        return
    supplied = nng.probe_specimen(_probe_of("synthetic-dense", _dense_names(), _dense_cfg()))
    assert supplied["ok"] is True
    assert supplied["names_via"] == "caller"
    assert "model.layers.0.mlp.gate_proj.weight" in supplied["tensor_names"]


def test_adapt_is_name_agnostic():
    """Same config+index, two ids → same mapping. The id is not the architecture."""
    cfg = _dense_cfg()
    names = _dense_names()
    a = nng.adapt(_probe_of("alpha-not-a-real-model", names, cfg))
    b = nng.adapt(_probe_of("beta-also-not-real", names, cfg))
    assert a["ok"] is True and b["ok"] is True
    assert a["mlp_kind"] == b["mlp_kind"] == "dense"
    assert a["templates"] == b["templates"]
    assert a["collapse_plan"] == b["collapse_plan"]
    assert a["family"] == b["family"]
    assert a["pipeline_can_start"] is True


def test_adapt_derives_moe_vs_dense_from_index():
    dense = nng.adapt(_probe_of("synth-dense", _dense_names(), _dense_cfg()))
    moe = nng.adapt(
        _probe_of(
            "synth-moe",
            _moe_names(),
            {
                "architectures": ["WhateverMoeForCausalLM"],
                "model_type": "whatever_moe",
                "num_hidden_layers": 2,
                "hidden_act": "silu",
                "num_experts": 4,
                "num_experts_per_tok": 2,
            },
        )
    )
    assert dense["mlp_kind"] == "dense"
    assert dense["templates"]["gate"] and "{E}" not in dense["templates"]["gate"]
    assert moe["mlp_kind"] == "moe"
    assert moe["templates"]["gate"] and "{E}" in moe["templates"]["gate"]
    assert moe["router_collapse"]["applicable"] is True
    assert dense["router_collapse"]["applicable"] is False


def test_adapt_rejects_tensors_that_do_not_fit():
    """NEGATIVE CONTROL: embed-only names cannot start the pipeline."""
    ad = nng.adapt(
        _probe_of(
            "synth-embed-only",
            ["model.embed_tokens.weight", "lm_head.weight", "something.unrelated.weight"],
            {"architectures": ["WhateverForCausalLM"], "model_type": "whatever", "num_hidden_layers": 1},
        )
    )
    assert ad["pipeline_can_start"] is False
    assert ad["collapse_plan"] == []
    assert ad["probe_plan"] == []
    assert ad["missing"]


def test_adapt_refuses_nonsilu_fusion():
    """NEGATIVE CONTROL: gelu is not silently treated as SwiGLU."""
    cfg = _dense_cfg()
    cfg["hidden_act"] = "gelu"
    ad = nng.adapt(_probe_of("synth-gelu", _dense_names(), cfg))
    assert ad["swiglu"] is False
    assert ad["pipeline_can_start"] is False
    assert any("silu" in m.lower() or "swiglu" in m.lower() or "activation" in m.lower() for m in ad["missing"])


def test_adapt_path_has_no_hardcoded_specimen_name_gate():
    """A branch on Qwen3-0.6B would reproduce the original defect."""
    src = Path(nng.__file__).read_text()
    tree = ast.parse(src)
    fn_names = {
        "adapt",
        "probe_specimen",
        "_templates_from_names",
        "_pick_template",
        "_format_tensor",
        "_present_layers",
        "_n_layers",
        "_hidden_act",
        "_probe_dir",
    }
    chunks: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in fn_names:
            chunks.append(ast.get_source_segment(src, node) or "")
    blob = "\n".join(chunks)
    for tok in (
        nng.QWEN06_ID,
        nng.FALCON_ID,
        nng.QWEN30_ID,
        "Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "Falcon-H1",
        "c1899de289a0",
        "Qwen3-30B-A3B",
    ):
        assert tok not in blob, f"adaptation path hardcodes {tok!r}"


def test_callable_on_names_failing_stage():
    """Full path cannot run without a packed NX; the stage is named."""
    judged = nng.callable_on(_probe_of("synth-dense", _dense_names(), _dense_cfg()))
    assert judged["ok"] is False
    assert judged["first_failing_stage"] in nng.STAGE_ORDER
    assert judged["missing_input"]
    preview = {r["stage"]: r for r in judged["stage_preview"]}
    assert preview[judged["first_failing_stage"]]["ready"] is False
    assert preview["SourceIndependence"]["ready"] is False
    assert "packed NX" in (preview["SourceIndependence"]["missing_input"] or "").lower() or "nx" in (
        preview["NoeticExecutable"]["missing_input"] or ""
    ).lower()


def test_callable_on_names_stage_when_tensors_do_not_fit():
    """NEGATIVE CONTROL: a specimen whose tensors do not fit fails with the STAGE named."""
    judged = nng.callable_on(
        _probe_of(
            "synth-unfit",
            ["model.embed_tokens.weight", "lm_head.weight"],
            {"architectures": ["WhateverForCausalLM"], "model_type": "whatever", "num_hidden_layers": 1, "hidden_act": "silu"},
        )
    )
    assert judged["ok"] is False
    assert judged["first_failing_stage"]
    assert judged["first_failing_stage"] != "SKIPPED"
    assert judged["missing_input"]
    assert judged["first_failing_stage"] in {"Doctor", "PhysicalGraphCompiler", "SpecimenPresent"}


def test_run_does_not_mint_an_nx():
    result = nng.run(_probe_of("synth-dense", _dense_names(), _dense_cfg()))
    assert result["packed_nx"] is None
    assert result["packed_path"] is None
    assert result["callable_ok"] is False
    src = next(s for s in result["stages"] if s["stage"] == "SourceIndependence")
    assert src["status"] == nng.FAILED
    nx_stage = next(s for s in result["stages"] if s["stage"] == "NoeticExecutable")
    assert nx_stage["evidence"]["did_not_mint_nx"] is True


def test_doctor_uses_adapted_tensors_not_compiler_parent():
    doc = _receipt()
    row = next(s for s in doc["stages"] if s["stage"] == "Doctor")
    path = Path(str(doc["specimen"].get("specimen_path") or ""))
    if doc["specimen"].get("ok") and path.is_dir():
        ev = row["evidence"] or {}
        assert ev.get("did_not_call_probes") is True
        adapted = ev.get("adapted_probe_tensors") or []
        overlap = (ev.get("compiler_hardcoded_overlap") or {})
        if row["status"] == nng.PASSED:
            assert adapted
            assert ev.get("loaded_weights") is True
            assert overlap.get("doctor_probes_absent")
            for name in adapted:
                assert "language_model" not in name
        else:
            assert row["status"] in {nng.FAILED, nng.REFUSED, nng.BLOCKED}
            assert row["error"] or row["why"]
    else:
        assert row["status"] in {nng.FAILED, nng.REFUSED, nng.BLOCKED}
        assert row["status"] != "SKIPPED"


def test_run_on_unfit_specimen_names_the_stage():
    """NEGATIVE CONTROL: unfit tensors fail a named stage, never skip."""
    result = nng.run(
        _probe_of(
            "synth-unfit",
            ["model.embed_tokens.weight"],
            {"architectures": ["WhateverForCausalLM"], "model_type": "x", "num_hidden_layers": 1, "hidden_act": "silu"},
        )
    )
    first = result["first_failing_stage"]
    assert first is not None
    assert first["stage"] in nng.STAGE_ORDER
    assert first["status"] != "SKIPPED"
    for row in result["stages"]:
        assert row["status"] != "SKIPPED"


def test_callable_on_and_receipt_agree_on_unmet_nx():
    doc = _receipt()
    live = nng.generic_pipeline_callable(doc["stages"])
    assert doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is live
    if live:
        assert doc["FLASH_NX_READY"] is False
        nx_stage = next(s for s in doc["stages"] if s["stage"] == "NoeticExecutable")
        assert nx_stage["status"] == nng.PASSED
        assert nx_stage["evidence"]["did_not_execute_first_noetic_executable"] is True
        packed = Path(str((nx_stage.get("evidence") or {}).get("packed_path") or ""))
        assert packed.is_file()
    else:
        assert doc["first_failing_stage"] is not None


def _name_only_kernel(organ: str = "mlp_down") -> dict:
    return {
        "kernel_identity": f"name_only_{organ}",
        "organ_identity": organ,
        "representation_identity": "q2_affine",
        "compiled_identity": {
            "kind": "ABSENT",
            "value": None,
            "absent_reason": "synthetic name-only kernel; not compiled",
        },
        "specialization": {"kind": "DERIVED", "group_size": 64},
    }


def _compiled_shape_kernel(organ: str = "mlp_down", cols: int = 64, *, parametric: bool = False) -> dict:
    spec: dict = {"kind": "DERIVED", "specialized_cols": cols}
    if parametric:
        spec["shape_constraints"] = {"cols": [cols]}
    return {
        "kernel_identity": f"compiled_{organ}_{cols}",
        "organ_identity": organ,
        "representation_identity": "q2_affine",
        "compiled_identity": {"kind": "MEASURED", "value": "deadbeef" * 4},
        "specialization": spec,
    }


def test_shared_organ_name_is_not_a_compiled_kernel():
    """NEGATIVE CONTROL: a shared organ ROLE is not a compiled kernel for this body."""
    kernel = _name_only_kernel("mlp_down")
    shapes = nng.organ_shapes_from_config(_dense_cfg())
    judged = nng.is_compiled_kernel_for_body(
        kernel,
        specimen_id="synth-dense",
        organ="mlp_down",
        organ_shape=shapes.get("mlp_down"),
    )
    assert judged["ok"] is False
    assert judged["role_match"] is True
    assert judged["specimen_id_match"] is False
    assert judged["shape_constraints_satisfied"] is False
    assert judged["compiled_identity_present"] is False
    assert nng.NAME_IS_NOT_A_COMPILED_KERNEL in judged["why"]
    planned = nng.plan_kernels_for_specimen(
        ["mlp_down", "mlp_gate_up", "gqa_attention"],
        specimen_id="synth-dense",
        config=_dense_cfg(),
        kernels=[kernel, _name_only_kernel("mlp_gate_up"), _name_only_kernel("gqa_attention")],
        library_specimen_field=None,
    )
    assert planned["n_compiled"] == 0
    assert planned["n_native_unmeasured"] == 3
    assert planned["name_is_not_a_compiled_kernel"] is True
    for slot in planned["plan"]:
        assert slot["status"] == nng.NATIVE_UNMEASURED
        assert slot["occupying"]["kind"] == nng.NATIVE_UNMEASURED
        assert slot["occupying"]["compiled_kernel"] is None
        assert slot["name_is_not_a_compiled_kernel"] is True
        assert nng.NAME_IS_NOT_A_COMPILED_KERNEL in slot["why"]


def test_undeclared_shape_is_not_a_parametric_wildcard():
    """NEGATIVE CONTROL: omitting specialized_cols does not mean 'any shape'."""
    kernel = _name_only_kernel("mlp_down")
    kernel["compiled_identity"] = {"kind": "MEASURED", "value": "abc123"}
    shapes = nng.organ_shapes_from_config(_dense_cfg())
    judged = nng.is_compiled_kernel_for_body(
        kernel,
        specimen_id="synth-dense",
        organ="mlp_down",
        organ_shape=shapes.get("mlp_down"),
    )
    assert judged["compiled_identity_present"] is True
    assert judged["shape_constraints_satisfied"] is False
    assert judged["ok"] is False
    assert nng.NAME_IS_NOT_A_COMPILED_KERNEL in judged["why"]


def test_shape_mismatch_is_not_a_compiled_kernel():
    """NEGATIVE CONTROL: parent specialized_cols 5120 is not this body's 64/128."""
    kernel = _compiled_shape_kernel("mlp_down", cols=5120)
    shapes = nng.organ_shapes_from_config(_dense_cfg())
    judged = nng.is_compiled_kernel_for_body(
        kernel,
        specimen_id="synth-dense",
        organ="mlp_down",
        organ_shape=shapes.get("mlp_down"),
    )
    assert judged["role_match"] is True
    assert judged["compiled_identity_present"] is True
    assert judged["shape_constraints_satisfied"] is False
    assert judged["ok"] is False
    assert nng.NAME_IS_NOT_A_COMPILED_KERNEL in judged["why"]


def test_compiled_kernel_requires_identity_or_shapes_and_present_compiled_identity():
    """Inverse: declared cols that this organ satisfies plus a present compiled_identity."""
    kernel = _compiled_shape_kernel("mlp_down", cols=64)
    shapes = nng.organ_shapes_from_config(_dense_cfg())
    judged = nng.is_compiled_kernel_for_body(
        kernel,
        specimen_id="synth-dense",
        organ="mlp_down",
        organ_shape=shapes.get("mlp_down"),
    )
    assert judged["ok"] is True
    assert judged["shape_constraints_satisfied"] is True
    assert judged["compiled_identity_present"] is True
    assert nng.NAME_IS_NOT_A_COMPILED_KERNEL not in judged["why"]


def test_plan_then_compile_emits_native_unmeasured_for_unseen_body():
    lib = {
        "kernels": [
            _name_only_kernel("mlp_down"),
            _name_only_kernel("mlp_gate_up"),
            _compiled_shape_kernel("mlp_down", cols=5120),
        ],
        "specimen": None,
    }
    row = nng.stage_kernel_planner(
        ["mlp_down", "mlp_gate_up", "embed"],
        specimen_id="unseen-body",
        config=_dense_cfg(),
        library_doc=lib,
    )
    assert row["status"] == nng.PASSED
    assert row["invoked"] is True
    ev = row["evidence"]
    assert ev["route"] == nng.KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE
    assert ev["name_is_not_a_compiled_kernel"] is True
    assert ev["n_compiled"] == 0
    assert ev["n_native_unmeasured"] == 3
    assert nng.NAME_IS_NOT_A_COMPILED_KERNEL in row["why"]
    organs = {slot["organ"]: slot for slot in ev["plan"]}
    assert set(organs) == {"mlp_down", "mlp_gate_up", "embed"}
    for slot in ev["plan"]:
        assert slot["status"] == nng.NATIVE_UNMEASURED


def test_shape_parametric_route_when_library_declares_constraints_and_is_compiled():
    kernel = _compiled_shape_kernel("mlp_down", cols=64, parametric=True)
    lib = {"kernels": [kernel], "specimen": None}
    row = nng.stage_kernel_planner(
        ["mlp_down"],
        specimen_id="synth-dense",
        config=_dense_cfg(),
        library_doc=lib,
    )
    assert row["status"] == nng.PASSED
    ev = row["evidence"]
    assert ev["route"] == nng.KERNEL_PLANNER_ROUTE_SHAPE_PARAMETRIC
    assert ev["n_compiled"] == 1
    assert ev["plan"][0]["status"] == nng.COMPILED
    assert ev["plan"][0]["occupying"]["compiled_kernel"] == kernel["kernel_identity"]


def test_missing_library_is_refused_not_empty_success(monkeypatch):
    """NEGATIVE CONTROL: an unreachable library is REFUSED, not an empty pass."""
    monkeypatch.setattr(nng.nx_audit, "evidence_path", lambda rel: None)
    row = nng.stage_kernel_planner(
        ["mlp_down"],
        specimen_id="x",
        config=_dense_cfg(),
    )
    assert row["status"] == nng.REFUSED
    assert row["status"] != "SKIPPED"
    assert row["error"] == "missing_kernel_library"
    assert "empty success" in row["why"]


def test_empty_organs_is_failed_not_skipped():
    row = nng.stage_kernel_planner(
        [],
        specimen_id="x",
        config=_dense_cfg(),
        library_doc={"kernels": [_name_only_kernel()]},
    )
    assert row["status"] == nng.FAILED
    assert row["status"] != "SKIPPED"
    assert row["error"] == "no_organs"


def test_live_library_forces_plan_then_compile_for_this_body():
    """Library evidence that chose the route: no specimen field, no compiled identity,
    specialized_cols are the parent 5120/17408, not this body's 1024/3072."""
    doc, _path, err = nng.load_kernel_library()
    if doc is None:
        ok, why = nng.kernel_library_is_readable()
        assert ok is False
        assert why and "empty success" in why
        return
    assert doc.get("specimen") is None
    kernels = [k for k in (doc.get("kernels") or []) if isinstance(k, dict)]
    assert kernels
    n_compiled = sum(1 for k in kernels if nng._compiled_identity_present(k))
    assert n_compiled == 0
    declared = sorted({c for k in kernels if (c := nng._declared_specialized_cols(k)) is not None})
    assert 5120 in declared
    assert 17408 in declared
    shapes = nng.organ_shapes_from_config(
        {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
    )
    route = nng.kernel_planner_route(
        kernels,
        specimen_id=nng.QWEN06_ID,
        organ_shapes=shapes,
        library_specimen_field=doc.get("specimen"),
    )
    assert route["route"] == nng.KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE
    assert route["n_compiled_identity_present"] == 0
    assert route["n_parametric_range_declared"] == 0
    assert route["shape_overlap"] == []
    assert 1024 in route["specimen_extents"]
    assert 3072 in route["specimen_extents"]
    planned = nng.plan_kernels_for_specimen(
        ["gqa_attention", "mlp_down", "mlp_gate_up"],
        specimen_id=nng.QWEN06_ID,
        config={
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        kernels=kernels,
        library_specimen_field=doc.get("specimen"),
    )
    assert planned["n_compiled"] == 0
    assert planned["name_is_not_a_compiled_kernel"] is True
    assert set(planned["intersection"]) == {"gqa_attention", "mlp_down", "mlp_gate_up"}


def test_callable_on_kernel_planner_ready_when_library_exists():
    judged = nng.callable_on(_probe_of("synth-dense", _dense_names(), _dense_cfg()))
    preview = {r["stage"]: r for r in judged["stage_preview"]}
    ok, _why = nng.kernel_library_is_readable()
    if ok:
        assert preview["KernelPlanner"]["ready"] is True
        assert preview["DeviceCompiler"]["ready"] is True
        assert judged["first_failing_stage"] == "NoeticExecutable"
    else:
        assert preview["KernelPlanner"]["ready"] is False
        assert judged["first_failing_stage"] == "KernelPlanner"


def test_receipt_kernel_planner_passed_and_is_not_first_fail():
    doc = _receipt()
    kp = next(s for s in doc["stages"] if s["stage"] == "KernelPlanner")
    if doc["specimen"].get("ok") and nng.kernel_library_is_readable()[0]:
        assert kp["status"] == nng.PASSED
        assert kp["invoked"] is True
        ev = kp["evidence"] or {}
        assert ev.get("route") == nng.KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE
        assert ev.get("name_is_not_a_compiled_kernel") is True
        assert nng.NAME_IS_NOT_A_COMPILED_KERNEL in kp["why"]
        assert ev.get("n_compiled") == 0
        assert ev.get("n_native_unmeasured") == len(ev.get("plan") or [])
        assert doc["kernel_planner_route"] == nng.KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE
        first = doc["first_failing_stage"]
        if first is not None:
            assert first["stage"] != "KernelPlanner"
        else:
            assert nng.generic_pipeline_callable(doc["stages"]) is True
        dc_row = next(s for s in doc["stages"] if s["stage"] == "DeviceCompiler")
        assert dc_row["invoked"] is True
        assert dc_row["status"] != "SKIPPED"
        assert (dc_row.get("evidence") or {}).get("kernel_plan_received") is True
        assert (dc_row.get("evidence") or {}).get("entry_point") == (
            "tools.future.device_compiler.lower_plan"
        )
        blocker = (dc_row.get("evidence") or {}).get("qwen3_dense_gguf_blocker") or {}
        if doc["adaptation"].get("model_type") == "qwen3" or doc["adaptation"].get("family") == "dense_swiglu_transformer":
            assert blocker.get("id") == "QWEN3_DENSE_GGUF_MATCH_ARM_ABSENT"
            assert blocker.get("did_not_map_dense_onto_moe_arm") is True
            assert blocker.get("includes_qwen3_dense") is False
        for slot in (dc_row.get("evidence") or {}).get("plan") or []:
            if slot.get("status") == nng.COMPILED:
                identity = slot.get("compiled_identity") or {}
                assert identity.get("kind") == "METAL_PIPELINE"
                assert identity.get("shader_hash")
                assert identity.get("entry_point")
                assert identity.get("shader_hash") != identity.get("source_sha256")
            else:
                assert slot.get("status") == nng.NATIVE_UNMEASURED
                assert slot.get("compiled_identity") is None
        if dc_row["status"] == nng.PASSED:
            assert dc_row["error"] is None
            assert (dc_row.get("evidence") or {}).get("n_compiled", 0) > 0
            if doc["first_failing_stage"] is not None:
                assert doc["first_failing_stage"]["stage"] != "DeviceCompiler"
            nx_stage = next(s for s in doc["stages"] if s["stage"] == "NoeticExecutable")
            assert (nx_stage.get("evidence") or {}).get("nx_fragment_received") is True
            assert nx_stage["evidence"]["did_not_execute_first_noetic_executable"] is True
            if nx_stage["status"] == nng.PASSED:
                assert nx_stage["error"] is None
                ident = (nx_stage.get("evidence") or {}).get("identity") or {}
                assert ident.get("n_compiled_organs", 0) > 0
                assert ident.get("did_not_hardlink") is True
                packed = Path(str((nx_stage.get("evidence") or {}).get("packed_path") or ""))
                assert packed.is_file()
                if doc["first_failing_stage"] is not None:
                    assert doc["first_failing_stage"]["stage"] != "NoeticExecutable"
            else:
                assert nx_stage["status"] in {nng.FAILED, nng.REFUSED, nng.BLOCKED}
                assert nx_stage["status"] != "SKIPPED"
                assert doc["first_failing_stage"]["stage"] == "NoeticExecutable"
        else:
            assert dc_row["status"] in {nng.FAILED, nng.REFUSED, nng.BLOCKED}
            assert doc["first_failing_stage"]["stage"] == "DeviceCompiler"
            assert (dc_row.get("evidence") or {}).get("n_compiled", 0) == 0
    else:
        assert kp["status"] in {nng.FAILED, nng.REFUSED, nng.BLOCKED}
        assert kp["status"] != "SKIPPED"


def test_device_compiler_refuses_placeholder_on_the_generic_path():
    """NEGATIVE CONTROL: a placeholder identity cannot become a COMPILED organ."""
    from tools.future import device_compiler as dcomp

    plan = {
        "route": nng.KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE,
        "plan": [
            {
                "organ": "mlp_down",
                "status": nng.NATIVE_UNMEASURED,
                "occupying": {"kind": nng.NATIVE_UNMEASURED, "compiled_kernel": None},
                "specimen_shape": {"rows": 64, "cols": 128, "extents": [64, 128]},
                "why": nng.NAME_IS_NOT_A_COMPILED_KERNEL,
            }
        ],
        "n_compiled": 0,
        "n_native_unmeasured": 1,
    }
    native = {
        "path": nng.NATIVE_LOADER,
        "architectures": ["qwen2", "qwen3moe"],
        "includes_qwen2": True,
        "includes_qwen3moe": True,
        "includes_qwen3_dense": False,
        "includes_falcon_h1": False,
    }

    class _Lie:
        def compile_jobs(self, jobs):
            from pathlib import Path

            results = []
            for job in jobs:
                Path(job.archive_path).write_text(job.source)
                results.append(
                    {
                        "id": job.organ,
                        "ok": True,
                        "entry_point": job.entry_point,
                        "function_found": True,
                        "pipeline_created": True,
                        "pipeline_object": dcomp.PIPELINE_OBJECT,
                        "archive_sha256": job.source_sha256,
                        "archive_bytes": len(job.source),
                        "archive_path": job.archive_path,
                        "source_sha256": job.source_sha256,
                    }
                )
            return {"ok": True, "results": results, "backend": "lying"}

    lowering = dcomp.lower_plan(
        plan,
        family="dense_swiglu_transformer",
        config={"hidden_size": 64, "intermediate_size": 128, "model_type": "qwen3"},
        native_architectures=native["architectures"],
        model_type="qwen3",
        backend=_Lie(),
    )
    assert lowering["n_compiled"] == 0
    assert lowering["plan"][0]["status"] == nng.NATIVE_UNMEASURED
    row = nng.stage_device_compiler(
        native,
        family="dense_swiglu_transformer",
        kernel_plan=plan,
        config={"hidden_size": 64, "intermediate_size": 128, "model_type": "qwen3"},
        specimen_id="synth",
        model_type="qwen3",
    )
    assert row["invoked"] is True
    assert row["status"] != nng.PASSED or (row.get("evidence") or {}).get("n_compiled", 0) > 0
    # Live Metal may pass this tiny plan; a pass must still carry genuine identity.
    if row["status"] == nng.PASSED:
        for slot in (row.get("evidence") or {}).get("plan") or []:
            if slot.get("status") == nng.COMPILED:
                ident = slot.get("compiled_identity") or {}
                assert ident.get("shader_hash") != ident.get("source_sha256")
                assert ident.get("kind") == "METAL_PIPELINE"
    else:
        assert row["status"] in {nng.FAILED, nng.REFUSED, nng.BLOCKED}
        assert row["status"] != "SKIPPED"


def test_device_compiler_module_is_the_authority():
    src = Path(nng.__file__).read_text()
    assert "from tools.future import device_compiler as dcomp" in src
    assert "MTLCreateSystemDefaultDevice" not in src
    assert nng.dcomp.lower_plan is not None


def test_ephemeral_scratch_dir_names_are_scrubbed_before_receipt_write():
    """assemble() compiles through two mkdtemp scratch dirs neither call site
    pins a name for (nr_nx_generic's own "hawking-nx-cap-*" and
    device_compiler's "hawking-dc-archives-*"), both rmtree'd before build()
    returns. The random suffix is not evidence - a rerun that compiled
    byte-identical archives still gets a different suffix - so two runs that
    only differ by mkdtemp's random name must scrub to the same receipt.
    """
    doc = {
        "capture_dir": "/tmp/hawking-nx-cap-ab12cd34",
        "nested": {
            "path": "/tmp/hawking-nx-cap-ab12cd34/embed.mtlarchive",
            "archive_path": "/tmp/hawking-dc-archives-99zz11xx/rmsnorm.mtlarchive",
        },
        "unrelated": "/tmp/some-other-dir/file.txt",
    }
    scrubbed_a = nng._scrub_ephemeral_scratch_paths(doc)
    doc2 = json.loads(json.dumps(doc).replace("ab12cd34", "different99").replace("99zz11xx", "alsodiff01"))
    scrubbed_b = nng._scrub_ephemeral_scratch_paths(doc2)
    assert scrubbed_a == scrubbed_b, "two different mkdtemp suffixes must scrub to the same receipt content"
    assert scrubbed_a["capture_dir"] == "/tmp/hawking-nx-cap-EPHEMERAL"
    assert scrubbed_a["nested"]["archive_path"] == "/tmp/hawking-dc-archives-EPHEMERAL/rmsnorm.mtlarchive"
    assert scrubbed_a["unrelated"] == "/tmp/some-other-dir/file.txt", "unrelated paths must not be touched"


def test_no_pytest_skip_in_this_file():
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            assert name != "skip", "pytest.skip that actually fires is a P0"


def test_doctor_energy_is_the_gram_spectrum_and_matches_the_svd_it_replaced():
    """energy is the squared singular values, so an SVD is the wrong tool.

    _doctor_stats used np.linalg.svd(mat, compute_uv=False) and then only ever
    squared the result. Singular values are the square roots of the Gram
    eigenvalues, so that is a sqrt-then-square round trip through a routine ~3x
    more expensive than the eigendecomposition that yields energy directly.

    THIS CALLS _doctor_stats. An earlier version of this test recomputed the Gram
    spectrum inline and compared it to numpy -- which passed even when the source
    was mutated to reverse the energy ordering, because it never touched the
    implementation. A test that reimplements what it checks is checking itself.
    """
    import numpy as np

    from tools.future import nr_nx_generic as m

    rng = np.random.default_rng(5)
    for shape in ((256, 320), (400, 128)):
        mat = rng.standard_normal(shape, dtype=np.float32)

        got = m._doctor_stats(mat)

        # Independent oracle: the SVD path this replaced, computed here.
        ref_energy = np.linalg.svd(mat, compute_uv=False) ** 2
        denom = float(ref_energy.sum())
        cum = np.cumsum(ref_energy) / denom
        r50 = int(np.sum(cum < 0.50)) + 1
        r90 = int(np.sum(cum < 0.90)) + 1

        assert got["rank_for_50pct_energy"] == r50, (
            f"{shape}: r50 {got['rank_for_50pct_energy']} != SVD reference {r50}"
        )
        assert got["rank_for_90pct_energy"] == r90, (
            f"{shape}: r90 {got['rank_for_90pct_energy']} != SVD reference {r90}"
        )
        assert got["full_rank"] == min(shape)
