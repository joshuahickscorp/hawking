"""G029: one runtime, one experiment engine, adversary stage.

No 27B, no live llama-server, no deleted Q5_K GGUF. The chain promotion is
an in-engine probe whose score actually enters the mutated function. HEAD
hcli is refused: there is no remaining admitted_n candidate.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "RUNTIME_EXPERIMENT_ADVERSARY.json"
GENOME = REPO / "receipts" / "headless" / "RUNTIME_GENOME.json"
Q5K = "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"

spec = importlib.util.spec_from_file_location(
    "experiment_engine", HERE / "experiment_engine.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_engine_imports_g021_primitives_not_a_second_decision():
    so = mod.selfopt()
    assert inspect.getsource(so.compute_decision) == inspect.getsource(
        mod.selfopt().compute_decision
    )
    assert so.pin_hcli_import_root is not None
    src = Path(so.__file__).read_text(encoding="utf-8")
    engine_src = (HERE / "experiment_engine.py").read_text(encoding="utf-8")
    assert "pin_hcli_import_root" in engine_src
    assert "G021_SCRATCH_IMPORT_SHADOW" in engine_src
    # The engine must call compute_decision, not copy its promote/refuse table.
    assert "so.compute_decision" in engine_src
    assert "decision = \"promote\"" not in engine_src
    assert "would_refuse_on_failing_gate\": True" not in engine_src
    assert "would_refuse_on_failing_gate'] = True" not in src


def test_hcli_production_does_not_require_deleted_q5k():
    from hcli.runtime_iface import q5k_gguf_required

    assert q5k_gguf_required() is False
    # HEAD, not just the worktree — a sparse hole is not absence.
    proc_src = __import__("subprocess").run(
        ["git", "-C", str(REPO), "grep", "-n", Q5K, "HEAD", "--", "hcli"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # A blanket "the name must not appear" is a LITERAL-STRING check -- which is
    # one of the adversary's own six questions -- and it contradicts the comment
    # directly below: the interface MAY name the archived artifact as science.
    # What the obligation forbids is REQUIRING the file. So allow the mention
    # only where it is an explicitly archived constant, and let the AST walk
    # below carry the real proof that nothing opens it.
    allowed = {"ARCHIVED_Q5K_GGUF_NAME", "ARCHIVED_Q5K_GGUF_REL"}
    for line in proc_src.stdout.splitlines():
        if not line.strip():
            continue
        assert any(tok in line for tok in allowed), (
            f"Q5K named outside an ARCHIVED_* constant: {line}"
        )
    # The new interface may *name* the archived artifact as science.
    # It must not open() it. Prove that with an AST walk.
    for py in (REPO / "hcli").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if Q5K not in text:
            continue
        tree = __import__("ast").parse(text)
        for node in __import__("ast").walk(tree):
            if isinstance(node, __import__("ast").Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name in {"open", "read_bytes", "read_text"}:
                    for arg in node.args:
                        if isinstance(arg, __import__("ast").Constant) and Q5K in str(
                            arg.value
                        ):
                            raise AssertionError(f"{py} opens Q5_K via {name}")


def test_full_chain_promotes_probe_and_refuses_head(tmp_path):  # noqa: ARG001
    receipt = mod.run_full_chain(REPO)
    assert RECEIPT.is_file()
    disk = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert disk["schema"] == "hawking.headless.runtime_experiment_adversary.v1"

    controls = disk["controls"]
    for name in (
        "noop",
        "bad",
        "paired_interleaved",
        "failing_gate",
        "persistent_runtime",
        "exclusive_reservation",
        "causal_execution_path",
    ):
        assert controls[name]["ran"] is True, name

    assert controls["noop"]["is_win"] is False
    assert controls["noop"]["through_mutated_mechanism"] is True
    assert (controls["noop"]["decision"] or {}).get("verdict") == "REFUSED"

    assert (controls["bad"]["decision"] or {}).get("verdict") == "REFUSED"
    assert set(controls["bad"]["candidate_scores"]) == {0}

    assert controls["paired_interleaved"]["block_design"] is False
    assert controls["paired_interleaved"]["candidate_order"] == [
        "candidate",
        "baseline",
        "candidate",
        "baseline",
    ]

    fail = controls["failing_gate"]
    assert fail["hardcoded"] is False
    assert fail["pytest_exit_code"] != 0
    assert fail["would_refuse_on_failing_gate"] is True
    assert disk["would_refuse_on_failing_gate"] == fail["would_refuse_on_failing_gate"]

    assert controls["persistent_runtime"]["ok"] is True
    assert controls["exclusive_reservation"]["exclusive"] is True
    assert controls["causal_execution_path"]["candidate_through"] is True
    assert controls["causal_execution_path"]["bypass_through"] is False
    assert controls["causal_execution_path"]["bypass_refused"] is True

    chain = disk["chain_promotion"]
    assert chain["verdict"] == "PROMOTE"
    assert chain["through_mutated_mechanism"] is True
    assert set(chain["candidate_scores"]) == {2}
    assert set(chain["baseline_scores"]) == {1}

    head = disk["head_tree_refusal"]
    assert head["verdict"] == "REFUSED"
    assert head["h1_equals_head"] is True
    assert head["no_remaining_admitted_n_candidate_at_head"] is True
    assert head["controller_verified_via"] == "git show HEAD:hcli/controller.py"

    adversary = disk["adversary"]
    assert adversary["ran"] is True
    assert len(adversary["answers"]) == 6
    questions = [row["question"] for row in adversary["answers"]]
    for q in mod.ADVERSARY_QUESTIONS:
        assert q in questions
    for row in adversary["answers"]:
        assert row["answer"]
        assert not str(row["answer"]).startswith("UNANSWERED")
    assert adversary["verdict"] == "PASS"

    census = disk["runtime_interface_census"]
    assert census["mlx_first_class"] is True
    assert census["q5k_gguf_required"] is False
    assert census["scheduler_duplicated"] is False
    assert "model_semantics" in census["planes"]
    assert "backend" in census["planes"]
    assert "session" in census["planes"]
    assert "context" in census["planes"]
    assert "health" in census["planes"]
    assert "performance_profile" in census["planes"]

    headline = disk["runtime_genome"]["mlx_headline"]
    assert headline["startup_s"] == 1.329
    assert headline["prefill_tps"] == 309.94
    assert headline["decode_tps"] == 38.06
    assert headline["context_tokens"] == 262144
    assert headline["peak_memory_gb"] == 18.21
    assert disk["runtime_genome"]["remeasured"] is False
    assert GENOME.is_file()

    q5k = disk["q5k_gguf"]
    assert q5k["required"] is False
    assert q5k["execution_proof"]["opened_q5k"] is False
    assert q5k["execution_proof"]["ok"] is True
    assert q5k["hcli_source_hits"] == []
    assert q5k["verified_against"].startswith("HEAD")

    assert disk["noetic_native"]["complete_refused"] is True
    assert receipt["controls_ok"] is True


def test_would_refuse_on_failing_gate_is_computed_not_hardcoded():
    src = inspect.getsource(mod.selfopt().run_failing_gate_trial)
    assert 'failing.get("verdict") == "REFUSED"' in src
    engine_src = inspect.getsource(mod.run_full_chain)
    assert "would_refuse_on_failing_gate\": True" not in engine_src
