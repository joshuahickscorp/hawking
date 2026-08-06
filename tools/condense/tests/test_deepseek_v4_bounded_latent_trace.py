"""Contracts for bounded, hash-only DeepSeek-V4 diagnostic bridge traces."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity


class _SuiteTokenizer:
    def __init__(self) -> None:
        self._ids = {
            prompt: [index + 1, index + 11]
            for index, (_category, prompt) in enumerate(
                gravity._DIAGNOSTIC_LATENT_TRACE_PROMPT_SUITE
            )
        }

    def encode(self, prompt: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        return SimpleNamespace(ids=self._ids[prompt])


def _suite_runtime() -> gravity.DeepSeekV4DiagnosticRuntime:
    runtime = object.__new__(gravity.DeepSeekV4DiagnosticRuntime)
    runtime._lock = Lock()
    runtime.topk = 6
    runtime.position = 0
    runtime.tokenizer = _SuiteTokenizer()

    def reset() -> None:
        runtime.position = 0

    def capture(token_id: int) -> dict:
        routes = [int((token_id + offset) % 256) for offset in range(6)]
        trace = {
            "router": {
                "top6": {
                    "selected_expert_ids": routes,
                    "selected_probabilities": [1 / 6] * 6,
                    "applied_route_weights": [0.25] * 6,
                }
            },
            "sampling": {"completion_text_disclosed": False},
            "raw_activations_retained": False,
        }
        runtime.position += 1
        return trace

    runtime.reset = reset
    runtime._latent_capture_forward_token = capture
    return runtime


def test_latent_suite_has_exact_disjoint_hash_only_categories() -> None:
    runtime = _suite_runtime()

    run = gravity.DeepSeekV4DiagnosticRuntime.capture_latent_route_suite(runtime)

    assert run["categories"] == [
        "coding",
        "agent planning",
        "tool use",
        "long-context retrieval",
        "mathematical reasoning",
        "repair",
        "general conversation",
    ]
    assert run["membership_partition"] == {
        "name": "diagnostic_transplant_capture_only",
        "disjoint_prompt_hashes": True,
        "disjoint_source_token_sequences": True,
        "excluded_from": ["fit", "calibration", "public_test", "hidden_test"],
    }
    assert len(run["members"]) == 7
    assert len(run["trace_shards"]) == 14
    assert {member["prompt_sha256"] for member in run["members"]}.__len__() == 7
    assert {member["source_token_ids_sha256"] for member in run["members"]}.__len__() == 7
    assert all(member["qualified_trace_shard_count"] <= 2 for member in run["members"])
    long_context = next(member for member in run["members"] if member["category"] == "long-context retrieval")
    assert long_context["diagnostic_context_limited"] is True
    assert "does not establish long-context" in long_context["category_scope"]
    rendered = json.dumps(run, sort_keys=True)
    for _category, prompt in gravity._DIAGNOSTIC_LATENT_TRACE_PROMPT_SUITE:
        assert prompt not in rendered
    assert '"completion"' not in rendered


def test_latent_forward_shard_summarizes_real_operator_boundaries_without_arrays() -> None:
    runtime = object.__new__(gravity.DeepSeekV4DiagnosticRuntime)
    runtime.hc_mult = 4
    runtime.hc_iters = 20
    runtime.norm_eps = 1e-6
    runtime.topk = 6
    runtime.position = 0
    runtime.window_kv = np.zeros((2, 2), dtype=np.float32)
    runtime.main_compressor = SimpleNamespace(
        cache=[np.asarray([1.0, 2.0], dtype=np.float32)],
        kv_state=np.zeros((2, 2), dtype=np.float32),
        score_state=np.full((2, 2), -np.inf, dtype=np.float32),
    )
    runtime.index_compressor = SimpleNamespace(
        cache=[],
        kv_state=np.zeros((2, 2), dtype=np.float32),
        score_state=np.full((2, 2), -np.inf, dtype=np.float32),
    )

    runtime._embedding = lambda token_id: np.asarray([float(token_id), 2.0], dtype=np.float32)
    runtime._vector = lambda _name: np.ones(2, dtype=np.float32)
    runtime._hc_pre = lambda _hidden, _prefix: (
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.eye(4, dtype=np.float32),
    )
    runtime._hc_post = lambda _update, _residual, _post, _comb: np.ones(
        (4, 2), dtype=np.float32
    )

    def attention(_values: np.ndarray, position: int) -> np.ndarray:
        runtime.last_attention_execution = {
            "executed": True,
            "kind": "sparse_compressed_attention_cpu_fallback",
            "position": position,
            "window_key_count": 1,
            "main_compressor_cache_before": 0,
            "main_compressor_emitted": True,
            "main_compressor_cache_after": 1,
            "index_compressor_cache_before": 0,
            "index_compressor_emitted": False,
            "index_compressor_cache_after": 0,
            "index_query_executed": False,
            "compressed_index_count": 1,
            "compressed_key_count": 1,
            "compressed_key_indices": [0],
            "attention_key_count": 2,
        }
        return np.asarray([1.0, 2.0], dtype=np.float32)

    def moe(_values: np.ndarray, *, latent_capture: dict | None = None) -> np.ndarray:
        assert latent_capture is not None
        latent_capture.update(
            {
                "router_logits": gravity._latent_array_summary(
                    np.arange(8, dtype=np.float32)
                ),
                "router_selection_scores": gravity._latent_array_summary(
                    np.arange(8, dtype=np.float32)
                ),
                "top6": {
                    "selected_expert_ids": [0, 1, 2, 3, 4, 5],
                    "selected_probabilities": [1 / 6] * 6,
                    "applied_route_weights": [0.25] * 6,
                    "probability_sum": 1.0,
                    "route_scale": 1.5,
                    "top1_top2_score_margin": 1.0,
                    "selection_cutoff_margin": 1.0,
                },
            }
        )
        return np.asarray([1.0, 2.0], dtype=np.float32)

    runtime._attention = attention
    runtime._moe = moe
    runtime._head_logits = lambda _hidden: np.asarray([0.0, 1.0, 2.0], dtype=np.float32)

    trace = gravity.DeepSeekV4DiagnosticRuntime._latent_capture_forward_token(runtime, 17)

    assert trace["schema"] == "hawking.gravity.deepseek_v4.diagnostic_latent_trace_shard.v1"
    assert trace["router"]["top6"]["selected_expert_ids"] == [0, 1, 2, 3, 4, 5]
    assert trace["router"]["router_logits"]["raw_values_retained"] is False
    assert trace["attention_index_state"]["selected_compressed_key_indices"]["raw_indices_retained"] is False
    assert trace["mhc_state"]["attention"]["input_hidden"]["shape"] == [4, 2]
    assert trace["final_hidden_state"]["raw_values_retained"] is False
    assert trace["lm_head_logits"]["shape"] == [3]
    assert trace["sampling"]["completion_text_disclosed"] is False
    assert runtime.position == 1


def test_hcli_action_whitelist_drops_goal_and_event_payload(tmp_path: Path) -> None:
    artifact_seal = "a" * 64
    receipt = tmp_path / "hcli-audit.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hide.headless.audit.v1",
                "status": "step_limit",
                "goal": "private goal must not be copied",
                "runtime": {"context_before": {"artifact_seal_sha256": artifact_seal}},
                "agent": {
                    "tool_activity": {
                        "parsed_model_tool_calls": 0,
                        "durable_tool_call_events": 0,
                        "dispatched_model_tool_calls": 0,
                    },
                    "verification": {
                        "last_verdict": {
                            "status": "inconclusive",
                            "class": "deterministic",
                            "oracle": "test",
                            "failures": [
                                {
                                    "category": "test",
                                    "code": "POLICY_DENIED",
                                    "message": "private policy detail must not be copied",
                                }
                            ],
                        },
                        "verify_result_events": 2,
                    },
                },
                "event_chain": {"ok": True, "checked_events": 13, "chain_root": "b" * 64},
                "driver": {"compute_profile": {"effect_policy": "suggest_only"}},
                "event_window": {"raw": "private event payload"},
            }
        ),
        encoding="utf-8",
    )

    summary = gravity._latent_hcli_tool_action_summary(
        [receipt], artifact_seal_sha256=artifact_seal, output_path=tmp_path / "out.json"
    )

    assert summary["availability"] == "bound_hcli_audit_metadata_whitelist"
    assert summary["records"][0]["tool_activity"]["dispatched_model_tool_calls"] == 0
    assert summary["records"][0]["event_chain"]["ok"] is True
    assert summary["records"][0]["verification"]["failure_code_frequency"] == {
        "POLICY_DENIED": 1
    }
    assert "private goal" not in json.dumps(summary, sort_keys=True)
    assert "private event" not in json.dumps(summary, sort_keys=True)
    assert "private policy" not in json.dumps(summary, sort_keys=True)


def test_latent_capture_parser_accepts_bounded_hcli_receipts() -> None:
    args = gravity._parser().parse_args(
        [
            "capture-latent-routes",
            "--artifact-dir",
            "/tmp/artifact.gravity",
            "--hcli-tool-action-receipt",
            "/tmp/audit.json",
            "--out",
            "/tmp/latent.json",
        ]
    )
    assert args.command == "capture-latent-routes"
    assert args.hcli_tool_action_receipt == ["/tmp/audit.json"]
