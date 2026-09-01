"""Negative controls for tools/future/resident_provider.py.

A guard nobody has watched fail is not a guard. These tests actually fire:

- a request before ready is refused, not silently queued
- a dead process is ABSENT with rss_bytes null, never healthy-with-zero
- model_open_count stays 1 across a multi-request session; a climb is a defect
- a malformed reply raises rather than being treated as an answer
- no hardware-named field is written to any receipt
- a repeating fragment is DEGENERATE, never CLEAN
- a budget-hit mid-sentence is TRUNCATED, never CLEAN
- the applied template is the artifact's own chat_template.jinja
- session turns are templated, not User:/Assistant: concatenated
- thinking arm is recorded; the sealed false is not assumed obeyed on the body
- hbm_doctor.py is not regex-fished out of the measured loop garbage

No pytest.skip: absent inputs are asserted as refusals. The 9.9GB sealed body
is never spawned; every process here is a protocol double.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from hcli.resources import pid_is_alive
from tools.future import resident_provider as rp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


@pytest.fixture
def provider(tmp_path: Path):
    p = rp.ResidentProvider()
    yield p
    p.stop()


def _spec(tmp_path: Path, *, mode: str = "ok", sleep_s: float = 0.0) -> dict:
    return rp.write_protocol_double(tmp_path / f"double-{mode}-{sleep_s}", mode=mode, sleep_s=sleep_s)


# ---------------------------------------------------------------------------
# Receipt / entry point
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = rp.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_PROVIDER.json"
    assert doc["schema"] == rp.SCHEMA
    assert doc["schema"] == "hawking.future.resident_provider.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["started_model_process"] is False
    assert doc["started_sealed_resident"] is False
    assert doc["started_protocol_double"] is True
    assert doc["took_gpu_lease"] is False
    assert doc["flock"] is False
    assert doc["proofs"]["all_passed"] is True
    for name, row in doc["proofs"]["proofs"].items():
        assert row["fires"] is True, name
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    callable_ = doc["resident_callable"]
    assert callable_["entry_point"]
    assert callable_["workunit"]
    assert callable_["receipt"].endswith("RESIDENT_PROVIDER.json")
    assert callable_["frontier"] == "FT.CHILD_RESIDENT.launch"
    assert callable_["fails_closed"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    assert "VI" not in "".join(doc["eras"])
    assert all("Odyssey IV" not in item and not item.startswith("IV ") for item in doc["odysseys"])


def test_selftest_aliases_build():
    assert rp.selftest().name == "RESIDENT_PROVIDER.json"


def test_ast_module_is_parseable():
    src = Path(rp.__file__).read_text()
    compile(src, rp.__file__, "exec")
    for needle in ("TODO", "NotImplementedError", "pytest.skip"):
        assert needle not in src


def test_receipt_contains_no_hardware_measurement_fields():
    doc = json.loads(rp.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    raise AssertionError(f"{here} = {v!r} is a hardware field")
                if k in rp.RATE_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    raise AssertionError(f"{here} = {v!r} is a rate field")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


# ---------------------------------------------------------------------------
# Ask before ready
# ---------------------------------------------------------------------------


def test_ask_before_ready_is_refused_not_queued(provider):
    with pytest.raises(rp.NotReady) as exc:
        provider.ask("s1", "hello", 4)
    assert exc.value.fault == "not_ready"
    assert "queued" in exc.value.reason


def test_ask_during_never_ready_is_refused(tmp_path: Path):
    p = rp.ResidentProvider()
    try:
        with pytest.raises(rp.NotReady):
            p.start(_spec(tmp_path, mode="never_ready"), ready_timeout_s=0.25)
        with pytest.raises(rp.NotReady):
            p.ask("s1", "hello", 4)
    finally:
        p.stop()


def test_empty_session_is_refused(provider, tmp_path: Path):
    provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    with pytest.raises(rp.ProviderRefuse) as exc:
        provider.ask("", "hello", 4)
    assert exc.value.fault == "bad_request"


def test_empty_prompt_is_refused(provider, tmp_path: Path):
    provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    with pytest.raises(rp.ProviderRefuse) as exc:
        provider.ask("s1", "  ", 4)
    assert exc.value.fault == "bad_request"


def test_zero_tokens_is_refused(provider, tmp_path: Path):
    provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    with pytest.raises(rp.ProviderRefuse) as exc:
        provider.ask("s1", "hello", 0)
    assert exc.value.fault == "bad_request"


# ---------------------------------------------------------------------------
# Dead process health
# ---------------------------------------------------------------------------


def test_dead_process_health_is_absent_rss_null(provider, tmp_path: Path):
    handle = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    pid = int(handle["pid"])
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if provider._proc is not None and provider._proc.poll() is not None:
            break
        time.sleep(0.02)
    h = provider.health()
    assert h["presence"] == "ABSENT"
    assert h["rss_bytes"] is None
    assert h["alive"] is False
    assert h["dead"] is True
    assert h["ready"] is False
    assert "healthy" not in h
    with pytest.raises(rp.DeadProcess):
        provider.ask("s1", "hello", 4)


def test_undeclared_health_is_not_healthy_with_zero():
    h = rp.ResidentProvider().health()
    assert h["presence"] == "UNDECLARED"
    assert h["rss_bytes"] is None
    assert h["alive"] is False
    assert h["ready"] is False
    assert "healthy" not in h


# ---------------------------------------------------------------------------
# One body / reuse
# ---------------------------------------------------------------------------


def test_model_open_count_stays_one_across_session(provider, tmp_path: Path):
    handle = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    pid = handle["pid"]
    replies = [
        provider.ask("s1", "What is the capital of France?", 3),
        provider.ask("s1", "Name one reason a status label can be wrong.", 8),
        provider.ask("s1", "third", 2),
    ]
    for row in replies:
        assert row["model_open_count"] == 1
        assert row["weight_upload_count"] == 1
        assert row["pid"] == pid
        assert row["status"] == "ok"
    assert provider.health()["requests_served"] == 3
    assert provider.handle()["pid"] == pid


def test_start_reuses_live_process(provider, tmp_path: Path):
    a = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    b = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    assert a["pid"] == b["pid"]
    assert provider.health()["generation"] == 1


def test_two_sessions_share_one_pid(provider, tmp_path: Path):
    handle = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    provider.ask("alpha", "first session", 2)
    provider.ask("beta", "second session", 2)
    snap = provider.sessions()
    assert snap["n"] == 2
    assert snap["pid"] == handle["pid"]
    assert snap["second_model_body"] is False
    assert snap["same_process"] is True
    ids = {row["id"] for row in snap["sessions"]}
    assert ids == {"alpha", "beta"}


def test_slot_idle_when_nothing_in_flight(provider, tmp_path: Path):
    provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    provider.ask("s1", "ping", 2)
    h = provider.health()
    assert h["queue_depth"] == 0
    assert h["in_flight"] is None
    assert h["presence"] == "PRESENT"
    assert isinstance(h["rss_bytes"], int) or h["rss_bytes"] is None


def test_concurrent_asks_one_pid(provider, tmp_path: Path):
    handle = provider.start(
        _spec(tmp_path, sleep_s=0.15), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S
    )
    pid = handle["pid"]
    results: list[dict] = []
    errors: list[BaseException] = []

    def _go(name: str) -> None:
        try:
            results.append(provider.ask(name, f"hello {name}", 2, timeout_s=5.0))
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_go, args=("a",))
    t2 = threading.Thread(target=_go, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []
    assert len(results) == 2
    assert {row["pid"] for row in results} == {pid}
    assert {row["model_open_count"] for row in results} == {1}
    idle = provider.health()
    assert idle["queue_depth"] == 0
    assert idle["in_flight"] is None


# ---------------------------------------------------------------------------
# Malformed / error / reload
# ---------------------------------------------------------------------------


def test_malformed_reply_raises(provider, tmp_path: Path):
    provider.start(_spec(tmp_path, mode="malformed"), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    with pytest.raises(rp.MalformedReply) as exc:
        provider.ask("s1", "ping", 1)
    assert exc.value.fault == "malformed_reply"


def test_error_status_is_not_an_answer(provider, tmp_path: Path):
    provider.start(_spec(tmp_path, mode="error_status"), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    with pytest.raises(rp.AskFailed) as exc:
        provider.ask("s1", "ping", 1)
    assert exc.value.fault == "ask_failed"


def test_weight_reload_is_a_defect(provider, tmp_path: Path):
    provider.start(_spec(tmp_path, mode="reload"), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    with pytest.raises(rp.WeightReload) as exc:
        provider.ask("s1", "ping", 1)
    assert exc.value.fault == "weight_reload"
    assert "model_open_count" in exc.value.reason


def test_dirty_metrics_are_stripped_from_ask(provider, tmp_path: Path):
    provider.start(_spec(tmp_path, mode="dirty_metrics"), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    got = provider.ask("s1", "ping", 1)
    leaked = rp._hardware_numeric_keys(got)
    assert leaked == []
    for key in HARDWARE_FIELDS | rp.RATE_FIELDS:
        assert key not in got
        assert key not in got["cost"]
    assert got["cost"]["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert got["cost"]["gpu_authority"] is False
    assert got["cost"]["ranks_nothing"] is True
    assert "elapsed_s" in got["cost"]


# ---------------------------------------------------------------------------
# Stop / restart
# ---------------------------------------------------------------------------


def test_restart_reaches_ready_and_serves(provider, tmp_path: Path):
    first = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    provider.ask("s1", "before", 2)
    old = int(first["pid"])
    restarted = provider.restart(ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    new = int(restarted["handle"]["pid"])
    assert new != old
    assert not pid_is_alive(old)
    assert pid_is_alive(new)
    reply = provider.ask("s1", "after restart", 2)
    assert reply["status"] == "ok"
    assert reply["pid"] == new
    assert reply["model_open_count"] == 1
    assert restarted["ready"] is True


def test_stop_makes_process_gone(provider, tmp_path: Path):
    handle = provider.start(_spec(tmp_path), ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S)
    pid = int(handle["pid"])
    stopped = provider.stop()
    assert stopped["stopped"] is True
    assert stopped["result"] == "ok"
    assert not pid_is_alive(pid)
    h = provider.health()
    assert h["presence"] == "UNDECLARED"
    assert h["rss_bytes"] is None


def test_relative_argv_is_refused():
    with pytest.raises(rp.ProviderRefuse) as exc:
        rp.explicit_launch(argv=["python3", "/tmp/nope.py"])
    assert exc.value.fault == "runtime_resolution_refused"


def test_missing_binary_start_refuses(tmp_path: Path):
    missing = rp.explicit_launch(
        argv=[str(Path(sys.executable).resolve()), str(tmp_path / "no-such-double.py")],
        cwd=str(tmp_path),
        binary=str(tmp_path / "no-such-double.py"),
    )
    assert missing["present"] is False
    p = rp.ResidentProvider()
    try:
        with pytest.raises(rp.ProviderRefuse) as exc:
            p.start(missing)
        assert exc.value.fault == "launch_absent"
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Resolve launch / live probe
# ---------------------------------------------------------------------------


def test_resolve_launch_does_not_spawn_and_names_stdin_pipe():
    sealed = rp.resolve_launch()
    assert sealed["started_model_process"] is False
    assert sealed["gpu_authority"] is False
    assert sealed["stdin"] == "PIPE"
    assert sealed["protocol"] == rp.PROTOCOL
    if sealed.get("argv"):
        flags = [t for t in sealed["argv"] if str(t).startswith("--")]
        assert flags == [
            "--artifact-root",
            "--tokenizer",
            "--max-seq-len",
            "--resident-identity",
        ]
        assert sealed["identity"] == "sealed-3.14"
    else:
        assert sealed["present"] is False
        assert sealed.get("reason")


def test_live_probe_is_the_invocation_authority():
    probe, src = rp.load_live_probe()
    assert probe is not None, src
    assert probe["protocol"] == rp.PROTOCOL
    assert probe["verdict"] == "RESIDENT_STARTS_AND_GENERATES"
    assert probe["gpu_authority"] is False
    observed = probe["observed"]
    assert observed["reached_ready"] is True
    assert observed["requests_served"] == 2
    assert observed["fallbacks"] == 0
    assert "tps" not in probe or not isinstance(probe.get("tps"), (int, float))


def test_write_receipt_rejects_hardware_named_field(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rp, "RECEIPT", "RESIDENT_PROVIDER_HW_TRAP.json")
    # The module's write_receipt goes through _common; a forged doc must raise.
    from tools.future._common import HardwareClaimError, write_receipt

    with pytest.raises(HardwareClaimError):
        write_receipt(
            "RESIDENT_PROVIDER_HW_TRAP.json",
            {
                "schema": "trap",
                "evidence_class": "STATIC_ONLY",
                "gpu_authority": False,
                "wall_ns": 12,
            },
            "tools/future/resident_provider.py",
        )
    trap = RECEIPTS / "RESIDENT_PROVIDER_HW_TRAP.json"
    if trap.is_file():
        trap.unlink()


# ---------------------------------------------------------------------------
# Chat template: the artifact's own jinja, not an invented format
# ---------------------------------------------------------------------------


GOLD_FRANCE_THINK_OFF = (
    "<|im_start|>user\n"
    f"{rp.PROMPT_FRANCE}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
GOLD_FRANCE_THINK_ON = (
    "<|im_start|>system\n"
    f"{rp.REASONING_XHIGH}<|im_end|>\n"
    "<|im_start|>user\n"
    f"{rp.PROMPT_FRANCE}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
)


def test_template_came_from_artifact_chat_template_jinja():
    binding = rp.load_chat_template()
    artifact = Path("/Users/scammermike/noetic/NOETIC_PARENT_A") / "chat_template.jinja"
    assert binding["present"] is True, binding.get("reason")
    assert binding["filename"] == "chat_template.jinja"
    assert artifact.is_file(), binding.get("reason")
    assert binding["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert Path(binding["path"]).resolve() == artifact.resolve()
    assert binding["tokenizer_config_chat_template_match"] is True


def test_render_matches_artifact_gold_thinking_off():
    binding = rp.load_chat_template()
    assert binding["present"] is True, binding.get("reason")
    got = rp.render_chat(
        [{"role": "user", "content": rp.PROMPT_FRANCE}],
        enable_thinking=False,
        template_text=binding["text"],
        template_sha256=binding["sha256"],
    )
    assert got["text"] == GOLD_FRANCE_THINK_OFF
    assert got["source_sha256"] == binding["sha256"]
    assert got["templated"] is True
    assert got["concatenated"] is False
    assert "User: " not in got["text"]
    assert got["applied_arm"] == "thinking_off"
    assert got["text"].endswith(rp.CLOSED_THINK)


def test_render_thinking_on_arm_is_available_and_not_the_applied_default():
    binding = rp.load_chat_template()
    assert binding["present"] is True, binding.get("reason")
    got = rp.render_chat(
        [{"role": "user", "content": rp.PROMPT_FRANCE}],
        enable_thinking=True,
        template_text=binding["text"],
        template_sha256=binding["sha256"],
    )
    assert got["text"] == GOLD_FRANCE_THINK_ON
    assert got["applied_arm"] == "thinking_on"
    declared = rp.load_declared_thinking()
    assert declared["present"] is True
    assert declared["enable_thinking"] is False
    assert got["enable_thinking"] is not declared["enable_thinking"]


def test_session_turns_are_templated_not_concatenated():
    binding = rp.load_chat_template()
    assert binding["present"] is True, binding.get("reason")
    got = rp.render_chat(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
        enable_thinking=False,
        template_text=binding["text"],
        template_sha256=binding["sha256"],
    )
    assert "<|im_start|>user\none<|im_end|>" in got["text"]
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\ntwo<|im_end|>" in got["text"]
    assert "<|im_start|>user\nthree<|im_end|>" in got["text"]
    assert "User: one" not in got["text"]
    assert "Assistant: two" not in got["text"]
    assert got["text"].endswith(rp.CLOSED_THINK)


def test_foreign_template_is_refused():
    with pytest.raises(rp.TemplateRefuse):
        rp.render_chat(
            [{"role": "user", "content": "hi"}],
            enable_thinking=False,
            template_text="not a qwen chat template",
        )


def test_tool_role_is_refused_not_guessed():
    binding = rp.load_chat_template()
    assert binding["present"] is True, binding.get("reason")
    with pytest.raises(rp.TemplateRefuse) as exc:
        rp.render_chat(
            [{"role": "tool", "content": "x"}, {"role": "user", "content": "q"}],
            enable_thinking=False,
            template_text=binding["text"],
        )
    assert "tool" in exc.value.reason


def test_empty_messages_refused():
    binding = rp.load_chat_template()
    assert binding["present"] is True, binding.get("reason")
    with pytest.raises(rp.TemplateRefuse):
        rp.render_chat([], enable_thinking=False, template_text=binding["text"])


def test_declared_thinking_arm_is_false_not_assumed_obeyed():
    arm = rp.load_declared_thinking()
    assert arm["present"] is True
    assert arm["enable_thinking"] is False
    assert arm["source"].endswith("generation.enable_thinking")
    rec = rp.thinking_arm_record(
        declared=arm,
        applied=False,
        observed="no_think_tag",
        observed_on="protocol_double",
    )
    assert rec["config_obeyed"] == "UNPROBED_ON_SEALED_BODY"
    assert rec["applied"] is False
    live = rp.thinking_arm_record(
        declared=arm,
        applied=False,
        observed="thinking_opened",
        observed_on="sealed_resident",
    )
    assert live["config_obeyed"] is False


# ---------------------------------------------------------------------------
# Stop + quality. Negative controls that actually reject.
# ---------------------------------------------------------------------------


def test_repeating_fragment_is_degenerate_never_clean():
    got = rp.finalize_generation(
        text=rp.MEASURED_DEGENERATE_CHOICE,
        generated_tokens=64,
        max_new_tokens=64,
    )
    assert got["quality"] == rp.QUALITY_DEGENERATE
    assert rp.quality(
        {
            "raw_text": rp.MEASURED_DEGENERATE_CHOICE,
            "generated_tokens": 64,
            "max_new_tokens": 64,
        }
    ) == rp.QUALITY_DEGENERATE
    assert got["text"] != "hbm_doctor.py"
    assert "hbm_doctor.py" not in got["text"]


def test_status_loop_is_degenerate_never_clean():
    got = rp.finalize_generation(
        text=rp.MEASURED_DEGENERATE_STATUS,
        generated_tokens=48,
        max_new_tokens=48,
    )
    assert got["quality"] == rp.QUALITY_DEGENERATE
    assert got["text"] in {"", "h"} or got["quality"] == rp.QUALITY_DEGENERATE


def test_truncated_mid_sentence_is_not_clean():
    got = rp.finalize_generation(
        text="A status label is a claim about",
        generated_tokens=16,
        max_new_tokens=16,
    )
    assert got["quality"] == rp.QUALITY_TRUNCATED
    assert got["stopped_at"] == "max_new_tokens"
    assert rp.quality(
        {
            "text": "A status label is a claim about",
            "generated_tokens": 16,
            "max_new_tokens": 16,
        }
    ) == rp.QUALITY_TRUNCATED


def test_clean_paris_is_extractable():
    got = rp.finalize_generation(
        text=rp.MEASURED_CLEAN_FRANCE,
        generated_tokens=3,
        max_new_tokens=32,
        new_token_ids=[11, 12, rp.EOS_IM_END_ID],
    )
    assert got["quality"] == rp.QUALITY_CLEAN
    assert got["text"] == "Paris"
    assert got["extractable"] is True


def test_cut_at_im_end_keeps_the_answer():
    got = rp.finalize_generation(
        text="hbm_doctor.py<|im_end|>\n<|im_start|>user\njunk",
        generated_tokens=8,
        max_new_tokens=32,
    )
    assert got["text"] == "hbm_doctor.py"
    assert got["quality"] == rp.QUALITY_CLEAN
    assert got["stopped_at"] == "eos_im_end"


def test_cut_at_fabricated_turn_does_not_fish_the_middle():
    # Prefix empty: the measured defect starts with Assistant:. Must not
    # become CLEAN by scanning ahead for a filename.
    got = rp.finalize_generation(
        text="\nAssistant:\n\nhbm_doctor.py\n",
        generated_tokens=8,
        max_new_tokens=32,
        new_token_ids=[rp.EOS_IM_END_ID],
    )
    assert got["text"] != "hbm_doctor.py"
    assert got["quality"] == rp.QUALITY_DEGENERATE


def test_answer_then_one_fabricated_turn_is_clean_cut():
    got = rp.finalize_generation(
        text="hbm_doctor.py\nAssistant:\nmore",
        generated_tokens=6,
        max_new_tokens=32,
        new_token_ids=[1, 2, 3, rp.EOS_IM_END_ID],
    )
    assert got["text"] == "hbm_doctor.py"
    assert got["quality"] == rp.QUALITY_CLEAN
    assert got["stopped_at"] in {"fabricated_turn", "eos_token_id"}


def test_quality_without_budget_refuses_rather_than_calling_clean():
    with pytest.raises(rp.QualityUnproven):
        rp.quality({"text": "Paris"})
    with pytest.raises(rp.QualityUnproven):
        rp.finalize_generation(text="maybe done", generated_tokens=None, max_new_tokens=8)


def test_is_degenerate_negative_on_real_answers():
    assert rp.is_degenerate("Paris") is False
    assert rp.is_degenerate("hbm_doctor.py") is False
    assert rp.is_degenerate(
        "A status label is a claim; the world state is independently true."
    ) is False
    assert rp.is_degenerate(rp.MEASURED_DEGENERATE_CHOICE) is True


def test_ask_quality_scripted_three_prompts_clean(provider, tmp_path: Path):
    provider.start(
        rp.write_protocol_double(tmp_path / "q", mode="quality_scripted"),
        ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S,
    )
    france = provider.ask("s1", rp.PROMPT_FRANCE, 32)
    choice = provider.ask("s2", rp.PROMPT_CHOICE, 32)
    status = provider.ask("s3", rp.PROMPT_STATUS, 32)
    assert france["quality"] == rp.QUALITY_CLEAN
    assert france["text"] == "Paris"
    assert choice["quality"] == rp.QUALITY_CLEAN
    assert choice["text"] == "hbm_doctor.py"
    assert status["quality"] == rp.QUALITY_CLEAN
    assert status["text"]
    assert france["templated"] is True
    assert france["thinking_arm"]["applied"] is False
    assert france["thinking_arm"]["declared"] is False
    assert france["thinking_arm"]["config_obeyed"] == "UNPROBED_ON_SEALED_BODY"
    wire = provider._last_wire_prompt or ""
    assert "<|im_start|>user" in wire
    assert rp.CLOSED_THINK in wire
    assert "User: " not in wire


def test_ask_session_history_is_templated(provider, tmp_path: Path):
    provider.start(
        rp.write_protocol_double(tmp_path / "hist", mode="quality_scripted"),
        ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S,
    )
    provider.ask("hist", "one", 8)
    provider.ask("hist", "two", 8)
    wire = provider._last_wire_prompt or ""
    assert "<|im_start|>user\none<|im_end|>" in wire
    assert "User: one" not in wire
    assert "Assistant: " not in wire
    snap = provider.sessions()
    assert snap["templated"] is True
    assert snap["concatenated"] is False


def test_ask_degenerate_mode_is_labelled(provider, tmp_path: Path):
    provider.start(
        rp.write_protocol_double(tmp_path / "d", mode="degenerate"),
        ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S,
    )
    got = provider.ask("s", rp.PROMPT_CHOICE, 64)
    assert got["quality"] == rp.QUALITY_DEGENERATE
    assert got["text"] != "hbm_doctor.py"
    assert rp.quality(got) == rp.QUALITY_DEGENERATE


def test_ask_truncated_mode_is_labelled(provider, tmp_path: Path):
    provider.start(
        rp.write_protocol_double(tmp_path / "t", mode="truncated"),
        ready_timeout_s=rp.DOUBLE_READY_TIMEOUT_S,
    )
    got = provider.ask("s", rp.PROMPT_STATUS, 16)
    assert got["quality"] == rp.QUALITY_TRUNCATED
    assert rp.quality(got) == rp.QUALITY_TRUNCATED


def test_build_records_unprobed_sealed_quality():
    doc = json.loads(rp.build().read_text())
    assert doc["quality_probes"]["proved_on_sealed_body"] is False
    assert doc["thinking_arm"]["observed_on_sealed_body"] == "UNPROBED"
    assert doc["thinking_arm"]["declared"] is False
    assert doc["thinking_arm"]["applied_arm"] == "thinking_off"
    assert doc["chat_template"]["filename"] == "chat_template.jinja"
    assert doc["chat_template"]["present"] is True
    assert "text" not in doc["chat_template"]
    assert doc["started_sealed_resident"] is False
    assert "tps" not in json.dumps(doc["quality_probes"])


# --- G120: salvage is in the reply path, with provenance ----------------------

def test_salvage_provenance_on_a_clean_reply_claims_nothing():
    p = rp.salvage_provenance("a short clean answer.", "a short clean answer.",
                              False)
    assert p["degeneration_class"] == "none"
    assert p["degeneration_start"] is None
    assert p["salvaged"] is False
    assert p["salvage_fraction"] == 1.0


def test_salvage_provenance_records_where_the_loop_starts():
    """A caller must be able to tell a two-sentence answer that was always short
    from one carved out of a thousand-character loop."""
    full = "good start. " + ("loop loop " * 40)
    p = rp.salvage_provenance(full, "good start.", True)
    assert p["degeneration_class"] == "tail_after_clean_prefix"
    assert p["degeneration_start"] == len("good start.")
    assert p["salvaged"] is True
    assert 0.0 < p["salvage_fraction"] < 0.1
    assert p["full_chars"] > p["salvaged_chars"]


def test_a_wholly_degenerate_reply_is_classed_separately():
    p = rp.salvage_provenance("loop loop loop", "", True)
    assert p["degeneration_class"] == "whole_reply"
    assert p["salvaged"] is False


def test_the_two_hashes_differ_when_anything_was_dropped():
    full = "good start. " + ("loop loop " * 40)
    p = rp.salvage_provenance(full, "good start.", True)
    assert p["full_reply_hash"] != p["clean_prefix_hash"]
    same = rp.salvage_provenance("x", "x", False)
    assert same["full_reply_hash"] == same["clean_prefix_hash"]


def test_the_reply_path_calls_salvage_rather_than_leaving_it_to_the_caller():
    import inspect
    src = inspect.getsource(rp.finalize_generation)
    assert "degenerate_prefix(answer)" in src, (
        "salvage must happen in the provider, not be a library function every "
        "consumer has to remember"
    )
    assert '"salvage": salvage_provenance' in src


def test_reasoning_quality_rates_are_null_before_any_reply():
    """A rate reported as 0.0 on zero requests reads as a failure when it is an
    absence."""
    class _Bare:
        _quality_counts = {"replies": 0}
        reasoning_quality = rp.ResidentProvider.reasoning_quality
    q = _Bare().reasoning_quality()
    assert q["replies"] == 0
    assert q["degeneration_rate"] is None
    assert q["salvage_rate"] is None


def test_reasoning_quality_divides_by_replies():
    class _Some:
        _quality_counts = {"replies": 4, "clean": 3, "truncated": 1,
                           "degenerate": 2, "salvaged": 1}
        reasoning_quality = rp.ResidentProvider.reasoning_quality
    q = _Some().reasoning_quality()
    assert q["degeneration_rate"] == 0.5
    assert q["salvage_rate"] == 0.25
    assert q["clean_rate"] == 0.75


def test_reasoning_quality_names_what_it_does_not_measure():
    """Structured-reply rate and useful-hypothesis rate belong to the caller's
    schema and mission. Inventing them here would double-count."""
    class _Bare:
        _quality_counts = {"replies": 0}
        reasoning_quality = rp.ResidentProvider.reasoning_quality
    assert "double-count" in _Bare().reasoning_quality()["not_measured_here"]
