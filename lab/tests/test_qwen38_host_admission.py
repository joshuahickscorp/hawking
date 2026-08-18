"""Admission gate must refuse over-subscription. Local evidence only."""
from __future__ import annotations

from lab.operators.qwen38_host_admission import (
    DEFAULT_RESERVE_BYTES,
    MEASURED_PROCESS_CHILD_MACHINE_BYTES,
    MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES,
    SCHEMA,
    VERDICT_ADMIT,
    VERDICT_REFUSE,
    decide_admission,
    parse_vm_stat,
    process_pool_child_cost_bytes,
    prove_refuse,
)

VM = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  1000.
Pages active:                                2000.
Pages speculative:                             10.
Pages purgeable:                                5.
"""


def test_parse_vm_stat_free_is_pages_times_page_size() -> None:
    snap = parse_vm_stat(VM)
    assert snap["page_size_bytes"] == 16384
    assert snap["pages_free"] == 1000
    assert snap["free_bytes"] == 1000 * 16384


def test_refuses_when_cost_exceeds_free() -> None:
    memory = parse_vm_stat(VM)
    decision = decide_admission(
        memory,
        label="oversub",
        cost_bytes=memory["free_bytes"] + 1,
        kind="session",
    )
    assert decision["schema"] == SCHEMA
    assert decision["verdict"] == VERDICT_REFUSE
    assert decision["would_breach_reserve"] is True
    assert "refusing before swap" in decision["reason"]


def test_refuses_when_remainder_would_undercut_reserve() -> None:
    memory = parse_vm_stat(VM)
    reserve = memory["free_bytes"] // 2
    decision = decide_admission(
        memory,
        label="tight",
        cost_bytes=memory["free_bytes"] - reserve + 1,
        kind="session",
        reserve_bytes=reserve,
    )
    assert decision["verdict"] == VERDICT_REFUSE
    assert decision["free_after_if_admitted_bytes"] == reserve - 1


def test_admits_when_remainder_stays_above_reserve() -> None:
    memory = parse_vm_stat(VM)
    decision = decide_admission(
        memory,
        label="ok",
        cost_bytes=4096,
        kind="session",
        reserve_bytes=1024,
    )
    assert decision["verdict"] == VERDICT_ADMIT
    assert decision["free_after_if_admitted_bytes"] == memory["free_bytes"] - 4096


def test_prove_refuse_demonstrates_oversub() -> None:
    decision = prove_refuse(parse_vm_stat(VM))
    assert decision["verdict"] == VERDICT_REFUSE


def test_process_pool_cost_uses_measured_anchors() -> None:
    assert process_pool_child_cost_bytes(2048) == MEASURED_PROCESS_CHILD_MACHINE_BYTES
    assert process_pool_child_cost_bytes(128) == MEASURED_PROCESS_CHILD_MACHINE_BYTES
    assert process_pool_child_cost_bytes(8192) == MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES
    mid = process_pool_child_cost_bytes(4096)
    assert MEASURED_PROCESS_CHILD_MACHINE_BYTES < mid < MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES


def test_default_reserve_is_above_measured_load_spike_remainder() -> None:
    assert DEFAULT_RESERVE_BYTES > 370_000_000
