"""Detectors for the ways a measurement lies.

Every trap here FIRED during the O003 campaign on 2026-09-06 and cost real time.
They are kept as executable predicates rather than as remembered anecdotes,
because "the harness sometimes does X" is not a defence -- S008 section 14.

Each detector returns (ok, reason). ok=False means the trap is present.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def constant_masquerading_as_measurement(
    values: Sequence[float], *, min_variants: int = 3
) -> tuple[bool, str]:
    """A metric that cannot move when the thing it measures changes.

    FIRED: `tps_specimen` in all 55 O003 Gravity receipts held ONE value per
    specimen (O003 = 51.627) across q2-g32-experts, mixed-q2q3, q4-g128 and
    q4-g64 alike. A number identical across every variant is a stored attribute,
    not a measurement of the variant.
    """
    if len(values) < min_variants:
        return True, f"only {len(values)} variants; cannot judge"
    distinct = {round(float(v), 9) for v in values}
    if len(distinct) == 1:
        return False, (
            f"one distinct value {distinct.pop()} across {len(values)} variants: "
            "this cannot be measuring the variant"
        )
    return True, f"{len(distinct)} distinct values across {len(values)} variants"


def incomplete_accounting(acc: Mapping[str, Any]) -> tuple[bool, str]:
    """Component bytes reported as complete bytes.

    FIRED historically as 'component EBPW pretending to be complete EBPW'. A
    complete figure must account for payload AND scales AND biases AND metadata.
    """
    need = ("payload_bytes", "scales_bytes", "biases_bytes")
    missing = [k for k in need if k not in acc]
    if missing:
        return False, f"accounting omits {missing}; cannot be a COMPLETE figure"
    total = acc.get("complete_bytes")
    if total is None:
        return False, "no complete_bytes"
    parts = sum(float(acc[k]) for k in need) + float(acc.get("metadata_bytes", 0) or 0)
    if float(total) + 1e-6 < parts:
        return False, f"complete_bytes {total} is less than its own parts {parts}"
    return True, "payload, scales and biases all accounted"


def verdict_from_too_few_samples(n: int, *, minimum: int = 8) -> tuple[bool, str]:
    """A capability verdict from one generation.

    FIRED: affine-g64 was recorded COHERENT on ONE prompt scoring 4-gram 0.000.
    Its median over 8 prompts is 0.6398 against a bf16 reference of 0.0538. The
    n=1 verdict was wrong and it propagated into the ledger.
    """
    if n < minimum:
        return False, f"verdict rests on n={n}; need >= {minimum}"
    return True, f"n={n}"


def throughput_exceeds_physics(
    tok_s: float, active_bytes_per_token: float, peak_bytes_per_s: float
) -> tuple[bool, str]:
    """A rate the machine cannot produce.

    FIRED: a prefill harness reported 7,061,969 tok/s because mx.eval was called
    on a decoy array instead of the model output, so lazy graph CONSTRUCTION was
    timed rather than execution. Bytes/token times tok/s cannot exceed peak
    bandwidth.
    """
    demand = tok_s * active_bytes_per_token
    if demand > peak_bytes_per_s:
        return False, (
            f"{tok_s:.1f} tok/s x {active_bytes_per_token:.0f} B/tok = "
            f"{demand:.3e} B/s exceeds peak {peak_bytes_per_s:.3e}"
        )
    return True, f"{demand / peak_bytes_per_s * 100:.1f}% of peak"


def ablation_had_no_effect(
    signature_before: float, signature_after: float, *, tol: float = 1e-3
) -> tuple[bool, str]:
    """An ablation whose timing is meaningless because it removed nothing.

    FIRED: a decode decomposition patched __call__ on the INSTANCE. Python
    resolves dunder methods on the TYPE, so every stage measured ~0% and the
    residual was 100% unattributed -- which reads as 'it is all overhead'.
    """
    if abs(signature_before - signature_after) <= tol:
        return False, (
            f"output signature unchanged ({signature_before} vs {signature_after}); "
            "the ablation did not take effect, so its timing measures nothing"
        )
    return True, "ablation changed the output"


def distinct_specs_same_bytes(a_spec: str, a_bytes: int,
                              b_spec: str, b_bytes: int) -> tuple[bool, str]:
    """Two different configurations that produced identical bytes.

    FIRED TWICE. First when a quantize predicate accepted only nn.Linear, so 78
    SwitchLinear modules holding 90.2% of the weights were skipped and q4-g64
    and q2-g32-experts both returned 29,756,639,488. Again when a bad edit
    removed the dispatch entirely and every arm equalled bf16.
    """
    if a_spec != b_spec and a_bytes == b_bytes:
        return False, (
            f"{a_spec} and {b_spec} both produced {a_bytes} bytes: "
            "the transform probably matched nothing"
        )
    return True, "distinct specs, distinct bytes"


def gate_ignores_generation(gate: Mapping[str, Any]) -> tuple[bool, str]:
    """A capability gate scored on perplexity alone.

    FIRED FIVE TIMES on O003. q2-g128-experts scored ppl 4.085 -- only 41% above
    bf16 -- while emitting 87 consecutive identical tokens. Teacher-forced NLL
    conditions every position on ground truth and cannot see error compounding.
    """
    if not any(k in gate for k in ("median_4gram_max", "max_repeat_run_max",
                                   "distinct_ratio_min")):
        return False, "gate has no generation term; perplexity alone passes broken bodies"
    return True, "gate carries a generation term"


def no_new_information(proposals: Iterable[str]) -> tuple[bool, str]:
    """A loop that is alive but not moving.

    FIRED: the HCLI ladder proposed binarypercal0.005-g128 five times out of
    five. The cause was a harness defect -- rejections never reached the next
    prompt -- which is exactly why 'alive' must not be read as 'progressing'.
    """
    seen = list(proposals)
    if len(seen) >= 3 and len(set(seen)) == 1:
        return False, f"{len(seen)} identical proposals ({seen[0]}): no new information"
    return True, f"{len(set(seen))} distinct of {len(seen)}"


def self_certified(proposer: str, verifier: str) -> tuple[bool, str]:
    """A model agreeing with itself.

    S008: YOU DO NOT SELF-CERTIFY. YOU PROPOSE. TOOLS MEASURE. VERIFIERS DECIDE.
    """
    if proposer == verifier:
        return False, f"proposer and verifier are both {proposer}"
    return True, "verifier is independent of the proposer"


ALL_TRAPS = (
    constant_masquerading_as_measurement, incomplete_accounting,
    verdict_from_too_few_samples, throughput_exceeds_physics,
    ablation_had_no_effect, distinct_specs_same_bytes,
    gate_ignores_generation, no_new_information, self_certified,
)
