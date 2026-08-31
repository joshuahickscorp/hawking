"""TPS GAP — attribute complete-minus-decode from the resident's own clocks.

The sealed profile carries two numbers, 34.0 and 24.4086, and a live probe
today cited decode ~35.5 against complete ~27.2. This module answers two
questions and then stops: where the complete-token clock actually goes, and
whether the historical 34.0 is the same quantity as the current 24.4.

It refuses to become a campaign. It does not take a GPU lease, does not
flock a bench lock, does not quiesce the machine, and does not qualify a
TPS. Every number it writes is either a citation with a path or a
dimensionless share of clocks the resident already returned. write_receipt
raises on hardware-named fields; timings live under self_timing and say
what they are not.

Fail-closed: missing clocks are a named refusal, never a rounded pass.
A residue that is not a named field is UNATTRIBUTED and is never folded
into prefill or decode to make the arithmetic close. A change that WOULD
improve TPS is refused unless a paired observation labelled
SELF_MEASURED_DIRTY is in hand — and this module did not take one.

Cannot establish: a protected current-vs-historical A/B, that batched
prefill is legal given DeltaNet recurrent state, or the original
measurement conditions of the hardcoded 24.4086.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
from typing import Any, Mapping, Sequence

from tools.future._common import (
    HARDWARE_FIELDS,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)
from tools.future.qwen27_profile_schema import HISTORICAL_REL, SEALED_REL, load_authority

RECEIPT = "TPS_GAP.json"
SCHEMA = "hawking.future.tps_gap.v1"
VERSION = 1
RECORDED_BY = "tools/future/tps_gap.py"

QUAL_REL = "receipts/headless/QWEN_PERFORMANCE_QUALIFICATION.json"
PROBE_REL = "receipts/future/evidence/RESIDENT_LIVE_PROBE.json"
SMOKE_REL = "receipts/headless/HCLI_ACCELERATOR_NATIVE_SMOKE.json"
REGRESSION_REL = "receipts/headless/HCLI_ACCELERATOR_REGRESSION.json"
DISPATCH_COST_REL = "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"
RESIDENT_SRC = "crates/hawking-core/examples/ascension_qwen38_resident.rs"
GENERATE_SRC = "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
HYBRID_BIN = "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
RESIDENT_BIN = "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident"

EVIDENCE_CLASS = "STATIC_ONLY"
DIRTY = "SELF_MEASURED_DIRTY"
UNATTRIBUTED = "UNATTRIBUTED"
UNKNOWN = "UNKNOWN"
HYPOTHESIS = "HYPOTHESIS"

# Four clocks the resident returns per request. generation is result.wall_ns;
# request is the Instant the example wraps around generate_greedy.
CLOCK_PREFILL = "prefill_wall_ns"
CLOCK_DECODE = "decode_wall_ns"
CLOCK_GENERATION = "generation_wall_ns"
CLOCK_REQUEST = "request_wall_ns"

FUSION_KEYS = (
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM",
    "HAWKING_QWEN38_FUSE_GQA_QKV",
    "HAWKING_QWEN38_FUSE_DN_INPROJ",
    "HAWKING_QWEN38_FUSE_MLP",
)

# Selected fused graph from QWEN27_HISTORICAL_RUNTIME_IDENTITY; qualification
# runs recorded 964 dispatches/step, which is the unfused baseline.
FUSED_DISPATCHES = 628
UNFUSED_DISPATCHES = 964

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Citations and dimensionless clock shares only. "
    "SELF_MEASURED_DIRTY if a live reply is supplied; never PROTECTED_ABSOLUTE, "
    "never a qualified TPS, never a ranking, never a promotion input."
)


class GapRefuse(ValueError):
    """A required clock or receipt was absent, or a fold was attempted."""


class WouldImproveRefuse(ValueError):
    """A WOULD-improve-TPS claim without a labelled dirty paired observation."""


# ---------------------------------------------------------------------------
# Guards. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def assert_timing_key_legal(name: str) -> None:
    """Refuse hardware-named keys even for a citation."""
    if name in HARDWARE_FIELDS:
        raise HardwareClaimError(
            f"{name}: sidecar has no GPU authority; put the number under "
            "self_timing as prefill_ns / decode_ns / residue_ns, never as a "
            "hardware field name"
        )


def refuse_fold_into_named_bucket(residue_label: str, target: str) -> None:
    """UNATTRIBUTED stays UNATTRIBUTED. Closing the arithmetic is the failure."""
    raise GapRefuse(
        f"refusing to fold {residue_label} into {target}: a residue attributed "
        "to nothing is UNATTRIBUTED, never prefill or decode"
    )


def refuse_would_improve(
    change: str,
    *,
    dirty_measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A plan is not a speedup. Only a labelled dirty paired observation talks.

    Even then the return is a HYPOTHESIS that cannot promote. This module
    itself never took that observation.
    """
    if dirty_measurement is None:
        raise WouldImproveRefuse(
            f"{change!r} WOULD improve TPS: refused; no measurement at all"
        )
    klass = dirty_measurement.get("evidence_class")
    if klass != DIRTY:
        raise WouldImproveRefuse(
            f"{change!r} WOULD improve TPS: refused; evidence_class={klass!r} "
            f"is not {DIRTY}"
        )
    if not dirty_measurement.get("observed_before_and_after"):
        raise WouldImproveRefuse(
            f"{change!r} WOULD improve TPS: refused; {DIRTY} without a paired "
            "before/after is still a plan"
        )
    # A caller can hand in a fixture that satisfies the label. The production
    # receipt never calls this with a real pair; tests watch the admit path
    # so the guard cannot silently become "always raise".
    return {
        "change": change,
        "status": HYPOTHESIS,
        "evidence_class": DIRTY,
        "does_not_qualify": True,
        "does_not_promote": True,
        "does_not_rank": True,
        "this_module_did_not_take_the_pair": True,
        "would_improve": False,
        "why_not_would": (
            "a labelled dirty pair may rank and prune; it is not a WOULD "
            "that lands in a receipt as a causal claim"
        ),
    }


# ---------------------------------------------------------------------------
# Clock extraction. Missing is a refusal, never a zero.
# ---------------------------------------------------------------------------


def _as_ns(value: Any, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GapRefuse(f"{what} is {value!r}, not a duration in ns")
    if value < 0:
        raise GapRefuse(f"{what} is {value!r}; a duration cannot be negative")
    return int(value)


def _first_present(reply: Mapping[str, Any], names: Sequence[str]) -> tuple[str, Any] | None:
    for name in names:
        if name in reply and reply[name] is not None:
            return name, reply[name]
    return None


def extract_clocks(reply: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pull the four resident clocks. Absence is named, never defaulted."""
    if not isinstance(reply, Mapping):
        raise GapRefuse("resident reply is absent; cannot decompose complete vs decode")
    missing: list[str] = []
    got: dict[str, int] = {}
    aliases = {
        CLOCK_PREFILL: (CLOCK_PREFILL,),
        CLOCK_DECODE: (CLOCK_DECODE,),
        CLOCK_GENERATION: (CLOCK_GENERATION,),
        # Resident JSON names the wrapper Instant `wall_ns`. We read it; we
        # never write that key into a receipt.
        CLOCK_REQUEST: (CLOCK_REQUEST, "wall_ns"),
    }
    used_alias: dict[str, str] = {}
    for dest, names in aliases.items():
        hit = _first_present(reply, names)
        if hit is None:
            missing.append(dest)
            continue
        used, raw = hit
        got[dest] = _as_ns(raw, what=used)
        used_alias[dest] = used
    if missing:
        raise GapRefuse(
            "resident reply missing required clock(s): " + ", ".join(missing)
        )
    metrics = reply.get("metrics") if isinstance(reply.get("metrics"), Mapping) else {}
    complete = None
    if isinstance(metrics, Mapping) and metrics.get("complete_wall_ns") is not None:
        complete = _as_ns(metrics.get("complete_wall_ns"), what="metrics.complete_wall_ns")
    return {
        "prefill_ns": got[CLOCK_PREFILL],
        "decode_ns": got[CLOCK_DECODE],
        "generation_ns": got[CLOCK_GENERATION],
        "request_ns": got[CLOCK_REQUEST],
        "complete_ns": complete,
        "aliases_used": used_alias,
        "generated_tokens": reply.get("generated_tokens"),
        "decode_steps": reply.get("decode_steps"),
        "prompt_tokens": reply.get("prompt_tokens") if reply.get("prompt_tokens") is not None else reply.get("prompt_len"),
    }


def _share(part: int, whole: int) -> float | None:
    if whole <= 0:
        return None
    return part / whole


# ---------------------------------------------------------------------------
# Q1 — what separates decode from complete
# ---------------------------------------------------------------------------


def decompose_reply(reply: Mapping[str, Any] | None) -> dict[str, Any]:
    """Split request wall into prefill, decode, named ceremony, UNATTRIBUTED.

    Source structure (generate_greedy / ascension_qwen38_resident.rs):

    * resident Instant starts, then generate_greedy calls session.reset()
      again and allocates step vecs, then its own wall Instant starts.
      request_ns - generation_ns is that interval.
    * wall Instant, then prefill Instant over session.step per prompt token
      (last prompt step emits new-token[0]), then tokens.push + IGNORE_EOS
      env lookup, then decode Instant over the remaining new tokens.
      generation_ns - prefill_ns - decode_ns is that glue.
    * decode_steps = n_new - 1. complete_tps uses n_new / request_ns;
      decode_tps uses decode_steps / decode_ns. Off-by-one is a definition,
      not a kernel.

    None of that lets us fold a leftover into prefill or decode. Leftover
    is UNATTRIBUTED. A source hypothesis about the leftover is labelled
    HYPOTHESIS and lives next to it, not inside it.
    """
    clocks = extract_clocks(reply)
    prefill = clocks["prefill_ns"]
    decode = clocks["decode_ns"]
    generation = clocks["generation_ns"]
    request = clocks["request_ns"]
    complete = clocks["complete_ns"]

    named_generation = prefill + decode
    generation_residue = generation - named_generation
    request_ceremony = request - generation
    post = None if complete is None else complete - request

    buckets = [
        {
            "id": "prefill",
            "ns": prefill,
            "label": "PREFILL",
            "why": (
                "prompt walk, including the last prompt step that emits "
                "new-token[0]. Sequential session.step per prompt token."
            ),
            "source": GENERATE_SRC,
        },
        {
            "id": "decode",
            "ns": decode,
            "label": "DECODE",
            "why": "new-tokens[1..] inside generate_greedy's decode Instant",
            "source": GENERATE_SRC,
        },
        {
            "id": "request_ceremony",
            "ns": request_ceremony,
            "label": "REQUEST_CEREMONY",
            "why": (
                "resident Instant includes generate_greedy's inner session.reset "
                "and step-vec allocation, which sit before generate_greedy's "
                "own wall Instant"
            ),
            "source": RESIDENT_SRC,
            "status": HYPOTHESIS,
        },
    ]
    unattr_ns = generation_residue
    extra: list[dict[str, Any]] = []
    if post is not None:
        extra.append(
            {
                "id": "post_generation",
                "ns": post,
                "label": "POST_GENERATION",
                "why": (
                    "complete_wall_ns is taken after tokenizer.decode of the "
                    "full generated string; complete_tps uses request_ns, so "
                    "this interval is not in complete_tps"
                ),
                "source": RESIDENT_SRC,
                "in_complete_tps_denominator": False,
            }
        )

    # Negative residue is a clock inconsistency, not a gift to another bucket.
    if generation_residue < 0 or request_ceremony < 0:
        clock_state = "CLOCK_INCONSISTENT"
    else:
        clock_state = "ARITHMETIC_CLOSED_WITH_UNATTRIBUTED"

    return {
        "clocks": {
            "prefill_ns": prefill,
            "decode_ns": decode,
            "generation_ns": generation,
            "request_ns": request,
            "complete_ns": complete,
        },
        "aliases_used": clocks["aliases_used"],
        "generated_tokens": clocks["generated_tokens"],
        "decode_steps": clocks["decode_steps"],
        "prompt_tokens": clocks["prompt_tokens"],
        "buckets": buckets,
        "unattributed": {
            "id": "generation_residue",
            "ns": unattr_ns,
            "label": UNATTRIBUTED,
            "why": (
                "generation_ns - prefill_ns - decode_ns is not a named field. "
                "Source hypothesis (HYPOTHESIS): tokens.push + IGNORE_EOS env "
                "lookup between the two Instants. A large value falsifies that "
                "hypothesis rather than getting a friendlier name."
            ),
            "source_hypothesis": {
                "status": HYPOTHESIS,
                "cause": "inter_phase_glue",
                "does_not_establish_cause": True,
            },
            "folded_into_prefill": False,
            "folded_into_decode": False,
        },
        "post_generation": extra[0] if extra else None,
        "shares_of_request": {
            "prefill": _share(prefill, request),
            "decode": _share(decode, request),
            "request_ceremony": _share(request_ceremony, request),
            "unattributed": _share(unattr_ns, request) if unattr_ns >= 0 else None,
        },
        "clock_state": clock_state,
        "evidence_class": DIRTY,
        "not_protected_absolute": True,
        "not_a_qualified_tps": True,
    }


def first_token_accounting(reply: Mapping[str, Any] | None) -> dict[str, Any]:
    """decode_steps is n_new-1. That is a definition gap, not a kernel gap."""
    clocks = extract_clocks(reply)
    n_gen = clocks["generated_tokens"]
    n_dec = clocks["decode_steps"]
    if not isinstance(n_gen, int) or not isinstance(n_dec, int):
        return {
            "status": UNKNOWN,
            "reason": "generated_tokens or decode_steps absent; not inferred",
            "off_by_one": None,
        }
    expected = n_gen - 1
    return {
        "status": "MATCHES_SOURCE" if n_dec == expected else "DOES_NOT_MATCH_SOURCE",
        "generated_tokens": n_gen,
        "decode_steps": n_dec,
        "expected_decode_steps_from_source": expected,
        "off_by_one": n_dec == expected,
        "why": (
            f"{GENERATE_SRC}: decode_steps = tokens.len() - prompt_len - 1; "
            "the first new token is emitted by the last prefill step"
        ),
        "in_complete_numerator": n_gen,
        "in_decode_numerator": n_dec,
    }


def complete_minus_decode(reply: Mapping[str, Any] | None) -> dict[str, Any]:
    """Where the complete-token time goes, from the four clocks alone."""
    decomp = decompose_reply(reply)
    accounting = first_token_accounting(reply)
    request = decomp["clocks"]["request_ns"]
    decode = decomp["clocks"]["decode_ns"]
    prefill = decomp["clocks"]["prefill_ns"]
    ceremony = next(b for b in decomp["buckets"] if b["id"] == "request_ceremony")
    unattr = decomp["unattributed"]
    # The complete-vs-decode gap in the denominator is everything that is
    # not decode_ns. Numerator off-by-one is reported separately.
    denom_gap_ns = request - decode
    return {
        "decomposition": decomp,
        "first_token_accounting": accounting,
        "denominator_gap_ns": denom_gap_ns,
        "denominator_gap_is": [
            {"id": "prefill", "ns": prefill},
            {"id": "request_ceremony", "ns": ceremony["ns"]},
            {"id": "unattributed", "ns": unattr["ns"], "label": UNATTRIBUTED},
        ],
        "what_this_does_not_do": (
            "does not convert these clocks into a field named tps / "
            "wall_ns / gpu_ns; a speedup claim is refused without a dirty pair"
        ),
    }


def tallest_in_decomposition(decomp: Mapping[str, Any]) -> dict[str, Any]:
    """Name the largest bucket. UNATTRIBUTED dominating refuses a winner.

    Prefill is the usual mass in complete-minus-decode. It is necessary
    model work unless a batched prompt walk is legal — which this module
    does not establish (DeltaNet recurrent state). Request ceremony is the
    tallest *removable host* cost that the source actually names. A WOULD
    is still refused.
    """
    rows: list[dict[str, Any]] = []
    for b in decomp.get("buckets") or []:
        ns = b.get("ns")
        if not isinstance(ns, int):
            continue
        rows.append({"id": b["id"], "ns": ns, "label": b.get("label"), "kind": "named"})
    un = decomp.get("unattributed") or {}
    un_ns = un.get("ns")
    if isinstance(un_ns, int) and un_ns > 0:
        rows.append({"id": "unattributed", "ns": un_ns, "label": UNATTRIBUTED, "kind": UNATTRIBUTED})
    if not rows:
        raise GapRefuse("no buckets to rank")
    rows.sort(key=lambda r: (-int(r["ns"]), str(r["id"])))
    leader = rows[0]
    unattr_mass = un_ns if isinstance(un_ns, int) and un_ns > 0 else 0
    if leader["kind"] == UNATTRIBUTED or unattr_mass > max(
        (r["ns"] for r in rows if r["kind"] != UNATTRIBUTED), default=-1
    ):
        return {
            "named": False,
            "winner": None,
            "reason": (
                "UNATTRIBUTED dominates or leads; naming prefill or decode "
                "would fold a residue. Measure the leftover. Do not guess."
            ),
            "ranking": rows,
            "does_not_claim_would_improve": True,
        }
    removable = {
        "prefill": {
            "removable": UNKNOWN,
            "why": (
                "sequential session.step per prompt token is the mass of "
                "complete-minus-decode on short prompts. Whether a batched "
                "prefill is legal given DeltaNet recurrent state is not "
                "established here. A definition change (report decode-rate "
                "instead of complete) is not a speedup."
            ),
        },
        "decode": {
            "removable": False,
            "why": (
                "decode is the organ the 34.0 number already measured as "
                "1/median gpu step. Campaign record: bandwidth-bound; "
                "execution tuning nearly spent. Not this lane's target."
            ),
        },
        "request_ceremony": {
            "removable": True,
            "why": (
                "inner session.reset + vec alloc sit inside complete_tps's "
                "denominator and are not model work. Removal: do not reset "
                "twice; start the resident Instant after generate_greedy's "
                "reset, or reset once. This is a HYPOTHESIS, not a WOULD."
            ),
        },
    }
    info = removable.get(str(leader["id"]), {"removable": UNKNOWN, "why": "unlisted bucket"})
    return {
        "named": True,
        "winner": leader["id"],
        "ns": leader["ns"],
        "removable": info["removable"],
        "why": info["why"],
        "status": HYPOTHESIS,
        "ranking": rows,
        "does_not_claim_would_improve": True,
        "tallest_removable_host_ceremony": "request_ceremony",
        "tallest_mass_in_complete_minus_decode": "prefill",
    }


# ---------------------------------------------------------------------------
# Q2 — is 34.0 recoverable? conditions, not a regression story
# ---------------------------------------------------------------------------


def recover_sealed_anchors(sealed: Mapping[str, Any] | None, how: str) -> dict[str, Any]:
    if not isinstance(sealed, Mapping):
        return {
            "recovered": False,
            "reason": f"{SEALED_REL} {how}: sealed profile absent; not inferred",
            "historical": None,
            "current": None,
        }
    runtime = sealed.get("current_runtime") if isinstance(sealed.get("current_runtime"), Mapping) else {}
    hist = runtime.get("complete_tps_historical_qualified")
    cur = runtime.get("complete_tps_current_measured")
    if not isinstance(hist, (int, float)) or not isinstance(cur, (int, float)):
        return {
            "recovered": False,
            "reason": "current_runtime missing the two anchor fields; not inferred",
            "historical": None,
            "current": None,
        }
    fusion = sealed.get("fusion_env") if isinstance(sealed.get("fusion_env"), Mapping) else {}
    return {
        "recovered": True,
        "source": SEALED_REL,
        "loaded_from": how,
        "historical": {
            "recorded_tokens_per_second": hist,
            "profile_field": "current_runtime.complete_tps_historical_qualified",
            "role": "HISTORICAL_RECORD",
        },
        "current": {
            "recorded_tokens_per_second": cur,
            "profile_field": "current_runtime.complete_tps_current_measured",
            "role": "PROFILE_ANCHOR_NOT_A_PAIRED_AB",
        },
        "require_fusion_env": sealed.get("require_fusion_env"),
        "fusion_env": {k: fusion.get(k) for k in FUSION_KEYS},
        "executable_profile": sealed.get("executable_profile"),
        "resident_binary": sealed.get("resident_binary"),
        "binary": sealed.get("binary"),
        "max_seq_len": sealed.get("max_seq_len"),
        "prompt_contract": sealed.get("prompt_contract"),
        "evidence_class": "HISTORICAL_RECORD_NOT_A_SIDECAR_MEASUREMENT",
        "do_not_promote": True,
    }


def recover_historical_34(qual: Mapping[str, Any] | None, how: str) -> dict[str, Any]:
    """What the 34.0 actually was, from the qualification receipt.

    tools/odyssey/performance_qualification.py:
        single_stream_tps = round(1e9 / median(gpu_ns_per_step), 4)
    Default --max-new 24. Binary: ascension_qwen38_hybrid_greedy, one process
    per run. Prompt: the compiler-for-loop paragraph. Protected window with
    a standing contamination floor. Dispatches per step: 964 (unfused
    baseline), not the sealed 628 fused graph.
    """
    if not isinstance(qual, Mapping):
        return {
            "recovered": False,
            "quantity": UNKNOWN,
            "reason": f"{QUAL_REL} {how}: qualification receipt absent; not inferred",
        }
    body = (qual.get("bodies") or {}).get("sealed-3.14") if isinstance(qual.get("bodies"), Mapping) else None
    if not isinstance(body, Mapping):
        return {
            "recovered": False,
            "quantity": UNKNOWN,
            "reason": f"{QUAL_REL}: bodies.sealed-3.14 absent; not inferred",
        }
    latency = body.get("latency_vector") if isinstance(body.get("latency_vector"), Mapping) else {}
    tpot = latency.get("TPOT_ns_median")
    single = latency.get("single_stream_tps")
    runs = body.get("runs") if isinstance(body.get("runs"), list) else []
    dispatch_sets: list[int] = []
    prefill_ns: list[int] = []
    decode_ns: list[int] = []
    n_new: list[int] = []
    decode_steps: list[int] = []
    n_gpu_steps: list[int] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        if isinstance(run.get("prefill_wall_ns"), (int, float)):
            prefill_ns.append(int(run["prefill_wall_ns"]))
        if isinstance(run.get("decode_wall_ns"), (int, float)):
            decode_ns.append(int(run["decode_wall_ns"]))
        if isinstance(run.get("n_new_tokens"), int):
            n_new.append(run["n_new_tokens"])
        if isinstance(run.get("decode_steps"), int):
            decode_steps.append(run["decode_steps"])
        steps = run.get("gpu_ns_per_step") or []
        if isinstance(steps, list):
            n_gpu_steps.append(len(steps))
        disp = run.get("dispatches_per_step") or []
        if isinstance(disp, list) and disp:
            try:
                dispatch_sets.append(int(disp[0]))
            except (TypeError, ValueError):
                continue
    prompt_len = None
    if n_gpu_steps and decode_steps and len(n_gpu_steps) == len(decode_steps):
        inferred = n_gpu_steps[0] - decode_steps[0]
        if inferred > 0:
            prompt_len = inferred
    quantity = "inverse_median_gpu_ns_per_step"
    return {
        "recovered": True,
        "source": QUAL_REL,
        "loaded_from": how,
        "generated_at": qual.get("generated_at"),
        "generated_by": qual.get("generated_by"),
        "git_head": qual.get("git_head"),
        "quantity": quantity,
        "quantity_formula": "1e9 / median(gpu_ns_per_step across warm runs)",
        "recorded_single_stream_tokens_per_second": single,
        "tpot_ns_median": tpot,
        "prefill_ns_median": latency.get("prefill_ns_median"),
        "matches_sealed_34": (
            isinstance(single, (int, float)) and abs(float(single) - 34.0) < 0.2
        ),
        "binary": HYBRID_BIN,
        "prompt": (
            "Explain, in ordinary prose and at length, how a compiler turns a "
            "for-loop into basic blocks and then into machine code."
        ),
        "prompt_source": "tools/odyssey/performance_qualification.py PROMPT",
        "max_new_tokens_default": 24,
        "n_new_tokens_observed": sorted(set(n_new)),
        "decode_steps_observed": sorted(set(decode_steps)),
        "prompt_tokens_inferred_from_step_count": prompt_len,
        "dispatches_per_step_observed": sorted(set(dispatch_sets)),
        "fusion_graph": (
            "UNFUSED_BASELINE_964"
            if dispatch_sets and all(d == UNFUSED_DISPATCHES for d in dispatch_sets)
            else UNKNOWN
        ),
        "protected_window": bool((qual.get("protected_window") or {}).get("open")),
        "quiesced_means": (qual.get("quiesce_check_before") or {}).get("quiesced_means"),
        "contamination_floor_cpu_percent": (qual.get("contamination_floor") or {}).get(
            "total_cpu_percent"
        ),
        "executable_profile": "release-fast",
        "n_reps": body.get("n_reps"),
        "generation_clocks_from_runs": {
            "prefill_ns_by_run": prefill_ns,
            "decode_ns_by_run": decode_ns,
            "note": (
                "process wall_s of hybrid_greedy includes load and is not "
                "generation_ns. Residue of process wall minus prefill minus "
                "decode is UNATTRIBUTED (startup/load), not folded."
            ),
        },
        "not_complete_tps": True,
        "not_a_sidecar_measurement": True,
    }


def recover_live_probe(probe: Mapping[str, Any] | None, how: str) -> dict[str, Any]:
    """The live-probe receipt records generation, not clocks."""
    if not isinstance(probe, Mapping):
        return {
            "recovered": False,
            "reason": f"{PROBE_REL} {how}: probe receipt absent",
            "has_four_clocks": False,
        }
    observed = probe.get("observed") if isinstance(probe.get("observed"), Mapping) else {}
    r2 = observed.get("r2") if isinstance(observed.get("r2"), Mapping) else {}
    has = all(
        k in r2 for k in (CLOCK_PREFILL, CLOCK_DECODE, CLOCK_GENERATION, "wall_ns")
    )
    return {
        "recovered": True,
        "source": PROBE_REL,
        "loaded_from": how,
        "verdict": probe.get("verdict"),
        "evidence_class": probe.get("evidence_class"),
        "lease_taken": probe.get("lease_taken"),
        "r2_prompt": r2.get("prompt"),
        "r2_generated_tokens": r2.get("generated_tokens"),
        "has_four_clocks": bool(has),
        "reason_if_no_clocks": (
            None
            if has
            else (
                "RESIDENT_LIVE_PROBE.json records generated_text and "
                "generated_tokens only; prefill_wall_ns / decode_wall_ns / "
                "generation_wall_ns / wall_ns are absent. The lane brief's "
                "~35.5 / ~27.2 are a citation, not this receipt."
            )
        ),
        "lane_brief_citation": {
            "decode_tokens_per_second_approx": 35.5,
            "complete_tokens_per_second_approx": 27.2,
            "prompt_tokens_claimed": 13,
            "generated_tokens_claimed": 40,
            "provenance": "LANE_BRIEF",
            "in_probe_receipt": False,
            "status": HYPOTHESIS,
            "this_module_did_not_remeasure": True,
        },
    }


def condition_diff(
    sealed: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """What differs today vs the 34.0 measurement. Differ ≠ caused the gap."""
    rows: list[dict[str, Any]] = []

    def add(axis: str, historical_v: Any, today_v: Any, plausibly_accounts: str) -> None:
        same = historical_v == today_v
        rows.append(
            {
                "axis": axis,
                "historical_34": historical_v,
                "today": today_v,
                "differs": not same,
                "plausibly_accounts_for_34_vs_24": plausibly_accounts,
                "status": HYPOTHESIS if not same else "SAME",
            }
        )

    hist_quantity = historical.get("quantity") if historical.get("recovered") else UNKNOWN
    add(
        "quantity",
        hist_quantity,
        "profile field labelled complete_tps_current_measured",
        "YES — different denominators are sufficient to produce 34 vs 24 "
        "with no kernel change. Not a regression until the same quantity "
        "is compared.",
    )
    add(
        "binary",
        historical.get("binary") if historical.get("recovered") else UNKNOWN,
        RESIDENT_BIN,
        "UNKNOWN — hybrid_greedy vs resident. Process-wall of hybrid_greedy "
        "includes load; resident amortizes load. 34.0 did not use process-wall.",
    )
    add(
        "fusion_dispatches_per_step",
        historical.get("dispatches_per_step_observed") if historical.get("recovered") else UNKNOWN,
        FUSED_DISPATCHES,
        "UNKNOWN — 964 vs 628. Campaign already measured that a dispatch is "
        "not a unit of cost. Decode-rate ~34 on both graphs is the expected "
        "shape, not a fusion regression.",
    )
    add(
        "executable_profile",
        historical.get("executable_profile") if historical.get("recovered") else UNKNOWN,
        sealed.get("executable_profile"),
        "NO — both release-fast.",
    )
    add(
        "prompt_shape",
        {
            "text": historical.get("prompt") if historical.get("recovered") else UNKNOWN,
            "prompt_tokens_inferred": historical.get("prompt_tokens_inferred_from_step_count"),
        },
        sealed.get("prompt_contract"),
        "YES for complete_tps (prefill amortization), NO for 1/TPOT. The "
        "34.0 quantity is 1/TPOT, so prompt length does not move it.",
    )
    add(
        "sequence_length_new_tokens",
        historical.get("n_new_tokens_observed") if historical.get("recovered") else UNKNOWN,
        "profile max_new_tokens 2048; live probe r2 generated_tokens=40",
        "YES for complete_tps, NO for 1/TPOT. Same as prompt_shape.",
    )
    add(
        "contamination",
        {
            "protected_window": historical.get("protected_window") if historical.get("recovered") else UNKNOWN,
            "floor_cpu_percent": historical.get("contamination_floor_cpu_percent"),
        },
        "this host is contaminated; this sidecar holds no lease",
        "UNKNOWN — 34.0 already carried a standing floor (fileproviderd / "
        "WindowServer / avconferenced). Dirt today is not by itself a "
        "regression of a TPOT inverse.",
    )
    add(
        "max_seq_len",
        "hybrid_greedy --max-seq-len = max_new+64 (performance_qualification.py)",
        sealed.get("max_seq_len"),
        "UNKNOWN — not shown to move TPOT on this body.",
    )
    return rows


def recoverability_verdict(
    historical: Mapping[str, Any],
    diffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Do not assume regression. Do not assume 34.0 was wrong. UNKNOWN if unsure."""
    if not historical.get("recovered"):
        return {
            "verdict": UNKNOWN,
            "reason": (
                "conditions behind the historical 34.0 cannot be recovered "
                "from receipts; not inferred as a regression"
            ),
            "assumed_regression": False,
            "assumed_number_was_wrong": False,
        }
    quantity_row = next((d for d in diffs if d.get("axis") == "quantity"), None)
    if quantity_row and quantity_row.get("differs"):
        return {
            "verdict": UNKNOWN,
            "reason": (
                "34.0 is 1/median(gpu_ns_per_step) from "
                f"{QUAL_REL} (single_stream_tps="
                f"{historical.get('recorded_single_stream_tokens_per_second')}). "
                "24.4086 is a profile field labelled complete. Comparing them "
                "as a TPS regression mixes denominators. Recoverability of "
                "34.0 *complete* is UNKNOWN because 34.0 was not complete. "
                "Recoverability of 34.0 *as 1/TPOT* is a different question "
                "and is consistent with the lane-brief decode ~35.5, which "
                "this module did not remeasure."
            ),
            "assumed_regression": False,
            "assumed_number_was_wrong": False,
            "same_denominator_remeasure": "not taken; this sidecar has no lease",
            "plausible_accounts": [
                d["axis"]
                for d in diffs
                if d.get("differs") and str(d.get("plausibly_accounts_for_34_vs_24", "")).startswith("YES")
            ],
        }
    return {
        "verdict": UNKNOWN,
        "reason": "same quantity not established on both sides",
        "assumed_regression": False,
        "assumed_number_was_wrong": False,
    }


def process_wall_residue(historical: Mapping[str, Any]) -> dict[str, Any]:
    """hybrid_greedy process wall minus prefill minus decode is UNATTRIBUTED."""
    clocks = (historical.get("generation_clocks_from_runs") or {}) if historical.get("recovered") else {}
    prefill = clocks.get("prefill_ns_by_run") or []
    decode = clocks.get("decode_ns_by_run") or []
    if not prefill or not decode or len(prefill) != len(decode):
        return {
            "label": UNATTRIBUTED,
            "reason": "run clocks incomplete; residue not invented",
            "ns": None,
        }
    # Use the median pair. Process wall_s is not recovered into this function
    # on purpose: folding a Python subprocess wall into generation is the
    # mistake. We only report generation-internal residue here as zero-check.
    residues = [int(p) + int(d) for p, d in zip(prefill, decode)]
    return {
        "label": UNATTRIBUTED,
        "generation_named_sum_ns_by_run": residues,
        "folded_into_prefill": False,
        "folded_into_decode": False,
        "note": clocks.get("note"),
    }


def emit_workunit(tallest: Mapping[str, Any], recover: Mapping[str, Any]) -> dict[str, Any]:
    """One CPU_ANALYSIS unit. Attribution in, campaign out. Then stop."""
    return {
        "id": "WU.PROFILE_HOST_CEREMONY.tps_gap",
        "species": "PROFILE_HOST_CEREMONY",
        "module": "tools/future/tps_gap.py",
        "hypothesis": (
            "complete-minus-decode is sequential prefill (plus n_new vs "
            "n_new-1) sitting in the complete denominator; the tallest "
            "removable *host* cost the source names is the inner "
            "session.reset inside request_ns. 34.0 is 1/TPOT, not complete. "
            "Phase I: same-denominator dirty clocks, or batched-prefill "
            "legality given DeltaNet. The Odyssey does not wait on this."
        ),
        "frontier_item": "FT.LATENCY.complete-token",
        "output_contract": f"receipts/future/{RECEIPT}",
        "resource_class": "CPU_ANALYSIS",
        "allowed_authority": [
            "read_receipts",
            "write_sidecar_receipt",
            "emit_static_plan",
        ],
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "verifier": "tools/future/test_tps_gap.py",
        "stop_condition": (
            "receipt sealed; one WorkUnit emitted; then stop; no GPU lease; no second experiment"
        ),
        "status": HYPOTHESIS,
        "tallest": {
            "named": tallest.get("named"),
            "winner": tallest.get("winner"),
            "removable": tallest.get("removable"),
            "tallest_mass_in_complete_minus_decode": tallest.get(
                "tallest_mass_in_complete_minus_decode"
            ),
            "tallest_removable_host_ceremony": tallest.get(
                "tallest_removable_host_ceremony"
            ),
        },
        "recoverability": recover.get("verdict"),
        "does_not_claim_would_improve": True,
        "orchestration_bound": False,
        "handoff": "Phase I / PROFILE_HOST_CEREMONY on a dirty resident reply that actually carries the four clocks",
    }


def synthetic_reply(
    *,
    prefill_ns: int,
    decode_ns: int,
    generation_ns: int,
    request_ns: int,
    generated_tokens: int,
    decode_steps: int,
    prompt_tokens: int,
    complete_ns: int | None = None,
) -> dict[str, Any]:
    """A resident-shaped reply. Tests and the receipt self-check use this."""
    doc: dict[str, Any] = {
        CLOCK_PREFILL: prefill_ns,
        CLOCK_DECODE: decode_ns,
        CLOCK_GENERATION: generation_ns,
        "wall_ns": request_ns,
        "generated_tokens": generated_tokens,
        "decode_steps": decode_steps,
        "prompt_tokens": prompt_tokens,
    }
    if complete_ns is not None:
        doc["metrics"] = {"complete_wall_ns": complete_ns}
    return doc


def historical_generation_reply(historical: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a four-clock reply from a qualification run when request=generation.

    hybrid_greedy did not wrap generate_greedy in a second Instant, so
    request_ns is taken equal to prefill+decode (glue=0) for the
    *generation-internal* view. That is a stated assumption, and the
    process-wall residue stays UNATTRIBUTED elsewhere.
    """
    clocks = (historical.get("generation_clocks_from_runs") or {}) if historical.get("recovered") else {}
    prefill = clocks.get("prefill_ns_by_run") or []
    decode = clocks.get("decode_ns_by_run") or []
    n_new = historical.get("n_new_tokens_observed") or []
    d_steps = historical.get("decode_steps_observed") or []
    if not (prefill and decode and n_new and d_steps):
        return None
    # Median run by prefill (warm1-4 cluster; warm0 is the cold first).
    pairs = sorted(zip(prefill, decode), key=lambda pd: pd[0])
    mid = pairs[len(pairs) // 2]
    p_ns, d_ns = int(mid[0]), int(mid[1])
    gen = p_ns + d_ns
    prompt = historical.get("prompt_tokens_inferred_from_step_count") or 0
    return synthetic_reply(
        prefill_ns=p_ns,
        decode_ns=d_ns,
        generation_ns=gen,
        request_ns=gen,
        generated_tokens=int(n_new[0]),
        decode_steps=int(d_steps[0]),
        prompt_tokens=int(prompt),
    )


def build() -> Any:
    sealed_how, sealed = load_authority(SEALED_REL)
    qual_how, qual = load_authority(QUAL_REL)
    probe_how, probe = load_authority(PROBE_REL)
    hist_id_how, hist_id = load_authority(HISTORICAL_REL)
    smoke_how, smoke = load_authority(SMOKE_REL)
    regress_how, regress = load_authority(REGRESSION_REL)

    anchors = recover_sealed_anchors(sealed, sealed_how)
    historical = recover_historical_34(qual, qual_how)
    live = recover_live_probe(probe, probe_how)
    diffs = condition_diff(anchors if anchors.get("recovered") else {}, historical)
    recover = recoverability_verdict(historical, diffs)

    hist_reply = historical_generation_reply(historical)
    if hist_reply is None:
        hist_decomp = {
            "status": UNKNOWN,
            "reason": "qualification runs did not yield prefill/decode clocks",
        }
        hist_tallest: dict[str, Any] = {
            "named": False,
            "winner": None,
            "reason": "no generation clocks",
            "does_not_claim_would_improve": True,
        }
        hist_gap: dict[str, Any] = {"status": UNKNOWN}
    else:
        hist_gap = complete_minus_decode(hist_reply)
        hist_decomp = hist_gap["decomposition"]
        hist_tallest = tallest_in_decomposition(hist_decomp)

    # Live four-clock reply: only if the probe actually carried them.
    live_gap: dict[str, Any]
    if live.get("has_four_clocks") and isinstance(probe, Mapping):
        live_gap = complete_minus_decode(probe.get("observed") or probe)
    else:
        live_gap = {
            "status": UNKNOWN,
            "reason": live.get("reason_if_no_clocks")
            or "no live four-clock reply; this module did not start a resident",
            "this_module_did_not_run_the_resident": True,
            "why_not": (
                "a live Codex Accelerator campaign holds this host; starting "
                "ascension_qwen38_resident here would fight it. The probe "
                "receipt does not carry the four clocks. Fail closed."
            ),
        }

    proc_residue = process_wall_residue(historical)

    fused = None
    if isinstance(hist_id, Mapping):
        current = hist_id.get("current") if isinstance(hist_id.get("current"), Mapping) else {}
        fusion = current.get("fusion") if isinstance(current.get("fusion"), Mapping) else {}
        selected = (fusion.get("selected_graph") or {}).get("dispatch_consequence") or {}
        fused = {
            "source": HISTORICAL_REL,
            "loaded_from": hist_id_how,
            "baseline_source_derived": selected.get("baseline_source_derived"),
            "selected_source_derived": selected.get("selected_source_derived"),
        }

    smoke_result = None
    if isinstance(smoke, Mapping):
        result = smoke.get("result") if isinstance(smoke.get("result"), Mapping) else {}
        smoke_result = {
            "source": SMOKE_REL,
            "loaded_from": smoke_how,
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "performance_claim": result.get("performance_claim"),
            "note": (
                "native smoke recorded a complete_tokens_per_second on a 9-token "
                "completion; performance_claim=NONE. Not 24.4086 and not 34.0."
            ),
        }

    regression_finding = None
    if isinstance(regress, Mapping):
        hyps = ((regress.get("hypotheses") if isinstance(regress.get("hypotheses"), list) else []) or [])
        for h in hyps:
            if isinstance(h, Mapping) and h.get("id") == "H3-regression":
                regression_finding = {
                    "source": REGRESSION_REL,
                    "finding": h.get("finding"),
                    "this_module_closes": (
                        "attribution of the *quantities*: 34.0 is 1/TPOT, "
                        "24.4086 is a complete-labelled profile anchor. Cause "
                        "of any remaining same-denominator gap is still refused."
                    ),
                }
                break

    workunit = emit_workunit(hist_tallest, recover)

    self_timing = {
        "evidence_class": DIRTY,
        "what_this_block_is_not": [
            "not PROTECTED_ABSOLUTE",
            "not a qualified TPS",
            "not a ranking",
            "not a promotion input",
            "not a measurement this sidecar took on the GPU",
        ],
        "this_process_did_not_run_the_resident": True,
        "contamination": (
            "host is contaminated by construction; a live campaign is running; "
            "no bench lock was flocked; no machine was quiesced"
        ),
        "cited_generation_clocks": hist_decomp.get("clocks") if isinstance(hist_decomp, Mapping) else None,
        "cited_shares_of_request": hist_decomp.get("shares_of_request") if isinstance(hist_decomp, Mapping) else None,
        "process_wall_residue": proc_residue,
        "numbers_decide_nothing": True,
    }
    for key in self_timing:
        assert_timing_key_legal(str(key))

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Attribute complete-minus-decode from the resident's own clocks, "
            "recover what the historical 34.0 was measured under, and emit one "
            "WorkUnit for the tallest removable host cost. Then stop."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "question_1_complete_minus_decode": {
            "historical_generation_internal": hist_gap,
            "live_resident_reply": live_gap,
            "first_token_accounting_rule": (
                f"{GENERATE_SRC}: decode_steps = n_new - 1; complete uses n_new"
            ),
            "host_ceremony_from_source": [
                "inner session.reset + step-vec alloc (request_ns - generation_ns)",
                "inter-phase glue tokens.push + IGNORE_EOS env (UNATTRIBUTED leftover of generation)",
                "post tokenizer.decode of the full string (not in complete_tps)",
            ],
        },
        "question_2_is_34_recoverable": {
            "anchors": anchors,
            "historical_34": historical,
            "live_probe": live,
            "fused_graph_identity": fused,
            "native_smoke": smoke_result,
            "prior_gate": regression_finding,
            "condition_diff": diffs,
            "recoverability": recover,
        },
        "tallest": hist_tallest,
        "workunit": workunit,
        "self_timing": self_timing,
        "recovered_implementation": [
            f"{SEALED_REL} current_runtime.complete_tps_* anchors",
            f"{QUAL_REL} latency_vector.single_stream_tps = 1e9/median(gpu_ns_per_step)",
            f"{RESIDENT_SRC} four clocks per request",
            f"{GENERATE_SRC} generate_greedy Instants and decode_steps = n_new-1",
            "tools/future/qwen27_profile_schema.py load_authority / SEALED_REL / HISTORICAL_REL",
            "tools/future/_common.py write_receipt HARDWARE_FIELDS (raises)",
            "tools/future/dirty_measure.py SELF_MEASURED_DIRTY envelope (not forked)",
            "tools/future/accelerator_workunits.py PROFILE_HOST_CEREMONY species",
            "hcli/agentos/accelerator_regression.py H3-regression: gap recorded, cause refused",
            f"{DISPATCH_COST_REL} dispatch is not a unit of cost; bandwidth wall",
        ],
        "gaps_closed": [
            "complete-minus-decode attributed from the four resident clocks, with UNATTRIBUTED leftover",
            "34.0 recovered as inverse median gpu step from QWEN_PERFORMANCE_QUALIFICATION, not as complete_tps",
            "condition diff (quantity, binary, fusion dispatches, prompt, seq, contamination) without assuming regression",
            "one PROFILE_HOST_CEREMONY WorkUnit; campaign refused",
        ],
        "negative_findings": [
            "24.4086 measurement conditions are not in a receipt; hardcoded in hcli/hawking_native.py",
            "RESIDENT_LIVE_PROBE.json has no four clocks; ~35.5/~27.2 stay LANE_BRIEF citations",
            "this module did not run the resident and did not take a dirty pair",
            "batched prefill legality given DeltaNet is UNKNOWN",
            "UNATTRIBUTED process-wall residue of hybrid_greedy (load/startup) is not generation",
        ],
        "resident_callable": {
            "entry_point": "tools.future.tps_gap.decompose_reply(reply)",
            "workunit": workunit["id"] + "; one CPU_ANALYSIS unit; then stop",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.LATENCY.complete-token",
            "fails_closed": (
                "GapRefuse on missing clocks; residue is UNATTRIBUTED; "
                "WouldImproveRefuse without a labelled dirty pair; "
                "HardwareClaimError on tps/wall_ns/gpu_ns keys; "
                "recoverability is UNKNOWN rather than a guessed regression"
            ),
            "orchestration_bound": False,
        },
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
