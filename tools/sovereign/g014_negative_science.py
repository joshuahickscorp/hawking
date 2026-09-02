#!/usr/bin/env python3
"""G014 producer: durable, scoped, reopenable negative science.

Two jobs, one run.

1. LAND the scars into the store the runtime already reads.
   `tools/future/negative_index.py` is that store: `ingest()` parses a corpus
   of negative-science sources into keyed `Scar` rows and `refuse_if_dead()`
   is consulted before a stage runs (tools/future/adaptive_verification.py
   `screen()`, tools/future/autonomy_trial.py, tools/future/scar_scheduling.py
   `admit`). A scar the index cannot see prunes nothing, so this producer
   writes the rows into a source the index actually loads
   (`receipts/future/SOVEREIGN_NEGATIVE_SCIENCE.json`, named in
   `negative_index.SEED_SOURCES`) and then PROVES the round trip: every family
   below must come back out of a fresh `ingest()` as a PARSED, refuse-eligible
   row, and `refuse_if_dead()` must actually refuse it. If that round trip
   fails this producer raises and writes nothing.

2. MEASURE whether compute was spent re-testing an already-scarred family.
   The number has to be a reading, so the screen carries its own denominator
   AND its own exclusions: every unit of executed work this repo can date and
   key to a family is run through `refuse_if_dead`, the seconds of the hits are
   summed, and every wall-clocked receipt that did NOT make the corpus is
   counted by reason in `excluded_from_corpus`. A denominator that reports only
   its survivors lets the parser pick the answer -- which is exactly how the
   first version of this producer reported 0.0s: it accepted only string
   timestamps and silently dropped the 88 receipts that write `finished_at` as
   a float epoch, six of which are real re-burns.

   CURRENT READING: 2.335s over 6 hits, all family `cross_expert_structure`,
   all run 2026-08-27 against a scar reachable since 2026-08-15. The G014 gate
   asserts == 0, so it is RED, and honestly so: this is an open finding, not a
   green light.

   The window matters and is deliberate. A scar can only prune work that
   started AFTER the scar was recorded and reachable; compute spent before the
   scar existed was not a re-burn, it was the discovery. Counting it would be
   the same defect as reading a boot high-water mark as a live reading. So a
   hit counts only when the scar's own source is strictly older than the work
   item's start, and never when the work item IS the scar's source.

   Corpus:
     - the live sovereign resident's DAG units (.hcli/dag.json) that carry
       both `running_at` and `finished_at` -- real seconds of real runtime,
       read-only;
     - every receipt under receipts/ carrying a positive wall clock
       (`wall_s` / `elapsed_s` / ...) and a timestamp in EITHER encoding this
       repo uses (ISO-8601 string or bare float epoch) -- each one is a unit of
       compute that was actually spent, keyed to a family by the index's own
       parser rather than by anything this file invents. Receipts with no
       parseable stamp, or that the index parser keys to no family, are
       excluded and COUNTED as excluded; they are unscreened, not clean.

    python3 tools/sovereign/g014_negative_science.py

Read-only against .hcli/ and against the resident. Writes exactly two files:
the store source and the gate receipt.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.future import adaptive_verification as av  # noqa: E402
from tools.future import negative_index as ni  # noqa: E402

RECEIPT = REPO_ROOT / "receipts" / "sovereign" / "G014_negative_science.json"
STORE = REPO_ROOT / "receipts" / "future" / "SOVEREIGN_NEGATIVE_SCIENCE.json"
STORE_REL = "receipts/future/SOVEREIGN_NEGATIVE_SCIENCE.json"

COMMAND = "python3 tools/sovereign/g014_negative_science.py"
PRODUCER = "tools/sovereign/g014_negative_science.py"

WALL_KEYS = ("wall_s", "elapsed_s", "wall_clock_s", "duration_s", "wall_seconds")
STAMP_KEYS = ("generated_at", "produced_at", "recorded_at", "created_at", "finished_at")


# ---------------------------------------------------------------------------
# The scars. Every one is a result this repo already paid for, cited to the
# receipt that carries the numbers, and scoped to what was actually measured.
# ---------------------------------------------------------------------------

SCARS: list[dict] = [
    {
        "causal_question": (
            "Does a per-tensor codec that is locally adequate on every MLP tensor "
            "compose into a whole model that still emits the teacher's next token?"
        ),
        "tested_family": "whole_model_composition_of_locally_adequate_mlp_codec",
        "organ": "mlp",
        "parent": "qwen3.8-27b-abliterated-bf16",
        "level": "MODEL_SPECIFIC",
        "evidence": (
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json "
            "rungs[complete_token_loop]: teacher_argmax=9714, student_argmax=10895, "
            "argmax_agree=false, final_hidden rel_l2=0.7359732389450073, survives=false, "
            "free_rel_l2_at_L63=0.7315157651901245. Two controls in the identical harness, "
            "same prompt and same 16 token ids: NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64.json "
            "argmax_agree=true (9714==9714, rel_l2=0.3423074185848236) and "
            "NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json argmax_agree=true "
            "(9714==9714, rel_l2=0.3471333980560303). The ternary run's own component rung "
            "passes locally (q4 gate_proj on real X: cosine 0.9965, gain 0.9963, survives=true), "
            "which is what makes this a COMPOSITION failure and not a codec failure."
        ),
        "reason_rejected": (
            "Local per-tensor adequacy does not compose. Ternary grouped-64 across every "
            "MLP tensor in all 64 layers changes the greedy next token while q3-g64 and "
            "q2f-g64 under the same harness do not. Error accumulation is measured, not "
            "assumed, and is NOT monotonic (14 recorded layer transitions move free-run "
            "rel_l2 DOWN), so a per-layer bound cannot be summed into a whole-model bound."
        ),
        "scope": (
            "qwen3.8-27b-abliterated-bf16 parent, MLP tensors only, ternary grouped-64, "
            "ONE 16-token prompt, ONE greedy token from a streamed 64-layer python loop on "
            "CPU. It condemns ternary-g64 ACROSS ALL MLP LAYERS OF THIS PARENT and nothing "
            "wider: not ternary on other organs, not other parents, not other group sizes, "
            "and not any single organ in isolation -- the same receipt has one organ "
            "(L0 down_proj, binary 1.015625 bpw) surviving the full loop."
        ),
        "reopen_if": (
            "a ternary-class MLP codec whose whole-model complete_token_loop reproduces the "
            "teacher argmax under this harness; or evidence that one greedy token from a "
            "16-token prompt is the wrong proxy (a multi-token generate disagreeing with it)."
        ),
        "evidence_tier": "measured_numerical",
        "physical_claim": False,
        "source_receipts": [
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64.json",
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
        ],
    },
    {
        "causal_question": (
            "Is there a coherent shared-basis MLP executable below q2f's 2.25 bpw that also "
            "beats q2f's complete-token time on the full 64-layer model?"
        ),
        # canon_family() maps "shared ... basis" onto the index's existing
        # `shared_basis` key, so this scar lands in the vocabulary the runtime
        # already queries rather than minting a private synonym.
        "tested_family": "shared basis mlp below q2f",
        "organ": "mlp",
        "parent": "qwen3.8-27b sealed-3.14",
        "level": "MODEL_SPECIFIC",
        "evidence": (
            "receipts/headless/SHARED_BASIS_COHERENT.json (obligation N035): "
            "coherent_shared_basis_beats_q2f=false; composition_ladder rung="
            "local_functional_probe status=FAILED died_at=held_out_activation, "
            "first_healthy_k_on_held_out=null for K in {2,4,8}, k2_first_unhealthy_layer=0; "
            "K=8 operating point active_bpw=2.125 IS below q2f but COMPLETE_TOKEN_NS="
            "59754833 against q2f_baseline complete_token_ns=27547874 (2.17x slower), "
            "median of 7 device reps (mlp_graph_gpu_ns 47828791..48117999). "
            "K=16 is 4.25 bpw, above q2f, so it cannot win density. "
            "The kernel is competent, not the excuse: parity rows match at max_abs_diff=0.0 "
            "and the receipt records not_gaussian=true (fitted on real captured activations)."
        ),
        "reason_rejected": (
            "Both coupled frontiers stay at q2f. No K in {2,4,8} heals held-out real "
            "activation on the 64-layer joint fit, and the one point that IS denser than "
            "q2f is 2.17x slower per complete token. Density without coherence is not an "
            "executable, and fewer bits did not buy fewer ns."
        ),
        "scope": (
            "the MLP organ of qwen3.8-27b sealed-3.14 on this Apple M3 Ultra, shared-binary-"
            "basis family, K in {2,4,8} on a 64-layer JOINT fit. NOT the whole model's floor "
            "and NOT other organs -- attention, deltanet and embed floors are separately "
            "unmeasured. NOT a claim that the shared-basis kernel is incompetent; the fused "
            "kernel is competent. NOT a claim against a hybrid operator, which was never run."
        ),
        "reopen_if": (
            "a K, a protected-island subset, or a hybrid (binary bulk + shared-basis or "
            "sparse correction fused as ONE operator) that heals held_out_activation strictly "
            "below 2.25 bpw AND lands complete_token_ns below 27547874 on this machine; or a "
            "different organ, whose floor this scar does not speak for."
        ),
        "evidence_tier": "physical",
        "physical_claim": True,
        "source_receipts": ["receipts/headless/SHARED_BASIS_COHERENT.json"],
    },
    {
        "causal_question": (
            "Does weight-space cosine predict real-activation error, so a representation can "
            "be screened without ever touching real activations?"
        ),
        "tested_family": "weight space cosine as a proxy for real activation error",
        "organ": "attention",
        "parent": "qwen3-30b-a3b",
        "level": "FAMILY",
        "evidence": (
            "receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json "
            "(schema hawking.accelerator.receipt.v1, ACCEL-REPRESENTATION, pass=true): "
            "THE_ANSWER_IS_ORGAN_DEPENDENT -- on q_proj weight cosine ANTI-predicts "
            "real-activation error, Spearman -1.000 on the english and code token sets and "
            "-0.800 on digits and repeat; on the expert gate the same grader predicts at "
            "+1.000 on three token sets. Same representation (ws_rtn_q4_g64, 4.25 complete "
            "bpw), same specimens, same grader, opposite sign. Worked example, "
            "Qwen3-30B-A3B q_proj: weight_cosine 0.98815 and weight_rel_err 0.155195 against "
            "real_activation_rel_err 0.032677. The Gaussian-activation proxy fails on the "
            "same boundary: real/gaussian error ratio 0.623-1.008 on q_proj (the proxy "
            "overstates error by up to 1.6x) versus 0.932-1.031 on the expert gate."
        ),
        "reason_rejected": (
            "The proxy's sign is organ-dependent, so a screen built on it silently inverts "
            "the ranking on attention. Every fidelity number screened this way on q_proj "
            "ordered the candidates backwards; the failure is not scale-invariance, because "
            "scale-aware weight rel_err anti-predicts identically at Spearman -1.000."
        ),
        "scope": (
            "attention q_proj under ws_rtn_q4_g64 at layer 0, measured across three "
            "independent architecture groups (Qwen3-30B-A3B, Kimi-VL-A3B, Falcon-H1-7B). "
            "EXPLICITLY NOT the expert gate, where the same proxy predicts correctly and "
            "stays usable, and NOT a claim that weight cosine is useless. It condemns "
            "carrying an expert-gate cosine to attention, not the metric."
        ),
        "reopen_if": (
            "the Spearman sign re-measured against real activations on an organ other than "
            "attention q_proj, on a representation other than q4_g64, or at a layer other "
            "than 0 -- any of those is a different measurement, not a retry of this one."
        ),
        "evidence_tier": "measured_numerical",
        "physical_claim": False,
        "source_receipts": ["receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json"],
    },
    {
        "causal_question": (
            "Does a high draft acceptance rate imply a speedup for self-speculative decoding "
            "on this machine?"
        ),
        # canon_family() folds "speculative" onto the index's `spec_decode` key.
        "tested_family": "self speculative decoding acceptance rate",
        "organ": "whole_model",
        "parent": "qwen3-80b",
        "level": "MODEL_SPECIFIC",
        "evidence": (
            "receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json (schema "
            "hawking.nos.self_speculative.v1, commit 412668821393a164d135be86471cbd86b4356abf): "
            "verdict REFUTED ON COST, BEFORE ACCEPTANCE RATE ENTERS. At PERFECT acceptance "
            "K=4 is ratio_vs_baseline 1.2724 (slower); the optimistic variant still loses at "
            "1.1282. Measured tier costs, ms/token: q4 body 29.145, q3 verify 30.525, "
            "1-plane draft 23.378. Measured inputs: gemv_ms_at_q4 23.276, nongemv_ms 5.869, "
            "amortization_R4K4 2.57, ps_per_element q3_group64 0.85236 (re-run twice, 0.0% "
            "spread, after a contaminated run was found and CODEC_ALU_COST.json restored). "
            "the_recorded_trap: the campaign's earlier 87% acceptance measured 0.91x -- that "
            "result is explained by this cost ratio, not by the acceptance rate. "
            "DERIVATION NOTE: the tier costs and per-element costs are device measurements; "
            "the K-sweep is arithmetic over them, as the receipt's own "
            "`speculative_arithmetic` field states."
        ),
        "reason_rejected": (
            "Acceptance rate was never the binding variable; the draft-to-verify COST RATIO "
            "is, and on this machine it is 0.75 where it needs to be about 0.5 or lower. "
            "The codec only moves the GEMV term (25.568 -> 17.811 ms) while the 5.869 ms "
            "non-GEMV floor does not shrink with the representation at all, so no acceptance "
            "rate rescues a scheme that already loses at 100% acceptance."
        ),
        "scope": (
            "Matryoshka-prefix self-speculation on this Apple M3 Ultra, priced against this "
            "campaign's own measured tier costs. NOT a claim about speculative decoding in "
            "general, NOT about a draft of a different SHAPE (fewer layers or a smaller "
            "hidden, which Matryoshka prefixes do not provide and which was not measured), "
            "and NOT about a machine with a different non-GEMV floor."
        ),
        "reopen_if": (
            "a draft tier costing under 15.063 ms/token at K=4 on this machine; or any draft "
            "whose NON-GEMV cost falls too, not just its GEMV; or verify-side amortization "
            "better than 2.57x; or a draft of a different shape."
        ),
        "evidence_tier": "physical",
        "physical_claim": True,
        "source_receipts": ["receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json"],
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


EPOCH_FLOOR = 946684800.0  # 2000-01-01Z
EPOCH_CEIL = 4102444800.0  # 2100-01-01Z


def _parse_stamp(value) -> float | None:
    """Epoch seconds from an ISO-8601 stamp OR a numeric epoch, else None.

    Both encodings are in this repo. Accepting only the string one silently
    discarded 88 wall-clocked receipts and turned the re-burn total into an
    artifact of which encoding the parser happened to like. The resident DAG
    branch below already reads bare float epochs, so this is one convention
    applied on both paths instead of one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if EPOCH_FLOOR <= v <= EPOCH_CEIL else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _git_head() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 else ""


# ---------------------------------------------------------------------------
# 1. Land the scars in the store the runtime reads, and prove the round trip.
# ---------------------------------------------------------------------------


def store_rows() -> list[dict]:
    """SCARS in the shape negative_index._parse_landed_science_scars reads."""
    rows = []
    for s in SCARS:
        rows.append(
            {
                "family": s["tested_family"],
                "level": s["level"],
                "organ": s["organ"],
                "parent": s["parent"],
                "status": "MEASURED_NEGATIVE",
                "mechanism": s["reason_rejected"],
                "object": s["causal_question"],
                "reopen": s["reopen_if"],
                "not": s["scope"],
                "evidence_tier": s["evidence_tier"],
                "physical_claim": s["physical_claim"],
                "source_receipts": s["source_receipts"],
            }
        )
    return rows


def write_store() -> None:
    missing = [
        rel
        for s in SCARS
        for rel in s["source_receipts"]
        if not (REPO_ROOT / rel).is_file()
    ]
    if missing:
        raise SystemExit(
            "refusing to write: cited source receipts do not exist: " + ", ".join(missing)
        )
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(
        json.dumps(
            {
                "schema": "hawking.sovereign.negative_science.v1",
                "produced_by": PRODUCER,
                "produced_at": _now_iso(),
                "command": COMMAND,
                "purpose": (
                    "Scars landed into tools/future/negative_index.py so refuse_if_dead "
                    "can key them. Named in negative_index.SEED_SOURCES because "
                    "SKIP_PREFIXES excludes receipts/future/ from the discovery sweep."
                ),
                "scars": store_rows(),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def prove_round_trip() -> dict:
    """Every scar must come back out of the store, keyed and refusable."""
    if STORE_REL not in ni.SEED_SOURCES:
        raise SystemExit(
            f"{STORE_REL} is not in negative_index.SEED_SOURCES; the index cannot see it "
            "and a scar the index cannot see prunes nothing"
        )
    pool = ni.ingest(force=True)
    ours = [s for s in pool if s.source_path == STORE_REL]
    checks = []
    for scar in SCARS:
        family = scar["tested_family"]
        canon = ni.canon_family(family)
        row = next((s for s in ours if s.hypothesis_family == canon), None)
        if row is None:
            raise SystemExit(
                f"family {family!r} (canon {canon!r}) did not survive ingest() from {STORE_REL}"
            )
        if row.parse_status != ni.PARSED or not row.refuse_eligible:
            raise SystemExit(
                f"family {canon!r} ingested but is not refuse-eligible "
                f"(parse_status={row.parse_status}, refuse_eligible={row.refuse_eligible})"
            )
        refusal = ni.refuse_if_dead({"hypothesis_family": family}, pool)
        if not refusal or not refusal.get("refused"):
            raise SystemExit(f"refuse_if_dead did not refuse a landed scar: {canon!r}")
        # The call site, not the definition: run the real pre-stage screen the
        # runtime uses and require it to launch zero stages.
        live = av.screen({"id": f"g014_probe::{canon}", "hypothesis_family": family})
        if live.get("refused_by") != "negative_index" or live["cost"]["stages_executed"] != 0:
            raise SystemExit(
                f"adaptive_verification.screen did not refuse {canon!r} before any stage: "
                f"verdict={live.get('verdict')} stages={live['cost']['stages_executed']}"
            )
        checks.append(
            {
                "tested_family": family,
                "canon_family": canon,
                "scar_id": row.scar_id,
                "refuse_eligible": True,
                "refused_by_runtime_path": refusal["scar_id"],
                "level": row.level,
                "organ": row.organ,
                "model": row.model,
                "live_screen": {
                    "call_site": "tools/future/adaptive_verification.py::screen",
                    "verdict": live.get("verdict"),
                    "refused_by": live.get("refused_by"),
                    "stages_executed": live["cost"]["stages_executed"],
                },
            }
        )
    return {
        "store": STORE_REL,
        "index": "tools/future/negative_index.py",
        "seeded": True,
        "pool_size": len(pool),
        "scars_from_this_store": len(ours),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 2. Measure recomputed dead-family seconds against a real corpus.
# ---------------------------------------------------------------------------


def scar_source_epoch(rel: str, cache: dict) -> float | None:
    """When the scar first became reachable. None if it cannot be dated.

    The EARLIEST of the source's own recorded stamp, the commit that added it
    to the tree, and its mtime. Earliest on purpose: this date is the exclusion
    threshold, so the conservative choice is the one that makes a re-burn MORE
    likely to be counted, never less.
    """
    if rel in cache:
        return cache[rel]
    path = REPO_ROOT / rel
    candidates: list[float] = []
    if path.is_file() and path.suffix == ".json":
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                for k in STAMP_KEYS:
                    stamp = _parse_stamp(doc.get(k))
                    if stamp is not None:
                        candidates.append(stamp)
                        break
        except (OSError, ValueError):
            pass
    r = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%ct", "--", rel],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    added = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().isdigit()]
    if added:
        candidates.append(float(added[-1]))
    if path.is_file():
        try:
            candidates.append(path.stat().st_mtime)
        except OSError:
            pass
    cache[rel] = min(candidates) if candidates else None
    return cache[rel]


def work_items(excluded: dict | None = None) -> list[dict]:
    """Units of compute this repo can date, time, and key to a family.

    `excluded` is filled in place with the wall-clocked receipts this corpus
    does NOT admit, and why. A denominator that reports only its survivors
    hides the selection that produced its own answer.
    """
    items: list[dict] = []
    drop_no_stamp: list[str] = []
    drop_no_family: list[str] = []

    # (a) The live sovereign resident's own executed DAG units. Read-only.
    dag = REPO_ROOT / ".hcli" / "dag.json"
    if dag.is_file():
        try:
            units = json.loads(dag.read_text(encoding="utf-8")).get("units") or {}
        except (OSError, ValueError):
            units = {}
        for uid, unit in units.items():
            if not isinstance(unit, dict):
                continue
            start, end = unit.get("running_at"), unit.get("finished_at")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            seconds = float(end) - float(start)
            if seconds <= 0:
                continue
            items.append(
                {
                    "kind": "resident_dag_unit",
                    "id": str(uid),
                    "path": ".hcli/dag.json",
                    "started_at_epoch": float(start),
                    "seconds": seconds,
                    "families": [str(unit.get("description") or "")],
                }
            )

    # (b) Every receipt carrying a real wall clock. The family is derived by
    #     the index's OWN parser, not by anything this file invents.
    for abs_path in sorted(glob.glob(str(REPO_ROOT / "receipts" / "**" / "*.json"), recursive=True)):
        rel = os.path.relpath(abs_path, REPO_ROOT)
        try:
            doc = json.loads(Path(abs_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        seconds = None
        for k in WALL_KEYS:
            v = doc.get(k)
            if isinstance(v, (int, float)) and v > 0:
                seconds = float(v)
                break
        if seconds is None:
            continue
        end = None
        for k in STAMP_KEYS:
            end = _parse_stamp(doc.get(k))
            if end is not None:
                break
        if end is None:
            drop_no_stamp.append(rel)
            continue
        families = []
        try:
            for s in ni.parse_source(rel):
                if s.parse_status == ni.PARSED and s.hypothesis_family != ni.UNRECORDED:
                    families.append(s.hypothesis_family)
        except Exception:  # a source the parser chokes on is not a work item
            families = []
        if not families:
            drop_no_family.append(rel)
            continue
        items.append(
            {
                "kind": "receipt_with_wall_clock",
                "id": rel,
                "path": rel,
                "started_at_epoch": end - seconds,
                "seconds": seconds,
                "families": sorted(set(families)),
            }
        )
    if excluded is not None:
        admitted = sum(1 for i in items if i["kind"] == "receipt_with_wall_clock")
        excluded.update(
            {
                "wall_clocked_receipts_seen": admitted + len(drop_no_stamp) + len(drop_no_family),
                "admitted": admitted,
                "dropped_no_parseable_timestamp": len(drop_no_stamp),
                "dropped_no_family_from_the_index_parser": len(drop_no_family),
                "dropped_no_family_examples": sorted(drop_no_family)[:5],
                "dropped_no_parseable_timestamp_examples": sorted(drop_no_stamp)[:5],
                "meaning": (
                    "every receipt under receipts/ carrying a positive wall clock is counted "
                    "here. A drop is a receipt this screen could NOT key to a family or could "
                    "not date -- not evidence that no re-burn happened in it."
                ),
            }
        )
    return items


def screen_for_reburn(pool, items: list[dict] | None = None) -> dict:
    """Seconds actually spent re-testing a family that was ALREADY scarred.

    A hit needs all three: refuse_if_dead refuses the family; the refusing
    scar does not live in the work item's own file; and the scar's source
    predates the work item's start, because a scar cannot prune work that ran
    before it existed.
    """
    cache: dict = {}
    excluded: dict = {}
    items = work_items(excluded) if items is None else items
    hits = []
    total = 0.0
    screened_seconds = 0.0
    for item in items:
        screened_seconds += item["seconds"]
        for family in item["families"]:
            refusal = ni.refuse_if_dead({"hypothesis_family": family}, pool)
            if not refusal or not refusal.get("refused"):
                continue
            source = str(refusal.get("source_path") or "")
            if source == item["path"]:
                continue  # the work item IS the scar; discovery, not re-burn
            scar_epoch = scar_source_epoch(source, cache)
            if scar_epoch is None or scar_epoch >= item["started_at_epoch"]:
                continue  # the scar did not exist yet; rediscovery was free
            total += item["seconds"]
            hits.append(
                {
                    "work_item": item["id"],
                    "kind": item["kind"],
                    "seconds": item["seconds"],
                    "family": family,
                    "refused_by": refusal.get("scar_id"),
                    "scar_source": source,
                }
            )
            break
    return {
        "recomputed_dead_family_seconds": round(total, 6),
        "hits": hits,
        "denominator": {
            "work_items_screened": len(items),
            "seconds_of_executed_work_screened": round(screened_seconds, 3),
            "resident_dag_units": sum(1 for i in items if i["kind"] == "resident_dag_unit"),
            "receipts_with_wall_clock": sum(
                1 for i in items if i["kind"] == "receipt_with_wall_clock"
            ),
            "scars_in_pool": len(pool),
            "excluded_from_corpus": excluded,
            "meaning": (
                "this is seconds of executed work that re-tested a family already "
                "scarred when the work started, screened over this many dated, timed "
                "units against this many scars. It is a reading, not a written constant "
                "-- see detector_negative_control. What the corpus does NOT cover is in "
                "excluded_from_corpus; a receipt dropped there was not screened at all."
            ),
        },
        "window_rule": (
            "a hit counts only when the refusing scar's own source is strictly older than "
            "the work item's start, and never when the work item is that source. Compute "
            "spent before a scar existed was the discovery, not a re-burn."
        ),
    }


def detector_negative_control(pool) -> dict:
    """Prove the screen CAN return non-zero, so its zero is a reading.

    Same code path, same pool, one fabricated work item that really is in an
    already-scarred family and really did start after that scar was recorded.
    If this does not come back non-zero the screen is a dead branch and its
    zero over the real corpus means nothing.
    """
    family = SCARS[0]["tested_family"]
    seconds = 137.0
    probe = {
        "kind": "negative_control",
        "id": "synthetic::recompute_of_a_scarred_family",
        "path": "<synthetic, not a real file>",
        "started_at_epoch": time.time() + 86400.0,
        "seconds": seconds,
        "families": [family],
    }
    got = screen_for_reburn(pool, [probe])
    measured = got["recomputed_dead_family_seconds"]
    if measured != seconds:
        raise SystemExit(
            "detector negative control FAILED: a fabricated recompute of the scarred "
            f"family {family!r} scored {measured}s, expected {seconds}s. The screen is "
            "not measuring anything and its zero would be vacuous."
        )
    return {
        "ran": True,
        "fabricated_family": family,
        "fabricated_seconds": seconds,
        "screen_returned": measured,
        "detects_a_reburn": True,
        "note": (
            "the control is a synthetic work item passed to the same screen_for_reburn "
            "on the same scar pool; it is never written to the store or counted in the "
            "real measurement"
        ),
    }


def main() -> int:
    started = time.time()
    write_store()
    round_trip = prove_round_trip()
    pool = ni.ingest(force=True)
    control = detector_negative_control(pool)
    screen = screen_for_reburn(pool)

    doc = {
        "schema": "hcli.sovereign.negative_science.v1",
        "produced_by": PRODUCER,
        "produced_at": _now_iso(),
        "command": COMMAND,
        "git_head": _git_head(),
        "status": "MEASURED",
        "gate": "G014",
        "gate_result": (
            "GREEN"
            if screen["recomputed_dead_family_seconds"] == 0
            else "RED: recomputed_dead_family_seconds is non-zero, so "
            "test_dead_families_are_not_reburned fails. This is a real open finding "
            "-- compute WAS spent re-testing an already-scarred family -- not a "
            "producer defect. Do not zero it by narrowing the corpus."
        ),
        "store": {
            "path": STORE_REL,
            "index_module": "tools/future/negative_index.py",
            "runtime_consumers": [
                "tools/future/adaptive_verification.py::screen -> refuse_if_dead "
                "(consulted before any stage runs)",
                "tools/future/autonomy_trial.py -> refuse_if_dead",
                "tools/future/scar_scheduling.py::admit",
            ],
            "round_trip": round_trip,
        },
        "scars": SCARS,
        "recomputed_dead_family_seconds": screen["recomputed_dead_family_seconds"],
        "recompute_screen": screen,
        "detector_negative_control": control,
        "evidence": {
            "scars_landed": len(SCARS),
            "every_scar_refusable_from_the_runtime_path": True,
            "round_trip_checks": round_trip["checks"],
            "recompute_screen_denominator": screen["denominator"],
            "detector_negative_control": control,
        },
        "producer_wall_s": round(time.time() - started, 3),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT.relative_to(REPO_ROOT)}")
    print(f"wrote {STORE_REL}")
    print(
        "recomputed_dead_family_seconds="
        f"{screen['recomputed_dead_family_seconds']} over "
        f"{screen['denominator']['work_items_screened']} work items / "
        f"{screen['denominator']['seconds_of_executed_work_screened']}s screened"
    )
    for hit in screen["hits"]:
        print(f"  RE-BURN {hit['work_item']} {hit['seconds']}s family={hit['family']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
