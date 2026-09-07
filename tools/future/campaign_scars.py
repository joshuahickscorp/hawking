"""CAMPAIGN SCARS — seven numerical/procedural failures, kept as negative science.

Each one was caught by an independent check, not by the process that made it.
They are not representation hypotheses. They are the ways this campaign
laundered a narrow probe into a broad conclusion, mixed units, trusted a
falsy zero, and committed the index. Autonomous science that does not record
them will repeat them.

    python3 tools/future/campaign_scars.py --build

Registered in tools/future/negative_index.py SEED_SOURCES so refuse_if_dead
reaches them. The generic (observation, conclusion) checker lives in
status_causality.check_claim and is re-exported here.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any

from tools.future._common import write_receipt
from tools.verify.status_causality import (
    CAMPAIGN_CLAIM_CASES,
    CLAIM_CHECK_VERDICTS,
    CONTRADICTED,
    OVERREACHING,
    SUPPORTED,
    UNDERDETERMINED,
    check_campaign_claims,
    check_claim,
    campaign_claim_cases,
)

RECEIPT = "CAMPAIGN_SCARS.json"
SCHEMA = "hawking.future.campaign_scars.v1"
VERSION = 1
RECORDED_BY = "tools/future/campaign_scars.py"

# Re-export so tests and callers import one module for scars + claim check.
__all__ = (
    "SCARS",
    "RECEIPT",
    "SCHEMA",
    "SUPPORTED",
    "OVERREACHING",
    "UNDERDETERMINED",
    "CONTRADICTED",
    "CLAIM_CHECK_VERDICTS",
    "CAMPAIGN_CLAIM_CASES",
    "scars",
    "scar_ids",
    "build",
    "check_claim",
    "check_campaign_claims",
    "campaign_claim_cases",
)


SCARS: tuple[dict[str, Any], ...] = (
    {
        "id": "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR",
        "hypothesis_family": "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "Every *_per_generated_token field in the resident divides totals "
            "that span PREFILL AND DECODE by GENERATED tokens only. For the "
            "recorded run P=12, N=127, G=128 there are 139 forward passes and "
            "128 generated tokens, so the fields read (P+N)/G = 139/128 = "
            "1.085938x too high. dispatches_per_generated_token 1046.84375 "
            "equals 964 * 139/128 to the last digit; active_bytes_per_token "
            "10,727,793,881.75 is the catalog 9,878,901,136 inflated by the "
            "same factor (7 ppm residual, a 69 KB catalog/resident gap left open)."
        ),
        "wrongly_concluded": (
            "Those numbers are the per-token production costs. The clean-GEMV "
            "roof was written as 65.58 TPS and 71 TPS was moved above it. The "
            "same arithmetic L42 found in complete-vs-decode TPS was trusted "
            "a second time because the field was named per_token and the "
            "resident is the authority on its own ledger."
        ),
        "caught_by": (
            "An independent per-tensor catalog census "
            "(receipts/future/MLP_BYTE_CENSUS.json) that summed to "
            "9,878,901,136 without reference to the resident, and disagreed."
        ),
        "generalized_class": (
            "NUMERATOR_AND_DENOMINATOR_COUNT_DIFFERENT_EVENTS. A field named "
            "per_X is not per_X unless both sides count the same events."
        ),
        "cheapest_check": (
            "For every field whose name contains per_generated_token, assert "
            "the numerator's event set equals generated tokens (or divide the "
            "reported value by (P+N)/G and match an independent per-pass "
            "census / the decode-block). One mismatch is the class, not a field."
        ),
        "reopen_condition": (
            "a field whose numerator is decode-only (or generated-only) and "
            "whose denominator is generated tokens. Renaming the inflated "
            "fields to per_generated_token_including_prefill_work is a rename, "
            "not a reopen."
        ),
        "claim_refuted": (
            "that dividing prefill+decode totals by generated tokens yields a "
            "per-token production cost"
        ),
        "source_receipts": [
            "receipts/future/PER_GENERATED_TOKEN_INFLATION.json",
            "receipts/future/RESIDENT_TOKEN_BUDGET.json",
            "receipts/future/MLP_BYTE_CENSUS.json",
        ],
        "regression_test": (
            "tools/future/test_campaign_scars.py::"
            "test_prefill_over_generated_token_is_refuse_eligible"
        ),
    },
    {
        "id": "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
        "hypothesis_family": "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "A probe launched the resident without the sealed fusion env "
            "(HAWKING_QWEN38_FUSE_ADD_RMSNORM, FUSE_GQA_QKV, FUSE_DN_INPROJ, "
            "FUSE_MLP=swiglu) and measured 964 dispatches per decode step. "
            "The sealed-3.14 production graph with that env issues 628."
        ),
        "wrongly_concluded": (
            "964 is the production dispatch count. The author reported it as "
            "such (commit a94442cfe: 'corrects my own 964'). The 32.7 TPS "
            "reading was briefly treated as a correction of the 35.5 fusion "
            "anchor; it was the unfused arm."
        ),
        "caught_by": (
            "A static encode-path walk in tools/future/tps_budget.py that "
            "predicted 964 unfused and 628 fused, then a paired A/B on the "
            "same binary that returned exactly 964.00 and 628.00."
        ),
        "generalized_class": (
            "ENVIRONMENT_IS_PART_OF_EXPERIMENT_IDENTITY. A measurement whose "
            "env hash does not match the sealed config is a different "
            "experiment, not the incumbent."
        ),
        "cheapest_check": (
            "tools.future.artifact_identity.environment_identity hash over "
            "fusion flags, every HAWKING_* toggle, serving mode and "
            "measurement mode. A mismatch must not inherit the incumbent label."
        ),
        "reopen_condition": (
            "a measurement whose env_hash equals the sealed fusion env hash "
            "(including serving mode and measurement mode). An unfused probe "
            "that is labelled as unfused is not a reopen of this scar."
        ),
        "claim_refuted": (
            "that a probe of the unfused default graph is a production "
            "measurement of sealed-3.14"
        ),
        "source_receipts": [
            "receipts/future/RESIDENT_TOKEN_BUDGET.json",
            "receipts/future/ARTIFACT_IDENTITY.json",
        ],
        "regression_test": (
            "tools/future/test_artifact_identity.py::"
            "test_mismatched_env_does_not_inherit_incumbent_label"
        ),
    },
    {
        "id": "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE",
        "hypothesis_family": "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "The serving binary workspace/ops/build/rust/release-fast/examples/"
            "ascension_qwen38_resident was built 2026-08-26. The metrics block "
            "(dispatches, dispatches_per_generated_token, active_bytes_per_token, "
            "actual_read_bytes_per_token and kin) landed 2026-08-27 in "
            "8b6f50270. All eight fields are in the source and absent from "
            "the binary's string table; a live request returned none of them."
        ),
        "wrongly_concluded": (
            "The running resident can answer 'measure the CURRENT exact "
            "production dispatch count'. Source reality was treated as "
            "running-artifact reality."
        ),
        "caught_by": (
            "strings(binary) vs source, then a live probe returning none of "
            "the fields. receipts/future/RESIDENT_BINARY_DRIFT.json."
        ),
        "generalized_class": (
            "SOURCE_REALITY_IS_NOT_ARTIFACT_REALITY. A capability present in "
            "source is not a capability present in the running system."
        ),
        "cheapest_check": (
            "tools.future.artifact_identity.inspect_artifact, before any "
            "instrumentation is interpreted: record path/sha256/mtime/commit/"
            "dirty/flags/env/nx, and RAISE if the binary mtime predates the "
            "commit that introduced a field being read."
        ),
        "reopen_condition": (
            "a binary whose mtime is at or after the introducing commit of "
            "every field being read, and whose string table or live probe "
            "actually contains those fields. Rebuilding is the reopen, not "
            "re-reading the source."
        ),
        "claim_refuted": (
            "that instrumentation in source is instrumentation in the serving binary"
        ),
        "source_receipts": [
            "receipts/future/RESIDENT_BINARY_DRIFT.json",
            "receipts/future/ARTIFACT_IDENTITY.json",
        ],
        "regression_test": (
            "tools/future/test_artifact_identity.py::"
            "test_inspect_raises_when_binary_predates_field_commit"
        ),
    },
    {
        "id": "PRIORITY_ZERO_FALSY_OR_DEFAULT",
        "hypothesis_family": "PRIORITY_ZERO_FALSY_OR_DEFAULT",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "Kickoff ranked detachable jobs with `_detach_priority(j) or 99`. "
            "Long / composer-detached units have priority 0, which is falsy, "
            "so they were sent to the back of the queue and never started "
            "while overlap was being hunted."
        ),
        "wrongly_concluded": (
            "Priority 0 ranks first. The long jobs that were supposed to stay "
            "open while others ran were treated as missing priority."
        ),
        "caught_by": (
            "overlap_detached_work failing in the full suite after three of "
            "four 30m conditions had closed; inspection of rank_detachable. "
            "Commit ffccf71b2. Regression: "
            "test_long_detached_jobs_are_ranked_ahead_of_capabilities."
        ),
        "generalized_class": (
            "FALSY_ZERO_IS_A_VALUE. `x or default` is not a missing-check "
            "when 0 is a legitimate member of the domain."
        ),
        "cheapest_check": (
            "Write `x is None` (or a dedicated sentinel), never `x or default`, "
            "for a priority/count/index. A unit test that priority 0 ranks "
            "ahead of 2 would have caught this before a trial."
        ),
        "reopen_condition": (
            "never for the `prio or 99` form. A language or schema in which "
            "0 is not falsy does not reopen a Python `or`."
        ),
        "claim_refuted": (
            "that `_detach_priority(j) or 99` ranks priority-0 jobs first"
        ),
        "source_receipts": [
            "tools/future/autonomy_run.py",
            "tools/future/test_autonomy_run.py",
        ],
        "regression_test": (
            "tools/future/test_autonomy_run.py::"
            "test_long_detached_jobs_are_ranked_ahead_of_capabilities"
        ),
    },
    {
        "id": "EVENT_TIMESTAMP_UNIT_MISMATCH",
        "hypothesis_family": "EVENT_TIMESTAMP_UNIT_MISMATCH",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "started_at / finished_at are epoch seconds (e.g. 1788141337.2); "
            "t_s is trial-relative seconds (e.g. 35.0). A job cut off at the "
            "window boundary was emitted with no finished_at, so the overlap "
            "check fell back to t_s and subtracted mixed units. A real 2.34s "
            "overlap became a negative number."
        ),
        "wrongly_concluded": (
            "The two jobs did not overlap. The trial condition "
            "overlap_detached_work was scored unmet on a timeline that had "
            "a real same-unit overlap."
        ),
        "caught_by": (
            "The condition failing in the full suite while passing in "
            "isolation (an ordering-dependency shape). Tracing the interval "
            "arithmetic showed the mixed-unit subtraction. Commit ffccf71b2."
        ),
        "generalized_class": (
            "HETEROGENEOUS_UNITS_ARE_NOT_A_TIMELINE. Falling back across "
            "clocks is not a missing-value policy; it is a unit error."
        ),
        "cheapest_check": (
            "Refuse to subtract timestamps that do not share a unit tag. "
            "Never fall back from epoch to trial-relative (or wall to "
            "monotonic). A job with no finish stamp stays open or is tagged "
            "with a same-unit cutoff, never with a different clock."
        ),
        "reopen_condition": (
            "never for mixed-unit arithmetic. A schema that tags every stamp "
            "with its unit and rejects mixed subtraction is the guard, not a "
            "reopen of mixing them."
        ),
        "claim_refuted": (
            "that epoch started_at and trial-relative t_s may be used as "
            "ends of one interval"
        ),
        "source_receipts": [
            "tools/future/autonomy_trial.py",
            "tools/future/autonomy_run.py",
        ],
        "regression_test": (
            "tools/future/test_campaign_scars.py::"
            "test_mixed_timestamp_units_are_contradicted_as_no_overlap"
        ),
    },
    {
        "id": "ADJACENCY_IS_NOT_OVERLAP",
        "hypothesis_family": "ADJACENCY_IS_NOT_OVERLAP",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "The judge walked the event sequence and declared overlap as soon "
            "as two detached_started had been seen without their completions "
            "in between. Completions that arrived late, or omitted job_id, "
            "left the first job looking open. The driver hoped for overlap "
            "the same way: two adjacent starts were treated as two live pids."
        ),
        "wrongly_concluded": (
            "Two jobs were live at once. Sequential jobs (A 100-101, B 102-103) "
            "passed the adjacency rule and failed real interval arithmetic."
        ),
        "caught_by": (
            "A negative-control timeline of two provably sequential jobs that "
            "the old rule passed and the new one refuses "
            "(test_adjacent_starts_are_not_overlap_when_stamps_say_otherwise). "
            "The live check is two pids simultaneously not terminal."
        ),
        "generalized_class": (
            "ORDERING_IN_A_LOG_IS_NOT_SIMULTANEITY. Adjacency of start events "
            "is a statement about the log, not about two processes being live."
        ),
        "cheapest_check": (
            "Interval overlap on same-unit stamps, or two live pids from the "
            "supervisor. Adjacency of kinds is not sufficient. Emit "
            "overlap_confirmed only when poll reports >=2 not-terminal jobs."
        ),
        "reopen_condition": (
            "only if two live pids (or same-unit intervals with positive "
            "overlap) are shown. Two starts without a completion between "
            "them is never a reopen."
        ),
        "claim_refuted": (
            "that two detached_started without completions between them proves "
            "two jobs were live at once"
        ),
        "source_receipts": [
            "tools/future/autonomy_trial.py",
            "tools/future/test_autonomy_trial.py",
        ],
        "regression_test": (
            "tools/future/test_autonomy_trial.py::"
            "test_adjacent_starts_are_not_overlap_when_stamps_say_otherwise"
        ),
    },
    {
        "id": "SHARED_INDEX_BARE_COMMIT_SWEEPS_FOREIGN_STAGE",
        "hypothesis_family": "SHARED_INDEX_BARE_COMMIT_SWEEPS_FOREIGN_STAGE",
        "verdict": "FALSIFIED",
        "level": "GENERAL_PHYSICAL",
        "observed": (
            "fb4240dad, titled 'the resident supervisor is a loop, not a "
            "cycle', also carries tools/future/autonomy_run.py and "
            "test_autonomy_run.py from the m2 torture-wiring lane. "
            "`git apply --3way` had staged those files; a later "
            "`git add <supervisor paths> && git commit` took the whole index."
        ),
        "wrongly_concluded": (
            "A commit whose `git add` named supervisor paths contains only "
            "those paths. The index, not the add list, is what `git commit` "
            "without a pathspec writes."
        ),
        "caught_by": (
            "Comparing the commit tree to the message. Recorded in "
            "receipts/future/INDEX_PROVENANCE.json rather than rewritten. "
            "The same defect retired the legacy Odyssey driver the same day."
        ),
        "generalized_class": (
            "THE_INDEX_IS_THE_COMMIT_SET_UNLESS_A_PATHSPEC_SAYS_OTHERWISE. "
            "Any command that stages as a side effect (apply --3way, apply "
            "--index, stash pop on conflict, cherry-pick/merge on conflict) "
            "can put someone else's work under your message."
        ),
        "cheapest_check": (
            "`git commit -F <msg> -- <paths>` (pathspec, not bare). Before a "
            "bare commit, `git diff --cached --name-only` must equal the "
            "intended path list. Refuse a bare commit when the index has "
            "extra paths."
        ),
        "reopen_condition": (
            "never for a bare `git commit`. The pathspec form is the only "
            "safe one; an untracked new file must still be `git add`ed first."
        ),
        "claim_refuted": (
            "that `git add <paths> && git commit` commits only those paths "
            "when the index already holds a --3way apply"
        ),
        "source_receipts": [
            "receipts/future/INDEX_PROVENANCE.json",
        ],
        "regression_test": (
            "tools/future/test_campaign_scars.py::"
            "test_shared_index_scar_refuses_bare_commit_hypothesis"
        ),
    },
)


REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "observed",
    "wrongly_concluded",
    "caught_by",
    "generalized_class",
    "cheapest_check",
    "reopen_condition",
    "verdict",
    "hypothesis_family",
    "claim_refuted",
)


def scars() -> list[dict[str, Any]]:
    return [dict(s) for s in SCARS]


def scar_ids() -> list[str]:
    return [str(s["id"]) for s in SCARS]


def missing_fields() -> list[dict[str, str]]:
    out = []
    for scar in SCARS:
        for field in REQUIRED_FIELDS:
            if not str(scar.get(field) or "").strip():
                out.append({"id": str(scar.get("id")), "field": field})
    return out


def build() -> Path:
    missing = missing_fields()
    claims = check_campaign_claims()
    author_overreaches = [
        c for c in claims
        if c.get("author_was_the_one_who_concluded") and c["verdict"] == OVERREACHING
    ]
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Seven numerical and procedural failures from this campaign, each "
            "caught by an independent check, recorded so autonomous science "
            "stops repeating them. Negative science, not a changelog."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "n_scars": len(SCARS),
        "scar_ids": scar_ids(),
        "scars": scars(),
        "entries": scars(),
        "registered_in": [
            "tools/future/negative_index.py SEED_SOURCES",
            "receipts/future/CAMPAIGN_SCARS.json",
        ],
        "general_law": (
            "A STATUS MAY ASSERT ONLY WHAT ITS ACTUAL PROBE ESTABLISHES. "
            "Environment and the running artifact are part of experiment "
            "identity. Zero is a value. Adjacent is not overlapping. The "
            "index is the commit set."
        ),
        "claim_checker": {
            "entry_point": "tools.future.campaign_scars.check_claim",
            "implemented_in": "tools.verify.status_causality.check_claim",
            "verdicts": list(CLAIM_CHECK_VERDICTS),
            "n_seeded_cases": len(claims),
            "n_author_overreaches": len(author_overreaches),
            "author_overreach_ids": [c.get("id") for c in author_overreaches],
            "seeded_verdicts": [
                {"id": c.get("id"), "verdict": c["verdict"], "scar_id": c.get("scar_id")}
                for c in claims
            ],
        },
        "scars_missing_required_fields": missing,
        "recovered_implementation": [
            "tools/future/per_generated_token_inflation.py names the denominator class",
            "tools/future/resident_token_budget.py records the unfused-vs-sealed A/B",
            "tools/future/resident_binary_drift.py names source vs serving binary",
            "tools/future/autonomy_run.py::_detach_priority documents that 0 is a priority",
            "tools/future/autonomy_trial.py::_detached_overlap is interval arithmetic",
            "tools/future/index_provenance.py records the mixed commit fb4240dad",
            "tools/verify/status_causality.py already classified STATUS overreach; this extends it to (observation, conclusion)",
            "tools/future/artifact_identity.py is the refuse-on-stale-binary helper",
        ],
        "gaps_closed": [
            "the seven failures were each caught once and lived only in commit messages and sibling receipts",
            "refuse_if_dead could not see them: receipts/future was SKIP_PREFIXES",
            "no helper refused a binary older than the field it was about to interpret",
            "a mismatched env could inherit the incumbent label by silence",
            "the claim checker covered status labels, not (observation, conclusion)",
        ],
        "negative_findings": [
            "three of the overreaches were the author's own conclusions (964 as production, adjacency as overlap, bare commit as path list)",
            "the 2.34s overlap becoming negative was a unit error, not an absence of overlap",
            "priority 0 being falsy is a language trap that will recur anywhere `x or default` meets a ranked zero",
        ],
        "resident_callable": {
            "entry_point": "tools.future.campaign_scars.scars()",
            "claim_check": "tools.future.campaign_scars.check_claim",
            "workunit": "one CPU_ANALYSIS unit; consult before repeating a probe that produced one of these",
            "receipt": f"receipts/future/{RECEIPT}",
            "fails_closed": (
                "a scar missing a required field is reported; refuse_if_dead "
                "fires on these families; inspect_artifact raises rather than warns"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check-claim", metavar="ID")
    a = ap.parse_args()
    if a.check_claim:
        print(json.dumps(check_claim(case_id=a.check_claim), indent=1, sort_keys=True, default=str))
        return 0
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps({
        "n_scars": doc["n_scars"],
        "scar_ids": doc["scar_ids"],
        "claim_checker": {
            "n_seeded_cases": doc["claim_checker"]["n_seeded_cases"],
            "n_author_overreaches": doc["claim_checker"]["n_author_overreaches"],
        },
        "missing": doc["scars_missing_required_fields"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
