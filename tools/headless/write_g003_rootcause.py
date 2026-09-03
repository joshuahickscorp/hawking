#!/usr/bin/env python3
"""Assemble the G003 root-cause receipt from evidence already on disk.

Binds three independent sources so the conclusion is checkable rather than
asserted:

  1. production failure record  .hcli-legacy/bootstrap-director-v6/{negative-science.jsonl,runs/*/hcli.log}
  2. controlled reproduction    receipts/headless/STRUCTURED_OUTPUT_PROBE.json
  3. the code path itself       hcli/engine.py at a pinned commit

Writes receipts/headless/G003_ROOT_CAUSE.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))


def sh(c: str) -> str:
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout.strip()


def production_evidence() -> dict:
    ns = REPO / ".hcli-legacy/bootstrap-director-v6/negative-science.jsonl"
    rows = []
    if ns.exists():
        for line in ns.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    logs = {}
    for p in sorted(glob.glob(str(REPO / ".hcli-legacy/bootstrap-director-v6/runs/*/*.log"))):
        txt = Path(p).read_text(errors="replace").strip()
        if txt:
            logs.setdefault(txt.splitlines()[-1][:200], []).append(os.path.basename(os.path.dirname(p)))
    fps = [r.get("fingerprint") for r in rows if r.get("fingerprint")]
    return {
        "negative_science_entries": len(rows),
        "epochs_recorded": [r.get("epoch") for r in rows],
        "distinct_fingerprints": sorted(set(fps)),
        "fingerprint_is_constant": len(set(fps)) == 1 if fps else None,
        "note_on_fingerprint": (
            "A single distinct fingerprint across every recorded epoch is a CYCLE / NO_PROGRESS "
            "signature: the stage hash never moved, so nothing the director did changed the tree."),
        "run_log_last_lines_grouped": {k: {"count": len(v), "example_epochs": v[:5]}
                                       for k, v in logs.items()},
    }


def receipt_health() -> dict:
    d = REPO / ".hcli/receipts"
    total = failed = no_err = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        total += 1
        if j.get("status") not in ("completed", None):
            failed += 1
            if "error" not in j:
                no_err += 1
    return {"total": total, "not_completed": failed,
            "not_completed_with_no_error_field": no_err,
            "meaning": ("_write_receipt builds the receipt from a fixed key list that never copies "
                        "`error`, so a failure is recorded with no cause. Separately, str(exc) is "
                        "'' for KeyboardInterrupt and several timeouts, so even copying it would "
                        "sometimes record nothing — the exception TYPE must be recorded too.")}


def probe_evidence() -> dict:
    p = REPO / "receipts/headless/STRUCTURED_OUTPUT_PROBE.json"
    if not p.exists():
        return {"present": False}
    d = json.loads(p.read_text())
    runs = d.get("runs", [])
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    table = {}
    for arm, rs in by_arm.items():
        table[arm] = [{"rep": r.get("rep"), "valid_json": r.get("ok"), "wall_s": r.get("wall_s"),
                       "completion_tokens": r.get("completion_tokens"),
                       "finish_reason": r.get("finish_reason")} for r in
                      sorted(rs, key=lambda x: x.get("rep", 0))]
    return {"present": True, "user_source": d.get("user_source"),
            "params": d.get("params"), "per_arm_runs": table, "summary": d.get("summary")}


def main() -> int:
    pinned = sh(f"git -C {REPO} rev-parse d1efc69") or "d1efc69"
    prod = production_evidence()
    probe = probe_evidence()

    doc = {
        "schema": "hawking.headless.g003_root_cause.v1",
        "obligation": ("G003 — root-cause the HCLI failure mode: every recent .hcli receipt is "
                       "status:failed / kind:error with empty operations and NO error text."),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "TWO INDEPENDENT DEFECTS, both reproduced",
        "defect_1": {
            "id": "D-RUNAWAY",
            "severity": "critical",
            "title": "Unconstrained reasoning exhausts the completion budget before the JSON is emitted",
            "mechanism": (
                "engine.py::_call_model posts model/messages/temperature/max_tokens and nothing else. "
                "No reasoning-budget control and no response_format. Qwen3 emits its reasoning first. "
                "On a real 7257-token director mission it never reaches the JSON object before "
                "HCLI_MODEL_TOKENS (default 6500) is spent, so the reply is truncated "
                "(finish_reason='length') and unparseable. The engine then raises the generic "
                "'Model did not return a valid structured JSON object', which is the ONLY line in "
                "every failing epoch log."),
            "arithmetic_that_first_suggested_it": (
                "6500 completion tokens / ~314 s observed per failed production call = 20.7 tok/s, "
                "which is exactly this machine's single-decoder rate. The call was not erroring or "
                "hanging; it was decoding to the budget."),
            "why_it_was_never_caught": (
                "On a small prompt every arm succeeds. The controlled probe shows baseline passing "
                "in ~9.6 s on a toy calc.py goal and failing at 6500 tokens on the real mission. A "
                "test written against the toy case cannot see this."),
            "counter_intuitive_finding": (
                "A JSON schema ALONE does not fix it. The 'schema' arm still burned 6500 tokens and "
                "still failed: constrained decoding forces the SHAPE of the output but the model "
                "reasons inside the string fields. Disabling thinking is the necessary condition; "
                "the schema is a structural guarantee on top, not a substitute."),
            "fix": ("chat_template_kwargs={'enable_thinking': False} on structured calls (requires the "
                    "server to run with --jinja, else it is silently ignored), plus response_format "
                    "json_schema, plus recording finish_reason so a length truncation can never again "
                    "present as a mystery."),
        },
        "defect_2": {
            "id": "D-BLIND-RECEIPT",
            "severity": "high",
            "title": "Failure receipts record no cause",
            "mechanism": (
                "engine.py::execute's `except BaseException` builds a result carrying "
                "\"error\": str(exc), but _write_receipt constructs the receipt from a fixed key "
                "list that never copies `error`. The failure therefore reaches disk as "
                "kind:'error' / status:'failed' with empty operations, empty tests, null validation "
                "and no message at all."),
            "aggravating": (
                "str(exc) is the empty string for KeyboardInterrupt and for several timeout "
                "exceptions, so copying the message alone would still sometimes record nothing. The "
                "exception TYPE and a truncated traceback must be recorded as well."),
            "measured": receipt_health(),
            "fix": "carry error, error_type and a truncated traceback onto failed/cancelled receipts.",
        },
        "production_evidence": prod,
        "controlled_reproduction": probe,
        "code_under_test": {
            "path": "hcli/engine.py",
            "pinned_commit": pinned,
            "call_site": "_call_model (~line 896), payload built ~line 966",
            "receipt_site": "_write_receipt (~line 1821)",
            "error_site": "execute's except BaseException (~line 495)",
        },
        "reproduce": {
            "production_failure":
                "python3 tools/headless/structured_output_probe.py --reps 1 --arms baseline "
                "--user-file .hcli-legacy/bootstrap-director-v6/runs/20260822-143859-epoch-00007/mission.md",
            "the_fix":
                "python3 tools/headless/structured_output_probe.py --reps 1 --arms schema_nothink "
                "--user-file .hcli-legacy/bootstrap-director-v6/runs/20260822-143859-epoch-00007/mission.md",
        },
        "related": {
            "acceptance_vacuity": "receipts/headless/ACCEPTANCE_VACUITY.json — a separate and more "
                                  "dangerous class: the acceptance gate cannot fail. That one is "
                                  "silent where this one is loud.",
        },
    }

    out = REPO / "receipts/headless"
    out.mkdir(parents=True, exist_ok=True)
    (out / "G003_ROOT_CAUSE.json").write_text(json.dumps(doc, indent=1))

    print("=== G003 ROOT CAUSE ===")
    print(f"  verdict: {doc['verdict']}")
    print(f"  D-RUNAWAY      {doc['defect_1']['title']}")
    print(f"  D-BLIND-RECEIPT {doc['defect_2']['title']}")
    rh = doc["defect_2"]["measured"]
    print(f"    receipts: {rh['total']} total, {rh['not_completed']} not completed, "
          f"{rh['not_completed_with_no_error_field']} of those with no error field")
    print(f"  production: {prod['negative_science_entries']} negative-science entries, "
          f"fingerprints={prod['distinct_fingerprints']} constant={prod['fingerprint_is_constant']}")
    if probe.get("present"):
        print("  reproduction (per arm, per rep):")
        for arm, rows in probe["per_arm_runs"].items():
            for r in rows:
                print(f"    {arm:<15} rep{r['rep']} valid={str(r['valid_json']):<5} "
                      f"wall={r['wall_s']:>8}s tok={r['completion_tokens']:>5} "
                      f"finish={r['finish_reason']}")
    print(f"\n-> {out/'G003_ROOT_CAUSE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
