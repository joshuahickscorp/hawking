#!/usr/bin/env python3.12
"""receipt_verify.py — validate a Hawking condensation receipt.

Two layers of checking, both required to PASS:

  1. JSON Schema validation against
     receipts/schema/condensation_receipt.schema.json (schema v0.2).
  2. The eight §20.3 invalidation rules, enforced in code, because a schema
     cannot express conditional/cross-field logic (e.g. "no public win below
     R3", "best-effort baseline cannot back a win").

The eight invalidation rules (docs/plans/studio_maximization_2026_06_27.md
§20.3, lines ~1624-1636 — a receipt that trips any of these gets gate=invalid
and DOES NOT COUNT; the reason is published):

  R1  effective_bpw missing/<=0, or only nominal_bpw reported.
  R2  quality is from a single window (multiwindow_n < 4) on a quality claim,
      or the mean is reported but the worst window is hidden.
  R3  PPL passes but kl_parent_condensed exceeds the warn band (ppl-theater).
  R4  no source_sha256 / artifact_sha256 (the artifact is not identifiable).
  R5  no commands or no hawking_commit (not reproducible).
  R6  a density-win claim where the Q4 baseline loaded "ok" but the receipt is
      labelled a cliff — or a cliff claim where Q4 loaded ok (mislabelled
      claim_type vs baseline behavior).
  R7  the MPS backend produced the HEADLINE number without a CPU-bf16 confirm.
  R8  baseline_best_effort == true is used to back a public WIN.

Plus the §20.6 master rule: NO PUBLIC WIN BELOW R3. A "win" here means a
density or cliff receipt whose quality_gate is pass/warn (i.e. it asserts a
positive result), tagged below R3.

Exit codes:
  0  receipt is valid (schema OK + no invalidation rule tripped)
  1  receipt is INVALID (schema or rule failure) — reasons printed
  2  usage / file / schema-load error

Usage:
  receipt_verify.py <receipt.json> [<receipt2.json> ...]
  receipt_verify.py --self-test           # runs a valid + an invalid fixture
"""

import argparse
import copy
import json
import sys
from pathlib import Path

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    print("FATAL: jsonschema not installed (expected 4.24.x).", file=sys.stderr)
    sys.exit(2)

# --- locate the schema relative to this file (tools/condense/ -> repo root) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "receipts" / "schema" / "condensation_receipt.schema.json"

# KL warn band: above this, PPL-passing is "ppl-theater" (rule R3). Conservative
# placeholder gate; the public gate is frozen separately (§6 / §20-GATES).
KL_WARN_BAND = 0.10

# claim_types that assert a positive (public-win-eligible) result.
WIN_CLAIM_TYPES = {"density", "cliff"}
# quality_gate values that assert a shippable positive result.
POSITIVE_GATES = {"pass", "warn"}


def load_schema():
    if not SCHEMA_PATH.exists():
        print(f"FATAL: schema not found at {SCHEMA_PATH}", file=sys.stderr)
        sys.exit(2)
    with SCHEMA_PATH.open() as f:
        return json.load(f)


def schema_errors(receipt, schema):
    """Return a list of human-readable schema-violation strings (empty if OK)."""
    validator = Draft202012Validator(schema)
    out = []
    for err in sorted(validator.iter_errors(receipt), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "(root)"
        out.append(f"SCHEMA[{loc}]: {err.message}")
    return out


def invalidation_reasons(r):
    """Apply the eight §20.3 rules + the §20.6 'no win below R3' master rule.

    Returns a list of reason strings (empty => no rule tripped). Pure function
    over an already-schema-valid receipt dict.
    """
    reasons = []
    claim_type = r.get("claim_type")
    repro_level = r.get("repro_level", "R0")
    gate = r.get("quality_gate")

    is_positive = claim_type in WIN_CLAIM_TYPES and gate in POSITIVE_GATES
    # A "quality claim" is anything that leans on ppl/KL windows: density / cliff
    # / scale-point. baseline & negative don't have to clear the window bar.
    is_quality_claim = claim_type in {"density", "cliff", "scale-point"}

    # R1 — effective_bpw must be present and > 0; nominal-only is invalid.
    eff = r.get("effective_bpw")
    if eff is None or (isinstance(eff, (int, float)) and eff <= 0):
        if r.get("nominal_bpw") is not None:
            reasons.append(
                "R1: only nominal_bpw reported; effective_bpw (baker aggregate) "
                "missing or <= 0."
            )
        else:
            reasons.append("R1: effective_bpw missing or <= 0.")

    # R2 — quality from a single window, or worst-window hidden.
    n = r.get("multiwindow_n", 0)
    if is_quality_claim:
        if n < 4:
            reasons.append(
                f"R2: quality claim with multiwindow_n={n} (< 4 held-out windows)."
            )
        if r.get("ppl_condensed") is not None and r.get("multiwindow_worst_pct") is None:
            reasons.append(
                "R2: a quality number is reported but multiwindow_worst_pct "
                "(the worst window) is hidden."
            )

    # R3 — PPL passes but KL exceeds the warn band (ppl-theater).
    ppl_delta = r.get("ppl_delta_pct")
    kl = r.get("kl_parent_condensed")
    if ppl_delta is not None and kl is not None:
        ppl_passes = ppl_delta <= 2.0
        if ppl_passes and kl > KL_WARN_BAND:
            reasons.append(
                f"R3: ppl-theater — ppl_delta_pct={ppl_delta} passes but "
                f"kl_parent_condensed={kl} exceeds warn band {KL_WARN_BAND}."
            )

    # R4 — both hashes required for identity. (Schema marks them required, but
    # we re-assert here so a hash of the wrong shape / placeholder is caught and
    # reported as a §20.3 rule, not just a schema nit.)
    for field in ("source_sha256", "artifact_sha256"):
        v = r.get(field)
        if not v or not _is_sha256(v):
            reasons.append(f"R4: {field} missing or not a valid sha256 (artifact not identifiable).")

    # R5 — reproducibility: commands non-empty + a commit.
    if not r.get("commands"):
        reasons.append("R5: no commands (not reproducible).")
    if not r.get("hawking_commit"):
        reasons.append("R5: no hawking_commit (not reproducible).")

    # R6 — claim_type must match baseline behavior.
    q4 = r.get("baseline_q4_load_result")
    if claim_type == "cliff" and q4 == "ok":
        reasons.append(
            "R6: mislabelled — claim_type='cliff' but baseline_q4_load_result='ok' "
            "(Q4 loaded, so this is at most a density demo, not a cliff)."
        )
    if claim_type == "density" and q4 in {"oom", "swap-thrash"}:
        reasons.append(
            f"R6: mislabelled — claim_type='density' but baseline_q4_load_result="
            f"'{q4}' (Q4 cannot fit, so this is a cliff demo, not density)."
        )

    # R7 — MPS headline needs a CPU-bf16 confirmation.
    if r.get("mps_headline") and not r.get("cpu_bf16_confirmed"):
        reasons.append(
            "R7: MPS backend produced the headline number without a CPU-bf16 "
            "confirmation (§3)."
        )

    # R8 — best-effort baseline cannot back a public win.
    if r.get("baseline_best_effort") and is_positive:
        reasons.append(
            "R8: baseline_best_effort=true cannot back a public win "
            f"(claim_type='{claim_type}', quality_gate='{gate}')."
        )

    # §20.6 master rule — no public win below R3.
    if is_positive and repro_level in {"R0", "R1", "R2"}:
        reasons.append(
            f"R0: public-win claim (claim_type='{claim_type}', gate='{gate}') at "
            f"repro_level={repro_level} — no public WIN below R3 (§20.6)."
        )

    return reasons


def _is_sha256(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s)


def verify_receipt(receipt, schema):
    """Return (ok: bool, reasons: list[str])."""
    reasons = schema_errors(receipt, schema)
    if reasons:
        # schema failed; don't trust the rule layer on malformed data, but still
        # try the rules that don't depend on types we couldn't validate.
        try:
            reasons += invalidation_reasons(receipt)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"(rule layer skipped after schema failure: {e})")
        return False, reasons
    reasons = invalidation_reasons(receipt)
    return (len(reasons) == 0), reasons


def verify_file(path, schema):
    try:
        with open(path) as f:
            receipt = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, [f"LOAD: {e}"]
    return verify_receipt(receipt, schema)


# --------------------------------------------------------------------------- #
# Self-test fixtures
# --------------------------------------------------------------------------- #

def _valid_fixture():
    """A schema-valid, rule-clean R1 baseline receipt (the safe case)."""
    return {
        "project": "hawking",
        "receipt_version": "0.2",
        "repro_level": "R1",
        "claim_type": "baseline",
        "machine": "MacBook Pro M3 Pro, 18GB unified",
        "machine_class": "M3Pro-18",
        "os_build": "macOS 26.x",
        "model_family": "qwen",
        "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "source_sha256": "a" * 64,
        "source_precision": "bf16",
        "source_license": "apache-2.0",
        "condensed_artifact": "scratch/qwen-05b-tq3.safetensors.json",
        "artifact_sha256": "b" * 64,
        "recipe": ["tq3"],
        "effective_bpw": 3.65,
        "nominal_bpw": 3.0,
        "peak_rss_gb": 1.2,
        "multiwindow_n": 0,
        "quality_gate": "warn",
        "hawking_commit": "deadbeef",
        "commands": ["python3.12 tools/condense/audit_ladder.py scratch/qwen-05b 0.5B smoke reports/smoke_05b"],
        "raw_logs": ["scratch/qwen-05b-tq3.safetensors.json"],
    }


def _invalid_fixture():
    """A deliberately-invalid receipt: a density WIN at R1, best-effort baseline,
    nominal-only bpw, single window, missing artifact hash, no commands.
    Should trip several rules at once (R0/R1/R2/R4/R5/R8)."""
    return {
        "project": "hawking",
        "receipt_version": "0.2",
        "repro_level": "R1",
        "claim_type": "density",
        "machine": "MacBook Pro M3 Pro, 18GB unified",
        "machine_class": "M3Pro-18",
        "model_family": "qwen",
        "source_model": "Qwen/Qwen2.5-7B-Instruct",
        "source_sha256": "c" * 64,
        "source_precision": "bf16",
        "condensed_artifact": "scratch/fake.safetensors",
        "artifact_sha256": "0" * 64,
        "nominal_bpw": 2.0,
        "effective_bpw": 0.0,
        "peak_rss_gb": 5.0,
        "baseline_best_effort": True,
        "ppl_parent": 10.0,
        "ppl_condensed": 10.1,
        "ppl_delta_pct": 1.0,
        "kl_parent_condensed": 0.5,
        "multiwindow_n": 1,
        "quality_gate": "pass",
        "hawking_commit": "cafebabe",
        "commands": ["echo run"],
    }


def self_test():
    schema = load_schema()
    print("=== receipt_verify.py --self-test ===")
    print(f"schema: {SCHEMA_PATH}")

    ok_valid, reasons_valid = verify_receipt(_valid_fixture(), schema)
    print("\n[1] VALID fixture (R1 baseline, rule-clean):")
    print(f"    -> {'PASS' if ok_valid else 'FAIL'} ({len(reasons_valid)} reasons)")
    for x in reasons_valid:
        print(f"       - {x}")

    ok_invalid, reasons_invalid = verify_receipt(_invalid_fixture(), schema)
    print("\n[2] INVALID fixture (density WIN @R1, best-effort, nominal-only, 1 window):")
    print(f"    -> {'(correctly) REJECTED' if not ok_invalid else 'WRONGLY PASSED'} "
          f"({len(reasons_invalid)} reasons)")
    for x in reasons_invalid:
        print(f"       - {x}")

    # Self-test passes iff the valid one passes AND the invalid one is rejected.
    test_ok = ok_valid and (not ok_invalid) and len(reasons_invalid) > 0
    print(f"\nSELF-TEST: {'PASS' if test_ok else 'FAIL'}")
    return 0 if test_ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate Hawking condensation receipts.")
    p.add_argument("receipts", nargs="*", help="receipt .json file(s) to validate")
    p.add_argument("--self-test", action="store_true",
                   help="run a valid + an invalid fixture and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.receipts:
        p.error("no receipts given (use --self-test for the demo)")

    schema = load_schema()
    overall_ok = True
    for path in args.receipts:
        ok, reasons = verify_file(path, schema)
        status = "VALID" if ok else "INVALID"
        print(f"{status}: {path}")
        for x in reasons:
            print(f"   - {x}")
        overall_ok = overall_ok and ok
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
