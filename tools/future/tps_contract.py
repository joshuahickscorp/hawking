#!/usr/bin/env python3.12
"""A TPS number without its measurement contract is not comparable to any other one.

[S006 14] "Do not compare TPS numbers across mismatched measurement contracts. Every
receipt must identify specimen, NR, runtime, context, path, concurrency."

That rule already had a casualty. O003 carried two decode figures 25% apart, 87.67 and
69.97, and the disagreement could not be RESOLVED by re-reading them -- because neither
receipt recorded its mlx_lm version, so 69.97's compute path could not be reconstructed
at all. It had to be settled by internal consistency instead (87.67 x 5.158 GB/token =
452.2 GB/s, matching that receipt's own recorded bandwidth exactly).

This is the validator that makes such a receipt REFUSE rather than quietly sit in the
ledger looking comparable. It measures nothing. It reads receipts and says which ones
can be compared to which, and names the missing field when they cannot.

Not a linter for style: a missing `runtime` is the difference between a number that can
be reproduced and a number that can only be believed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The six identifying fields. A receipt missing ANY of them cannot be compared, because
# each one alone can move a decode figure by more than the differences being argued over.
REQUIRED = {
    "specimen":    ("specimen", "slug", "model", "model_id", "body"),
    "nr":          ("nr", "representation", "quantization", "quant", "variant", "arm"),
    "runtime":     ("runtime", "mlx_lm_version", "engine", "binary", "runtime_version"),
    "context":     ("context", "prompt_tokens", "max_seq_len", "context_tokens", "seq_len"),
    "path":        ("path", "compute_path", "route", "kernel", "shim", "method"),
    "concurrency": ("concurrency", "n_parallel", "batch", "batch_size", "streams"),
}

# What makes a receipt PHYSICAL: it carries a NUMBER that is a rate. Substring matching
# on the serialized blob was tried first and classified 282 of 443 receipts as physical,
# including a namespace audit and a dead-callsite sweep, because "decode" and "toks"
# appear in prose. An inflated denominator would have made "zero comparable" look far
# worse than it is, so the test is a numeric key, not a word.
RATE_KEYS = ("tps", "tok_s", "toks_per_s", "tok_per_s", "tokens_per_s",
             "tokens_per_second", "decode_tps", "prefill_tps", "throughput_tps",
             "gb_per_s", "gbps", "bandwidth_gb_s", "achieved_bandwidth_gb_s")


def _is_rate_key(key: str) -> bool:
    k = key.replace("/", "_").replace("-", "_")
    return any(r in k for r in RATE_KEYS)


def _walk(node, depth=0):
    """Every key in the receipt, at any depth. Contracts get recorded in odd places."""
    if depth > 8:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k).lower(), v
            yield from _walk(v, depth + 1)
    elif isinstance(node, list):
        for v in node[:200]:
            yield from _walk(v, depth + 1)


def audit(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        return {"receipt": path.name, "verdict": "UNREADABLE", "why": str(exc)[:120]}
    keys = {}
    for k, v in _walk(doc):
        keys.setdefault(k, v)
    rates = {k: v for k, v in keys.items()
             if _is_rate_key(k) and isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not rates:
        return {"receipt": path.name, "verdict": "NOT_A_PHYSICAL_RECEIPT"}
    missing, found = [], {}
    for field, aliases in REQUIRED.items():
        hit = next((a for a in aliases if a in keys and keys[a] not in (None, "", [], {})), None)
        if hit is None:
            missing.append(field)
        else:
            found[field] = hit
    return {
        "receipt": path.name,
        "verdict": "COMPARABLE" if not missing else "INCOMPARABLE",
        "identifies": found,
        "missing": missing,
        "rates": {k: rates[k] for k in sorted(rates)[:6]},
        # The whole point: say WHY it cannot be compared, in the receipt's own terms.
        "why": None if not missing else
               f"cannot be reproduced or compared: no {', '.join(missing)}",
    }


def audit_all(directory: Path) -> dict:
    rows = [audit(p) for p in sorted(directory.glob("*.json"))]
    physical = [r for r in rows if r["verdict"] in ("COMPARABLE", "INCOMPARABLE")]
    return {
        "directory": str(directory.relative_to(ROOT)),
        "n_json": len(rows),
        "n_physical": len(physical),
        "n_comparable": sum(1 for r in physical if r["verdict"] == "COMPARABLE"),
        "n_incomparable": sum(1 for r in physical if r["verdict"] == "INCOMPARABLE"),
        "rows": physical,
    }


def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        full = {"specimen": "x", "nr": "bf16", "runtime": "mlx_lm 0.31.3",
                "context": 512, "path": "split_kv_b_proj", "concurrency": 1,
                "decode_tps": 85.64}
        # OVER-CLASSIFICATION CONTROL: prose containing "decode" and "toks" with no rate
        # NUMBER is not a physical receipt. The first version of this tool called 282 of
        # 443 receipts physical on exactly this mistake.
        (d / "prose.json").write_text(json.dumps(
            {"note": "the decode path emits toks and bandwidth is discussed here"}))
        assert audit(d / "prose.json")["verdict"] == "NOT_A_PHYSICAL_RECEIPT", (
            "prose about decode was classified as a measurement")
        # and a rate that is a STRING is not a measured number either
        (d / "strrate.json").write_text(json.dumps({"decode_tps": "fast"}))
        assert audit(d / "strrate.json")["verdict"] == "NOT_A_PHYSICAL_RECEIPT"
        (d / "full.json").write_text(json.dumps(full))
        # POSITIVE CONTROL: a complete receipt must pass.
        assert audit(d / "full.json")["verdict"] == "COMPARABLE"
        # NEGATIVE CONTROL, one field at a time. Each must fail, and must NAME the field
        # it is missing -- a validator that fails without saying why is not usable.
        for field in REQUIRED:
            partial = {k: v for k, v in full.items() if k != field}
            (d / "p.json").write_text(json.dumps(partial))
            r = audit(d / "p.json")
            assert r["verdict"] == "INCOMPARABLE", f"{field} missing but passed"
            assert field in r["missing"], f"{field} missing but not named: {r['missing']}"
            assert field in (r["why"] or ""), f"why does not name {field}: {r['why']}"
        # A receipt that is not about physics is SKIPPED, not failed. Otherwise the pass
        # rate is diluted by every unrelated receipt in the directory.
        (d / "other.json").write_text(json.dumps({"anatomy": {"deficit_pct": 12.0}}))
        assert audit(d / "other.json")["verdict"] == "NOT_A_PHYSICAL_RECEIPT"
        # A present-but-EMPTY field is missing. This is the one that would otherwise let
        # a receipt claim a contract it does not have.
        empty = dict(full, runtime="")
        (d / "empty.json").write_text(json.dumps(empty))
        assert "runtime" in audit(d / "empty.json")["missing"], "an empty field counted as present"
        # Nesting: contracts get recorded under a sub-object, and must still be found.
        nested = {"measurement_contract": {k: full[k] for k in REQUIRED}, "decode_tps": 85.64}
        (d / "nested.json").write_text(json.dumps(nested))
        assert audit(d / "nested.json")["verdict"] == "COMPARABLE", "nested contract not found"
    print("selftest: PASS (complete passes; each of the 6 fields fails ALONE and is named; "
          "prose-about-decode and string rates are NOT physical; empty counts as missing; "
          "nested contract found)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    out = audit_all(ROOT / "receipts" / "future")
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        raise SystemExit(0)
    print(f"  {out['n_physical']} physical receipts of {out['n_json']} json in {out['directory']}")
    print(f"  COMPARABLE {out['n_comparable']}   INCOMPARABLE {out['n_incomparable']}")
    for r in out["rows"]:
        if r["verdict"] == "INCOMPARABLE":
            print(f"    {r['receipt']:<52} missing {','.join(r['missing'])}")
    for r in out["rows"]:
        if r["verdict"] == "COMPARABLE":
            print(f"    OK {r['receipt']:<49} {r['identifies']}")
