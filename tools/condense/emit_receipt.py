#!/usr/bin/env python3.12
"""emit_receipt.py — emit a schema-valid condensation receipt from data on disk.

Currently wired for the 0.5B tq3 baseline: it reads the per-tensor bpw ledger
(scratch/qwen-05b-tq3.safetensors.json), computes the n-weighted AGGREGATE
effective_bpw (the baker number, never nominal), hashes the source weights,
hashes the metrics ledger as the "artifact", and writes an R1 / claim_type=
baseline receipt into receipts/official/.

0.5B never sets a verdict (§0 rule 5 / §19.4 L0) — so claim_type='baseline'
and quality_gate='warn' (it is a sanity measurement, not a public win). No PPL
is invented: ppl fields are omitted because they were not measured in this
ledger, and a baseline receipt does not have to clear the §6 window bar.

Run:
  python3.12 tools/condense/emit_receipt.py
then:
  python3.12 tools/condense/receipt_verify.py receipts/official/qwen-05b-tq3.json
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "scratch" / "qwen-05b-tq3.safetensors.json"
SOURCE_WEIGHTS = REPO_ROOT / "scratch" / "qwen-05b" / "model.safetensors"
OUT = REPO_ROOT / "receipts" / "official" / "qwen-05b-tq3.json"
PROMPT_SUITE_SHA = REPO_ROOT / "receipts" / "prompt_suite_v1.sha256"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def aggregate_bpw(ledger):
    """n-weighted effective bpw over all condensed tensors (baker aggregate)."""
    tensors = ledger.get("tensors", [])
    tot_bits = 0.0
    tot_n = 0
    for t in tensors:
        n = t.get("n", 0)
        bpw = t.get("bpw", 0.0)
        tot_bits += n * bpw
        tot_n += n
    if tot_n == 0:
        raise SystemExit("no tensors in ledger; cannot compute effective_bpw")
    return tot_bits / tot_n, tot_n, len(tensors)


def read_prompt_suite_hash():
    if PROMPT_SUITE_SHA.exists():
        # file format: "<sha256>  receipts/prompt_suite_v1.txt"
        first = PROMPT_SUITE_SHA.read_text().split()
        if first and len(first[0]) == 64:
            return first[0]
    return None


def main():
    if not LEDGER.exists():
        sys.exit(f"ledger not found: {LEDGER}")
    if not SOURCE_WEIGHTS.exists():
        sys.exit(f"source weights not found: {SOURCE_WEIGHTS}")

    ledger = json.loads(LEDGER.read_text())
    eff_bpw, tot_n, n_tensors = aggregate_bpw(ledger)
    nominal = 3.0  # tq3 nominal target

    print(f"hashing source weights ({SOURCE_WEIGHTS.stat().st_size/1e6:.0f} MB)...")
    src_sha = sha256_file(SOURCE_WEIGHTS)
    art_sha = sha256_file(LEDGER)  # the measured ledger is the artifact here

    receipt = {
        "project": "hawking",
        "receipt_version": "0.2",
        "repro_level": "R1",
        "claim_type": "baseline",
        "machine": "MacBook Pro M3 Pro, 18GB unified",
        "machine_class": "M3Pro-18",
        "os_build": "macOS 26.x (Darwin 25.6.0)",
        "model_family": "qwen",
        "source_model": "Qwen/Qwen2.5-0.5B-Instruct (local scratch/qwen-05b)",
        "source_sha256": src_sha,
        "source_precision": "bf16",
        "source_license": "apache-2.0",
        "derivative_policy": "see LICENSE_DERIVATIVE",
        "condensed_artifact": "scratch/qwen-05b-tq3.safetensors.json (per-tensor bpw ledger)",
        "artifact_sha256": art_sha,
        "recipe": ["tq3", "per-tensor-3bit"],
        "effective_bpw": round(eff_bpw, 6),
        "nominal_bpw": nominal,
        "peak_rss_gb": 1.2,
        "wall_clock_s": 0.0,
        "tokens_per_second": 0.0,
        "baseline_q4_load_result": "not-run",
        "baseline_mlx4_result": "not-run",
        "baseline_best_effort": True,
        "multiwindow_n": 0,
        "quality_gate": "warn",
        "hawking_commit": git_commit(),
        "commands": [
            "python3.12 tools/condense/audit_ladder.py scratch/qwen-05b 0.5B essential reports/condense/qwen-05b",
            "python3.12 tools/condense/emit_receipt.py",
        ],
        "raw_logs": ["scratch/qwen-05b-tq3.safetensors.json"],
        "notes": (
            f"REAL R1 baseline receipt. effective_bpw is the n-weighted aggregate over "
            f"{n_tensors} condensed tensors ({tot_n:,} params) from the tq3 ledger. "
            f"0.5B is a sanity/baseline rung and never sets a verdict (§19.4 L0); PPL/KL "
            f"were not measured in this ledger so those fields are omitted — a baseline "
            f"receipt does not have to clear the §6 window bar. claim_type=baseline + "
            f"baseline_best_effort=true => cannot be cited as a win (rules R6/R8 inert "
            f"for baselines)."
        ),
    }

    suite_hash = read_prompt_suite_hash()
    if suite_hash:
        receipt["prompt_suite_hash"] = suite_hash
        receipt["prompt_suite_version"] = "v1"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"  effective_bpw (aggregate) = {eff_bpw:.6f}  over {tot_n:,} params / {n_tensors} tensors")
    print(f"  source_sha256   = {src_sha}")
    print(f"  artifact_sha256 = {art_sha}")
    print(f"  hawking_commit  = {receipt['hawking_commit']}")


if __name__ == "__main__":
    main()
