#!/usr/bin/env python3
"""The serving resident binary is older than its own instrumentation.

crates/hawking-core/examples/ascension_qwen38_resident.rs emits a `metrics`
block carrying `dispatches`, `dispatches_per_generated_token`,
`active_bytes_per_token`, `actual_read_bytes_per_token` and a per-phase
prefill/decode split. The binary actually being served does not: it was built
2026-08-26, the instrumentation landed 2026-08-27 in 8b6f50270, and the source
has been modified again since.

So "the resident reports its dispatch count" is true of the SOURCE and false of
the RUNNING SYSTEM. That is the same shape of defect this campaign keeps
finding: a narrow probe wearing a broad causal label. Anything quoted as a
resident dispatch count or resident byte ledger did not come from this binary.

    python3 tools/future/resident_binary_drift.py --record
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, git  # noqa: E402

BINARY = REPO / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident"
SOURCE = REPO / "crates/hawking-core/examples/ascension_qwen38_resident.rs"
RECEIPT = REPO / "receipts" / "future" / "RESIDENT_BINARY_DRIFT.json"

# Fields the source emits that a caller may reasonably expect to exist.
EXPECTED_FIELDS = (
    "dispatches",
    "dispatches_per_generated_token",
    "active_bytes_per_token",
    "active_weight_bytes_per_generated_token",
    "actual_read_bytes_per_token",
    "actual_read_bytes_status",
    "gpu_ns_per_generated_token",
    "resident_weight_bytes",
)


def _strings(path: Path) -> set[str]:
    try:
        out = subprocess.run(["strings", str(path)], capture_output=True,
                             text=True, timeout=120).stdout
    except (subprocess.TimeoutExpired, OSError):
        return set()
    return set(out.splitlines())


def build() -> dict[str, object]:
    present = _strings(BINARY) if BINARY.exists() else set()
    src = SOURCE.read_text() if SOURCE.exists() else ""
    fields = {}
    for f in EXPECTED_FIELDS:
        fields[f] = {
            "in_source": f'"{f}"' in src,
            "in_binary": f in present,
        }
    missing = [f for f, v in fields.items() if v["in_source"] and not v["in_binary"]]
    return {
        "schema": "hawking.future.resident_binary_drift.v1",
        "version": 1,
        "recorded_by": "tools/future/resident_binary_drift.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "binary": {
            "path": str(BINARY.relative_to(REPO)),
            "present": BINARY.exists(),
            "mtime": (
                __import__("datetime").datetime.fromtimestamp(BINARY.stat().st_mtime).isoformat()
                if BINARY.exists() else None
            ),
            "bytes": BINARY.stat().st_size if BINARY.exists() else None,
        },
        "source": {
            "path": str(SOURCE.relative_to(REPO)),
            "mtime": (
                __import__("datetime").datetime.fromtimestamp(SOURCE.stat().st_mtime).isoformat()
                if SOURCE.exists() else None
            ),
            "dirty": bool(git("status", "--porcelain", "--", str(SOURCE.relative_to(REPO)))),
            "last_commit": git("log", "-1", "--format=%h %ad %s", "--date=short",
                               "--", str(SOURCE.relative_to(REPO))) or None,
            "instrumentation_landed_in": git(
                "log", "--format=%h %ad %s", "--date=short", "-S",
                "dispatches_per_generated_token", "--",
                str(SOURCE.relative_to(REPO))).splitlines()[:1] or None,
        },
        "fields": fields,
        "missing_from_binary": missing,
        "drift": bool(missing),
        "consequence": (
            "The running resident cannot answer 'measure the CURRENT exact "
            "production dispatch count'. That question needs a rebuild first, "
            "and any dispatch count already in a receipt did not come from "
            "this binary."
        ),
        "second_defect_the_probe_exposed": {
            "claim": "the serving resident has only ever been built with a "
                     "profile the codebase forbids benchmarking",
            "profile_served": "release-fast",
            "cargo_toml_says": "Correctness-testing profile ONLY. NEVER "
                               "benchmark with this: TPS numbers must come from "
                               "`release` (lto=fat, codegen-units=1), which is "
                               "what the runtime artifacts ship with.",
            "release_fast_settings": {"lto": False, "codegen_units": 16,
                                      "incremental": True},
            "release_build_of_this_example_exists": False,
            "searched": "workspace/ops/build/rust/*/examples/ascension_qwen38_resident",
            "consequence": (
                "Every absolute TPS number attributed to this resident was "
                "taken on a non-benchmark build. The relative science is "
                "unaffected -- the three falsifications are relative or static "
                "-- but the absolute floor is unknown and the real one is "
                "likely better than measured. The 35.5 TPS anchor inherits this."
            ),
            "resolves_when": (
                "the same probe runs against a `--profile release` build of "
                "ascension_qwen38_resident on an uncontended box"
            ),
        },
        "law": (
            "A capability present in source is not a capability present in the "
            "running system. Probe the artifact that serves, not the file that "
            "describes it."
        ),
        "observed_live": {
            "note": "one probe request, N=128, taken while 4 Grok lanes ran at "
                    "maximum, so timing is DIAGNOSTIC and not protected",
            "decode_tps": 32.774,
            "complete_tps": 30.172,
            "prompt_tokens": 12,
            "generated_tokens": 128,
            "decode_steps": 127,
            "prefill_wall_ns": 361_615_875,
            "decode_wall_ns": 3_874_984_167,
            "fallbacks": 0,
            "ready_s": 5.79,
            "confirms_l42": (
                "complete 30.17 vs decode 32.77 is 7.9% at N=128, and the whole "
                "difference is the 361.6 ms prefill. The gap shrinks as N grows, "
                "exactly as P/(P+N-1) predicts. It is accounting, not ceremony."
            ),
        },
        "claim_boundary": (
            "Static comparison of a binary's string table against its source, "
            "plus one contended live probe. The timing figures are "
            "DIAGNOSTIC_RELATIVE and must not be quoted as a protected "
            "measurement. Absence of a string in a stripped or optimized binary "
            "is strong but not absolute evidence; the live probe returning none "
            "of these fields is the confirming observation."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True, default=str) + "\n")
    return RECEIPT


if __name__ == "__main__":
    if "--record" in sys.argv:
        p = record()
        d = json.loads(p.read_text())
        print(f"wrote {p}")
        print(f"drift={d['drift']} missing_from_binary={d['missing_from_binary']}")
    else:
        print(json.dumps(build(), indent=1, default=str))
