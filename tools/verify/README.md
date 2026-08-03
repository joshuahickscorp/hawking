# `tools/verify` — rebuild verification instruments

## Assertion ledger (Core F F1)

Seals every current verification obligation under a durable `CASE.<kind>.<slug>`
identity before any test mass is rewritten or deleted. Apparatus only — zero
product behaviour change, zero test deletion.

```bash
python3.12 tools/verify/case_extract.py --json
python3.12 tools/verify/case_extract.py --write \
  workspace/campaign/governance/control/catalog/manifests/ASSERTION_LEDGER.json
python3.12 tools/verify/case_extract.py --write \
  workspace/campaign/governance/control/catalog/manifests/ASSERTION_LEDGER.json --rev HEAD
python3.12 tools/verify/case_extract.py --check \
  workspace/campaign/governance/control/catalog/manifests/ASSERTION_LEDGER.json

python3.12 tools/verify/test_case_manifest.py --seal-check \
  workspace/campaign/governance/control/catalog/manifests/ASSERTION_LEDGER.json \
  workspace/campaign/governance/control/catalog/manifests/TEST_CASE_MANIFEST.json
python3.12 tools/verify/test_case_manifest.py --enumerate \
  workspace/campaign/governance/control/catalog/manifests/TEST_CASE_MANIFEST.json
python3.12 tools/verify/test_case_manifest.py --dry-run \
  workspace/campaign/governance/control/catalog/manifests/TEST_CASE_MANIFEST.json
python3.12 tools/verify/test_case_manifest.py --gate \
  --before workspace/campaign/governance/control/catalog/manifests/ASSERTION_LEDGER.json \
  --after workspace/campaign/governance/control/catalog/manifests/TEST_CASE_MANIFEST.json
```

- Extraction reads exact git tree/blob objects (`git ls-tree` / `git show`) at a revision — no worktree checkout.
- `--write` / `--json` default to `HEAD` (or `--rev`). The ledger records `sealed_at_commit`.
- `--check LEDGER` re-extracts at the ledger's `sealed_at_commit` (or an explicit identical `--rev`), not current HEAD. A later unrelated commit must leave a prior sealed ledger green.
- Commands use the live compact paths. Sealed metadata can retain its historical `control/...` path; readers resolve that identity to the live file without rewriting it.
- Vitest identities: repository path + lexical `describe` chain + literal title; content digest only for same-chain title collisions; identical duplicates collision-fail (never `#L{line}`).
- Fingerprints bind the full Rust test item (attr+signature+body) and the full Vitest call (including body/`expect`).
- `--seal-check` may pass with an empty F1 scaffold when the ledger hash and phase/status are valid.
- Normal `--gate` **must fail** while `entries` is empty and lists every unaccounted ledger id. There is no ledger-only gate pass.
- Gate rejects unknown manifest ids, dual accounting owners for one sealed id, and incomplete N-entry rewrite receipt identity maps.

Adversarial unit tests:

```bash
PYTHONNOUSERSITE=1 python3.12 -m unittest tools.verify.test_case_extract tools.verify.test_test_case_manifest
```

## Performance gate

Instrument that decides the rebuild hard gate:

- no >2% regression in base TPS
- no >2% regression in accelerated TPS
- no >2% regression in transformation throughput
- no material startup/compile regression without a measured trade receipt

### Commands

```bash
python3.12 tools/verify/perfgate.py --list
python3.12 tools/verify/perfgate.py --capture --out \
  workspace/campaign/evidence/runtime/rebuild/REBUILD_PERFORMANCE_BASELINE_MEASURED.json
python3.12 tools/verify/perfgate.py --compare A.json B.json --gate 2.0
python3.12 tools/verify/perfgate.py --paired --a-cmd '…' --b-cmd '…' --n 9
```

Stdlib only (+ `statistics`). Optional: existing release binary (`HAWKING_BIN`), GGUF (`HAWKING_GGUF`), `CARGO_TARGET_DIR`.

## Design rules

1. **No silent empty measurements.** Every metric is `measured` | `skipped` | `unavailable` with a reason. Compare exits non-zero if a metric that was measured in A is no longer measured in B.
2. **No fabricated TPS.** Base / accelerated TPS need Metal + a real artifact. If either is missing, status is `unavailable` — never a synthetic proxy labeled TPS.
3. **Contamination is assumed.** This box is not a clean room. Every sample records 1/5/15 loadavg, free/active memory, and (when `ps` is permitted) whether another process held >4 cores. Prefer `--paired` (ABAB interleave + sign test) over absolute numbers hours apart.
4. **Statistics.** `n` runs including 1 discarded warm-up; report median and min–max, never mean alone. Protocol default `n=8` ⇒ 7 kept samples.

## Metric families

| Family | What | When unavailable |
|---|---|---|
| `build` | `cargo check`, warm release build (touch leaf), optional cold (`--include-cold`), binary size | no cargo / no binary |
| `startup` | `hawking --help`, `version`, `doctor --json` | no binary; doctor needs GGUF |
| `base_tps` | `gravity_tps` on llama-1B `.gravity`; optional GLM Math-Preserve (`--include-glm-tps`) | no Metal, no artifact, no example binary |
| `accelerated_tps` | `hawking bench --suite decode --profile fast` | no Metal / no GGUF |
| `transform` | `gravity_format` selftest; fixture-scale shard write/verify bytes/s; `glm52_pack` pack_indices bytes/s | missing lab scripts |
| `kernel` | `bench-q4k-shapes` (no model); `bench-kernel` marked unavailable (CLI extracted) | no Metal |
| `numeric_parity` | `gravity_format` write/verify/tamper path (CPU container oracle) | missing script |

## Compare semantics

For each metric measured on both sides, compute `delta_pct_improvement` (positive = B better). Fail if improvement &lt; `−gate` (default 2%). Higher-is-better (tps, bytes/s) and lower-is-better (seconds, bytes, µs) are handled explicitly.

## Env

| Variable | Purpose |
|---|---|
| `HAWKING_BIN` | Path to `hawking` binary |
| `HAWKING_GGUF` | Path to a GGUF for doctor / accelerated bench |
| `CARGO_TARGET_DIR` | Shared cargo target (defaults to repo or main checkout target) |
