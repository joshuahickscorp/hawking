# Hawking Rename — Execution Kit (push-button, post-training)

Date: 2026-06-19
Companion to: `hawking_total_rename_plan_2026_06_19.md` (the strategy). This is the
runnable kit: exact commands, gates, and hazards from a read-only inventory of the
current tree.

## STATUS: DO NOT EXECUTE THE DIRECTORY/CRATE RENAME YET

A live RWKV-7 QAT job + its `autocycle` relaunch every ~5 steps depend on these
exact paths: `tools/training/rwkv7_qat.py`, `tools/training/autocycle_step50_ozempic.sh`,
`tools/training/hawking_after_ema.py`, `models/`, `artifacts/lowbit_rwkv7/runs/`.
A `git mv` / folder rename mid-run breaks the next relaunch and kills training.

- **Phase 1 (compat layer)** edits Rust source only → SAFE to run even while training
  (the trainer is Python and never imports the crates). Optional to start now.
- **Phase 3/4 (dir + crate + tooling rename)** → WAIT until G1a + the post-G1a chain
  (phase2 → v2_expansion → draft sweep → hawking) finish.

## Rename surface (measured)

| Surface | Scale |
|---|---|
| `dismantle_core/serve/bench` Rust imports | 466 occurrences, ~145 files (top: `crates/dismantle/src/main.rs` 57) |
| `DISMANTLE_*` env vars (crates) | 110+ distinct, read via `env_on/env_opt_out/env_usize` in `crates/dismantle-core/src/lib.rs:48-74` |
| `DISMANTLE_*` env refs (tools) | 243 occ / 44 files (only 4 SET it for a subprocess) |
| `target/release/dismantle` refs (tools) | 75 occ / 60 files (all have env-var defaults) |
| Docs with `dismantle` | ~143 files: 5 PUBLIC (README, CONTRIBUTING, docs/serve|autotune|profile.md), rest INTERNAL/historical |
| profiles/*.json | 0 references — no change needed |
| "dismantle" used as English verb | 0 found — sed is safe |

## HAZARDS — never blind-sed these (wire/format/live)

1. **`.dismantle` sidecar extension** — persisted binary format with magic bytes
   (`crates/dismantle-core/src/sidecar.rs`, used in `qwen_dense.rs`). Users have baked
   `.dismantle` files on disk. Keep the extension; if ever renamed, add dual-path read.
2. **`dismantle_*` Prometheus metrics** (`crates/dismantle-serve/src/http.rs:186-209`) —
   scraped by dashboards. Dual-EMIT, never replace.
3. **`/v1/dismantle/*` HTTP routes** (`http.rs:167-168`) — clients hardcode them. Add
   `/v1/hawking/*` as ALIASES to the same handlers; keep the old routes.
4. **`DISMANTLE_*` env vars** — live training/bench depend on them. New resolver must
   accept both, old as fallback.
5. **LIVE, do not rename until training ends:** `tools/training/hawking_after_ema.py`,
   `tools/training/hawking_after_ema.sh`, `artifacts/lowbit_rwkv7/hawking_arc/`, and the
   `hawking_branch_512_g8` / `hawking_anchor_1024_g16` screen/run names.

## PHASE 1 — Compatibility layer (additive, Rust-only, ~7 files ~100 lines)

Purpose: introduce `HAWKING_*` names that work WITHOUT removing `DISMANTLE_*`. Nothing
breaks; new wins if both set. Safe to land anytime, including during training.

1. **Env resolver** — `crates/dismantle-core/src/lib.rs` (after line 74): add
   `env_on_compat/env_usize_compat` that check `HAWKING_<suffix>` then `DISMANTLE_<suffix>`.
   Leave existing `env_on/env_opt_out/env_usize` untouched; route new call-sites through
   compat. Hot call-sites: `profile.rs` (RuntimeLevers::from_env ~137-178), `qwen_dense.rs`.
2. **Prefix cache / home** — `cache/prefill_disk.rs:open_from_env()` check
   `HAWKING_PREFIX_CACHE_DIR` then `DISMANTLE_PREFIX_CACHE_DIR`; `mixed_quant_store.rs:cache_root()`
   check `HAWKING_CACHE_HOME` first. Keep `.cache/dismantle` dir name for now.
3. **HTTP aliases** — `dismantle-serve/src/http.rs:router()` add 2 lines:
   `/v1/hawking/tokens` + `/v1/hawking/generate` → same handlers. Keep old routes.
4. **Metrics dual-emit** — `http.rs:metrics()` append `hawking_*` copies of all 8 metrics
   with identical values; keep `dismantle_*`.
5. **CLI alias** — `crates/dismantle/Cargo.toml` add a second `[[bin]] name = "hawking"
   path = "src/main.rs"`. No code change.
6. **One-shot deprecation log** — in `main.rs` startup, warn once if any `DISMANTLE_*`
   seen and no `HAWKING_*`.

Gate:
```bash
cargo test --workspace
rg -n "HAWKING_|/v1/hawking|hawking_" crates
```

## PHASE 3 — Mechanical crate rename (ONE no-logic commit, post-training)

```bash
# 1. dirs
git mv crates/dismantle       crates/hawking
git mv crates/dismantle-core  crates/hawking-core
git mv crates/dismantle-serve crates/hawking-serve
git mv crates/dismantle-bench crates/hawking-bench

# 2. Cargo.toml package names + path deps + [[bin]] names
#    root Cargo.toml workspace members; each crate [package].name;
#    deps dismantle-core -> hawking-core (in hawking/, -serve/, -bench/);
#    bin: dismantle -> hawking, dismantle-spec-acceptance-measure -> hawking-spec-acceptance-measure
#    (keep an extra [[bin]] name="dismantle" path="src/main.rs" alias for 2-3 releases)

# 3. import rewrites (BSD sed on macOS: -i '')
find crates -name '*.rs' -exec sed -i '' -E \
  's/\bdismantle_core::/hawking_core::/g; s/\bdismantle_serve::/hawking_serve::/g; s/\bdismantle_bench::/hawking_bench::/g' {} \;

# 4. CLI command name
#    crates/hawking/src/main.rs: #[command(name = "hawking", ...)]
#    crates/hawking/src/spec_acceptance_measure.rs: name = "hawking-spec-acceptance-measure"

# 5. gate
cargo test --workspace && cargo build --release
ls target/release/hawking target/release/dismantle   # alias bin still present
```
Rule: Phase 3 is mechanical only — NO behavior edits in the same commit (keeps the diff
auditable). Do NOT touch the `.dismantle` extension, metric names, or `/v1/dismantle/*`
routes here (those are the Phase-1 dual-stack, already additive).

## PHASE 4 — Tooling + docs (post-training; old env vars still accepted)

```bash
# tools env vars: safe scripted swap (resolver keeps DISMANTLE_* working as fallback)
find tools -name '*.sh' -o -name '*.py' | xargs sed -i '' 's/DISMANTLE_/HAWKING_/g'
# binary path refs + build a compat symlink
find tools -name '*.sh' -o -name '*.py' | xargs sed -i '' 's#target/release/dismantle#target/release/hawking#g'
# 4 scripts that EXPORT DISMANTLE_* for subprocesses: quick_bench.sh, clean_room_batch.sh,
#   quality_scout.sh, path_to_50_verify.sh
# tools/bench/engines/dismantle_serve.json launch_command -> hawking
```
Docs: PUBLIC (README, CONTRIBUTING, docs/serve|autotune|profile.md) — manual rewrite +
"formerly dismantle" banner. INTERNAL docs/plans|reports|strand (~186 files) — leave as
dated historical records.

## PHASE 6 — Repo rename
Only after a user can `hawking serve` / `hawking bench`. GitHub repo rename, remote URL,
badges; tag `dismantle-final-alias`.

## Pre-execution checklist
- [ ] G1a + chain finished (no live process under `artifacts/lowbit_rwkv7/runs/`)
- [ ] On a fresh branch `rename/hawking-phaseN`
- [ ] Phase 1 merged & green before Phase 3
- [ ] `cargo test --workspace && cargo build --release` green after Phase 3
- [ ] `dismantle` alias bin still runs (compat window)
