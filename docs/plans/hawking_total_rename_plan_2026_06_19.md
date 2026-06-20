# Hawking Total Rename Plan

Date: 2026-06-19

This is a future migration plan, not an instruction to rename the repository now.
The current codebase is mid-flight on batching, RWKV, low-bit QAT, STRAND/TQ,
spec decode, and serving work. The right move is to design the full rename now,
add compatibility seams first, and only flip public names when the product
surface is ready.

## Thesis

`dismantle` was a good builder name: it says "take the stack apart, learn every
piece, own it." The next public surface is different. If users will say "I
downloaded a Hawking-tuned model", "I used Hawking instead of llama.cpp for this
Apple Silicon run", or "this is a Hawking quant", then Hawking should become the
top-level identity.

The public promise becomes:

> Hawking is a local model foundry and runtime for Apple Silicon: tuned,
> compressed, measured models optimized for capability density.

`dismantle` can survive as the old codename, a historical branch name, or a
temporary crate namespace during the migration. Long-term, Hawking should own
the CLI, package names, model names, release artifacts, docs, endpoints, metrics,
env vars, and quant format branding.

## Naming Decisions

| Surface | Future name | Compatibility plan |
|---|---|---|
| Project | Hawking | Announce as "formerly dismantle" for the first public cycle. |
| Runtime binary | `hawking` | Keep `dismantle` binary alias for 2-3 releases. |
| Runtime crate | `hawking` or `hawking-cli` | Rename `crates/dismantle` after CLI alias exists. |
| Core crate | `hawking-core` | Rename `dismantle_core` imports in one mechanical commit. |
| Server crate | `hawking-serve` | Preserve API structs while endpoints dual-stack. |
| Bench crate | `hawking-bench` | Keep benchmark output comparable across rename. |
| Home dir | `~/.hawking` | Read `HAWKING_HOME`, fallback to `DISMANTLE_HOME`, fallback to `~/.dismantle` during migration. |
| Env prefix | `HAWKING_*` | Accept `DISMANTLE_*`; if both set, `HAWKING_*` wins and log once. |
| Native endpoints | `/v1/hawking/*` | Keep `/v1/dismantle/*` aliases. |
| Metrics | `hawking_*` | Emit old `dismantle_*` metrics for one release with deprecation comments. |
| Headbank schema | `hawking-headbank-manifest-v1` | Read old `dismantle-headbank-manifest-v1`. |
| Model line | `Hawking-*` | Public models should never use the old name. |

## Public Product Taxonomy

Hawking should not present as one thing. It should be a stack with named layers:

| Layer | Public label | Meaning |
|---|---|---|
| Runtime | Hawking Runtime | The local Apple Silicon serving/generation engine. |
| Models | Hawking Models | Tuned/distilled/quantized releases. |
| Quantization | Hawking Quant | The artifact and evaluation pipeline. |
| Drafts | Hawking Draft | Small proposal models for spec decode. |
| Profiles | Hawking Profiles | Hardware-specific runtime configs and measured presets. |
| Lab notes | Hawking Reports | Model cards, benchmark reports, negative results, eval ledgers. |

Example model names:

- `Hawking-RWKV7-G1-0.4B-Instruct`
- `Hawking-RWKV7-G1-0.4B-STRAND2`
- `Hawking-Qwen2.5-1.5B-Apple-STRAND2`
- `Hawking-Draft-RWKV-191M`
- `Hawking-Qwen3B-Code-M3Ultra-Q4K`

## TQ / STRAND Rename Options

Current reality:

- `tools/tq_bake` writes `.tq` artifacts.
- The wire format is STR2 / STRAND-derived.
- The vendor crates are `strand-quant` and `strand-decode-kernel`.
- Runtime code has `tq.rs`, `tq_gpu.rs`, `ProjWeight::Tq`, and tests named
  `rwkv7_tq_*`.

Do not blindly rename the internal algorithm. STRAND is already meaningful as
the trellis/archive lineage. The public artifact can still be Hawking-branded.

Recommended split:

| Level | Name | Extension / magic | Rationale |
|---|---|---|---|
| Internal algorithm | STRAND-1 / STRAND-2 | existing STR2 internals | Precise, research-linked, avoids confusing the code while still experimental. |
| Current dev artifact | TQ | `.tq` | Keep for prototypes and old tests. |
| Public release wrapper | Hawking Quant Archive | `.hqa` / `HQA1` | A stable user-facing container that can wrap STRAND/TQ payloads plus eval/provenance. |
| Optional ideology name | Event Horizon Quant | `.ehq` / `EHQ1` | Strong theme, but easier to overdo. Use only if it becomes a distinct algorithm. |

Default recommendation: **ship public artifacts as `.hqa` once the format is
stable**, but keep `.tq` as the developer extension until real Hawking model
releases exist. `.hqa` should contain:

- payload kind: `strand1`, `strand2`, `q4k`, `mixed`, etc.
- source model id and hash
- bake command and git revision
- hardware target profile
- bits-per-weight ledger including sideinfo/outliers
- eval ledger hash
- compatibility flags
- deterministic decode metadata

This gives "Hawking Quant" a public story without prematurely erasing the
STRAND/TQ implementation.

## Deep Code Rename Map

| Current | Future | Notes |
|---|---|---|
| `crates/dismantle` | `crates/hawking` or `crates/hawking-cli` | Binary package. Prefer `hawking` if the crate is only CLI. |
| `crates/dismantle-core` | `crates/hawking-core` | This is the largest import rewrite. |
| `crates/dismantle-serve` | `crates/hawking-serve` | Endpoint aliases should land before crate move. |
| `crates/dismantle-bench` | `crates/hawking-bench` | Preserve benchmark labels for before/after comparability. |
| `dismantle_core::` | `hawking_core::` | Mechanical Rust import rewrite. |
| `dismantle_serve::` | `hawking_serve::` | Mechanical Rust import rewrite. |
| `#[command(name = "dismantle")]` | `#[command(name = "hawking")]` | Add alias/subcommand shim first. |
| `target/release/dismantle` | `target/release/hawking` | Keep symlink or second bin target. |
| `DISMANTLE_QWEN_*` | `HAWKING_QWEN_*` | Build env resolver with precedence + warning. |
| `DISMANTLE_FORCE_CPU` | `HAWKING_FORCE_CPU` | Keep old alias because scripts use it heavily. |
| `DISMANTLE_KERNEL_PROFILE` | `HAWKING_KERNEL_PROFILE` | Profiles are public surface. |
| `DISMANTLE_HOME` | `HAWKING_HOME` | Migrate `tools/headbank`. |
| `/v1/dismantle/generate` | `/v1/hawking/generate` | Keep old route. |
| `/v1/dismantle/tokens` | `/v1/hawking/tokens` | Keep old route. |
| `dismantle_*` metrics | `hawking_*` metrics | Emit both for one compatibility window. |
| `dismantle-headbank-manifest-v1` | `hawking-headbank-manifest-v1` | Reader must accept both. |
| `docs/dismantle...` wording | `Hawking` | Docs pass after code aliases are green. |
| GitHub repo | `hawking` or `hawking-ai` | Rename only after package install path is stable. |

## Migration Phases

### Phase 0 - Naming Freeze, No Code Rename

Deliverables:

- This plan.
- A single glossary: Hawking, Dismantle, STRAND, TQ, HQA.
- A public positioning paragraph.
- A list of all old-name surfaces from `rg`.

Rule: no directory moves yet.

### Phase 1 - Compatibility Layer

Add new names without removing old ones:

- Add `HAWKING_*` env resolution with `DISMANTLE_*` fallback.
- Add `/v1/hawking/generate` and `/v1/hawking/tokens` aliases.
- Add `hawking_*` metrics while still emitting `dismantle_*`.
- Add `HAWKING_HOME` while reading old home dirs.
- Add `hawking-headbank-manifest-v1` reader/writer, old schema reader.
- Add CLI wrapper or second bin target named `hawking`.

Gate:

```bash
cargo test --workspace
rg -n "HAWKING_|/v1/hawking|hawking_" crates tools docs
```

### Phase 2 - Public Docs and Model Naming

Update public-facing docs only:

- README top-level identity becomes Hawking.
- Old Dismantle section becomes "origin / formerly".
- Model release docs use `Hawking-*`.
- Launch post draft becomes Hawking-oriented.
- Bench reports keep old command names only where reproducing old results.

Gate:

- No public release doc should present `dismantle` as the current brand.
- Reproduction commands can still use `dismantle` aliases until binary rename
  lands.

### Phase 3 - Crate and Module Rename

Do one mechanical commit with no behavior changes:

1. Move crate directories.
2. Rename package names in `Cargo.toml`.
3. Rewrite Rust imports.
4. Rewrite integration test crate imports.
5. Regenerate lockfile.
6. Keep old binary alias.

Gate:

```bash
cargo test --workspace
cargo build --release --workspace
./target/release/hawking --help
./target/release/dismantle --help
```

### Phase 4 - Tooling and Script Rename

Update shell/python tooling:

- `tools/headbank` env and schema names.
- `tools/bench` default binary path.
- `tools/spec` scripts.
- `tools/orchestrator` capture scripts.
- profile JSON env references.
- launch/runbook docs.

Rule: old env vars remain accepted. Scripts should prefer new names.

### Phase 5 - Quant Artifact Public Rename

Only after TQ serving is real:

- Define `HQA1` public archive schema.
- Add `.hqa` writer that wraps existing STR2/TQ payloads.
- Keep `.tq` reader for developer artifacts.
- Add conversion command:

```bash
hawking quant wrap --in model.tq --out model.hqa --eval-ledger eval.jsonl
```

Gate:

- `.hqa` can round-trip into the runtime.
- `hawking inspect model.hqa` prints source hash, bpw, tensor map, eval result,
  hardware target, and compatibility.

### Phase 6 - Repo Rename

Only after users can install and run `hawking`:

- Rename GitHub repo.
- Update remote URL in docs.
- Update badges and package metadata.
- Preserve redirects where platform allows.
- Tag one release as `dismantle-final-alias`.

## Compatibility Rules

1. Old names must never silently break a benchmark or model card.
2. New names win if both old and new env vars are set.
3. Every deprecation warning must print once, not per token.
4. Old native endpoints remain aliases until a public stable Hawking release is
   at least one release old.
5. Benchmark reports must state which binary name and git revision produced the
   numbers.

## Risk Register

| Risk | Mitigation |
|---|---|
| Massive import churn hides behavior changes | Phase 3 must be mechanical only, no logic edits. |
| Env var aliasing changes perf profiles | Add unit tests for old/new precedence. |
| Metrics dashboards break | Emit old and new metrics during transition. |
| Model cards become unreproducible | Keep old command aliases and exact git tags. |
| `.tq` rename happens before runtime is real | Delay `.hqa` until `ProjWeight::Tq` and eval gates pass. |
| Users confuse Hawking Runtime with Hawking Models | Separate docs: runtime, model cards, quant archives, reports. |

## Done Means

The rename is complete when a new user can:

```bash
hawking serve --weights Hawking-RWKV7-G1-0.4B-STRAND2.hqa
hawking bench --weights Hawking-RWKV7-G1-0.4B-STRAND2.hqa
hawking quant inspect Hawking-RWKV7-G1-0.4B-STRAND2.hqa
```

and all old workflows still work through compatibility aliases:

```bash
dismantle serve --weights old-model.gguf
DISMANTLE_QWEN_Q4K_PREDEC=1 dismantle generate ...
```

The public story at that point is not "Dismantle renamed." It is:

> Hawking ships measured local models and the runtime that makes them dense,
> fast, and useful on Apple Silicon.

