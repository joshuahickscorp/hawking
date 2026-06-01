# CLAUDE.md — agent operating contract for dismantle

You are an agent working in the dismantle repo (a pure-Rust + Metal
inference engine for Apple Silicon — see [README.md](README.md) and
[ARCHITECTURE.md](ARCHITECTURE.md)). Read this before you start and
obey it. It is short on purpose; the plans, the kill-ledger, and the
bench harness carry the rest.

## Identity & authorship

All commits are authored by Joshua Hicks via inline git options:

```
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' commit -m "..."
```

**Never** run `git config` to set identity globally — the inline-only
rule keeps the machine config untouched and makes authorship
intentional per commit. **Never** add any AI/Claude attribution: no
`Co-Authored-By: Claude`, no "Generated with Claude Code" footer, no
mention in PR bodies. The commit message ends after its content.

**Commits stay local.** Do not push to remote unless the user asks.

## Scope discipline

Do one change at a time. Do not implement, refactor, or "improve"
anything outside the active task. If you notice a bug or a smell
mid-task, log it in the active report/closeout (or surface it to the
user) for a separate pass — do not fix it inline. Phase 0 taught us
that mid-flight scope creep ("while I'm here, fix the tokenizer too")
is how a 1-hour task becomes a 12-hour debugging spiral.

## Correctness gate (parity-first)

A source change to a kernel or a decode path is not done until its
parity test is green. The parity tests live in
`crates/dismantle-core/tests/*.rs` (e.g.
`crates/dismantle-core/tests/phase1_kernel_parity.rs`,
`q4k_batched_mma_parity.rs`, `prefix_cache_parity.rs`); the
HTTP/server tests live in
`crates/dismantle-serve/tests/http_integration.rs`. Golden baselines
(kernel + token hashes) live in `tests/golden/*.hashes`.

- **Numerical parity:** Metal kernel vs CPU reference checks `atol=1e-3`
  fp16 on a fixed-seed input. That is the floor — fp16 quant noise is
  ~1e-3; real bugs produce diffs orders of magnitude larger. A
  reduction-reorder kernel (e.g. MMA) may add `rtol=1e-4` on top, never
  loosen `atol`.
- **Token-output parity:** first 3 greedy (temp=0) token IDs must match
  the locked baseline. Greedy is deterministic; 3 is sufficient.
- **Bit-identity:** a lever claimed "bit-identical" must produce
  byte-identical decoded output (b3sum) vs the feature-off path on a
  real model — verify it, don't assert it. `dismantle batch-hash` runs
  many prompts through one model load for this.

**Correctness before performance, always.** A faster kernel that fails
parity is a regression, not a win. When a worktree agent reports
"parity passed," re-run the parity/bit-identity check yourself in the
target branch before trusting it.

## Kill Protocol (mandatory before any NO-GO)

Before marking any lever **NO-GO** (in the kill-ledger, a report, or a
closeout), record three things:

1. **Type-1 or Type-2.** *Type-1* = died on a measured property of
   reality no implementation cleverness changes (bandwidth ceiling, no
   hardware gather on Apple Silicon, uniform sensitivity). *Type-2* =
   died only in the **form** tested, where a different formulation
   attacks the same goal (data-free vs data-aware, gather vs
   gather-free, extracted-post-hoc vs trained-for).
2. **The reframe considered** — name the specific alternative
   formulation, even if only to reject it.
3. **Why the reframe also dies, OR a pointer to its oracle.** A Type-2
   reframe is "alive" only with a named, cheap (offline / CPU
   NumPy-scale) oracle that could kill it.

Hard rules: never resurrect on vibes (no nameable kill-oracle → stays
dead); never re-test a recorded Type-1 kill (its death is a fact about
reality); accept Type-1 deaths — do not manufacture a reframe to avoid
a kill.

The protocol + worked example is `plans/bible_archive.md` §8.3.1. The
**canonical kill-ledger** — every dead lever with its
classification, evidence, and resurrection check — is
`reports/dead_levers.md`. Read a lever's resurrection check before
re-spawning it; update the ledger when a new lever dies.

## Bench discipline (contamination is real)

A running Claude session inflates `dec_tps` 4–5×. So:

- **Absolute throughput numbers require a clean room.** Quit Claude and
  run `tools/bench/clean_room_batch.sh` (or `clean_room_queue.sh` for
  the deferred-absolute queue). The honest Qwen2.5-3B-Q4_K_M anchor is
  **~31 dec_tps** on M3 Pro; treat anything far above that as
  contamination.
- **Paired A/B deltas are contamination-robust** — the inflation
  cancels in the relative number, so a paired lever bench
  (`tools/bench/paired_lever.sh`, `coexist_bench.sh`) is valid with
  Claude open. Don't ask the user to quit for a relative gate.
- **Validate via parity + kernel-count ratios**, not raw tps, when in
  doubt. Report the full spread (range, not just the mean) and tag each
  number proxy/estimate vs measured.

## Coexistence (when another GPU/RAM-heavy process is running)

dismantle may run alongside other heavy work (model training, etc.).

- **Cooperative scheduling:** launch dismantle subprocesses with
  `nice -n 19 taskpolicy -b ./target/release/dismantle ...` (background
  QoS, so the foreground process gets first dibs on CPU/Metal).
- **RSS sentinel:** a dismantle process over ~5 GB RSS is a regression
  (Qwen-3B steady state is ~0.8–2 GB; the zero-copy loader keeps it
  near model size). Stop and investigate.
- **Don't run heavy-RAM Python** (training, large corpus jobs) as part
  of an autonomous pass without the user's say-so.

## Build hygiene

A change that touches source:

1. `cargo build --release --workspace` before any validation. If it
   fails, stop — do not run validators on a broken build.
2. `cargo test --workspace --lib` after the change (currently ~94
   dismantle-core / 9 dismantle-serve / 5 dismantle-bench lib tests;
   they must stay green).
3. The change's own parity/integration test (see Correctness gate).
4. Commit only the files the task prescribes — no sweep-commits.

## Commits

Each logical change is one commit with a clear, human subject (no AI
attribution, per Identity). No squashing/rebasing of already-landed
work mid-pass — the git log is the audit trail. Cargo.toml dependency
additions need the user's approval (don't pull in large new deps
autonomously).

## Repo workflow notes

- The old `tools/haul/` runner (manifests, evidence-triples, halt-budget
  gates) is **retired**. The live workflow is the bench/oracle harness
  in `tools/bench/*` plus the parity tests in
  `crates/dismantle-core/tests/`.
- `reports/`, `colab/`, `silicon-builds/`, `models/`, `.claude/` are
  **gitignored**. Working notes and oracle outputs live in `reports/`
  on disk; when a report is meant to ship as a durable record (e.g. the
  kill-ledger), `git add -f` it deliberately.
- Plans live in `plans/`; the operative decode strategy is
  `plans/bible_active.md` (lean) with `plans/bible_archive.md` as the
  fence store. Memories (cross-session facts) are in the agent memory
  dir, indexed by `MEMORY.md`.

## On halt

If a task can't complete cleanly (a gate fails, a precondition is
missing, the build breaks and the fix is out of scope), **halt and
write it up** — a short blocked-doc / closeout with root cause, what
ran (with hashes/commits), what unblocks it, and followups. A clean
logged halt is a success. Do not "peel-onion" fix the underlying issue
mid-pass; that surfaces a gap for the next attended session, it does
not repair it now.

## On disagreement with this contract

If a situation seems to require breaking a rule ("the fix is one line,
why not just do it?"), the answer is: write it up as a *proposed*
change for the user to adopt or reject. The contract evolves through
explicit attended changes, never through autonomous "just this once"
exceptions.
