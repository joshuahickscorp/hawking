# CLAUDE.md — agent operating contract for dismantle

You are the autonomous agent executing dismantle hauls. Read this
**before** every haul and obey it without deviation. The contract is
short on purpose; the spec, manifest, and brief carry the rest.

## Identity & authorship

All commits made during a haul are authored by Joshua Hicks via
inline git options:

```
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' commit -m "..."
```

**Never** run `git config` to set the identity globally. The
inline-only rule keeps the user's machine config untouched and makes
authorship intentional per commit.

## Scope rule

The manifest (`_phaseN_haul_manifest.md`) is the **authoritative
scope**. Do not implement, refactor, or "improve" anything outside
the items it lists. If you notice a bug or smell mid-haul, log it
in the active blocked-doc-or-closeout for the next attended session
to triage. Do not fix it in this haul.

## Halt rule (the most important rule)

When an item fails — whether by validator non-zero exit, evidence
attestation false, memory-pressure persistent, or RSS sentinel
firing — you:

1. Write the blocked-doc stub (the runner does this for you with the
   manifest's `_phaseN_haul_attemptN_blocked.md` path) and fill in
   root cause + what attended work unblocks + followups.
2. Stop work on that item.
3. Apply the halt-budget rule from the spec:
   - **G1.1 (Metal scaffold) — 1 halt ends the haul.**
   - **G1.2 / G1.3 / G1.4 (GEMV ports) — 2 halts in this group ends
     the haul.** First halt: continue to next independent item.
4. Do **not** "peel-onion" fix the underlying issue. That's the
   next attended session's job. Halts surface gaps; they don't
   mid-flight repair them.

## Evidence rule

Every gate writes three JSON files to
`tools/haul/_evidence/$GATE/`:

- `pre.json` — captured by `run-gates.sh` before validator runs.
- `post.json` — captured after validator runs.
- `verify.json` — captured from a clean independent re-run of the
  validator. Contains `attestation: true|false`. **If attestation
  is false, the gate fails regardless of post.json's exit_code.**

Without all three files green, the gate has not passed. No exceptions.

## Kill Protocol rule

Before marking any lever **NO-GO** (in `reports/dead_levers.md`, a
blocked-doc, or a closeout), the kill MUST record three things:

1. **Type-1 or Type-2.** *Type-1* = died on a measured property of
   reality that no implementation cleverness changes (a delta with
   higher variance than the original; an FFN active set that is
   ~half the neurons; no hardware gather on Apple Silicon).
   *Type-2* = died only in the **form** tested, where a different
   formulation attacks the same goal (data-free vs **data-aware**,
   gather vs **gather-free**, extracted-post-hoc vs **trained-for**).
2. **The reframe considered.** Name the specific alternative
   formulation explicitly — even if only to reject it.
3. **Why the reframe also dies, OR a pointer to its oracle.** A
   Type-2 reframe is "alive" **only** with a named, cheap
   (offline / CPU NumPy-scale) oracle that could kill it.

Hard rules:
- **Never resurrect on vibes.** No nameable kill-oracle → the lever
  stays dead.
- **Never re-test a recorded Type-1 kill.** Its death is a fact
  about reality, not about our effort.
- **Accept Type-1 deaths.** Do not manufacture a reframe to avoid a
  kill. The point is to catch the *rare* genuine Type-2, not to pad
  the ledger — "all of these are genuinely Type-1" is a correct and
  common answer.

The worked example (the four Phase-A kills, classified and
retro-filled) lives in `plans/throughput_bible_2026_05_30.md`
§8.3.1 "Kill Protocol".

## Memory-coexist rule (Phase 1 specific)

dismantle hauls may run concurrently with another GPU/RAM-heavy
process (slm training, etc.). Historical co-existence rules live in
`docs/archive/phase-history/_phase1-spec.md § Co-existence mode`:

- **Probe before every item:** call `tools/haul/coexist.sh probe`.
  Exit 0 = safe; 1 = degraded (sleep 30s × up to 5 times before
  halting); 2 = critical (wait 5 min for recovery, else halt with
  `reason: memory_pressure_critical`).
- **Active modulation while a gate runs:** `tools/haul/coexist.sh
  watch` (started automatically by `coexist.sh launch`) polls the
  probe every 10s and SIGSTOP/SIGCONTs the active gate's process tree
  on degraded/safe transitions, so dismantle yields CPU/GPU back to
  slm during pressure spikes without losing state. Pause/resume
  timeline lands in `_evidence/<gate>/throttle.log`.
- **Cooperative scheduling:** every dismantle subprocess in this
  haul is launched with `nice -n 19 taskpolicy -b ./target/release/dismantle ...`.
  This marks dismantle as background QoS so the foreground process
  (slm) gets first dibs on CPU and Metal resources.
- **RSS sentinel:** if any dismantle process hits > 5 GB RSS, that's
  a regression (Phase 0 baseline ~2 GB). Halt the item, write
  blocked doc with `reason: rss_ceiling`.
- **Synthetic-first validation:** prefer `tests/correctness/phase1_kernel_parity.rs`
  (synthetic small fixed inputs, no model load) over integration
  smoke (model load + generate). Run the integration smoke only if
  `coexist.sh probe` returns 0; record `PASS-PARITY-ONLY` if skipped.
- **Inter-item cool-down:** `run-gates.sh` sleeps 30s between items.
  Don't shortcut this.
- **Reduced token validation:** 3 tokens, not 5, when running token
  regression. Greedy temp=0 is deterministic; 3 is sufficient.

## Verification rule

Numerical-correctness gates (Metal kernel parity vs CPU reference)
check `atol=1e-3` fp16 on a fixed-seed input. Tighter than that is
not required (fp16 quantization noise is around 1e-3). Looser is
forbidden — the parity regime is tight enough that real bugs
produce diffs orders of magnitude beyond 1e-3.

Token-output gates check first **3** token IDs match the locked
baseline at `tests/golden/_phase1_token_baseline.hashes`. Mismatch = halt.

Smoke gates check non-empty UTF-8 stdout from `dismantle generate`
with exit code 0. Empty stdout or non-zero exit = halt.

## Time rule

Per-item soft ceiling: **60 min** (45 implementation + 15
validation). Haul hard ceiling: **4 hr**. On hard ceiling, halt
cleanly with `reason: ceiling`. The blocked doc lists what was in
flight when the ceiling fired; the next attended session re-scopes.

## Build hygiene rule

Every haul item that touches source code:

1. Runs `cargo build --release --workspace` before validation. If
   it fails, halt; do not run validators on a broken build.
2. Runs `cargo test --workspace --lib` after the impl change. The
   pre-existing 15 tests must still all pass.
3. Runs the gate's own validator (e.g. parity test).
4. Commits **only the files prescribed by the manifest item**. No
   sweep-commits.

## Single-purpose commit rule

Each item lands in **exactly one commit** with the manifest-
prescribed subject (e.g., `phase 1: G1.1 Metal scaffold + rmsnorm
round-trip`). On halt, write a halt commit with subject
`phase 1: HALT — G1.X: <root cause>` and body explaining what was
attempted. No squashing, no rebasing during a haul. The git log
is the audit trail.

## What the agent does autonomously

- Read the manifest + brief + spec.
- Execute manifest items in order.
- Run cargo build / cargo test and parse outputs.
- Generate baseline hashes via `tools/haul/capture-baseline.sh`.
- Write evidence JSON triples (pre/post/verify) for each gate.
- Apply patches if the manifest prescribes them.
- Author commits with the inline-git-identity rule.
- On halt: write the blocked doc with root-cause analysis.
- On haul completion: write `_phaseN_closeout.md` summarizing
  outcomes per gate.

## What the agent does NOT do autonomously

- Modify the spec, manifest, brief, CLAUDE.md, or ROADMAP.md
  mid-haul. Those are attended-session products.
- "Just one more fix" beyond what's in the active item.
- Skip `nice -n 19 taskpolicy -b` on dismantle subprocesses.
- Continue past the halt-budget threshold.
- Run model training or any heavy-RAM Python work.
- Commit large new dependencies (Cargo.toml additions need attended
  approval).
- Push to remote. Commits stay local; user decides when to push.

## Reproducibility

Every commit landed during a haul carries a baseline anchor: the
relevant entry in `tests/golden/_phase1_kernel_baseline.hashes` and/or
`tests/golden/_phase1_token_baseline.hashes` is regenerated and committed with
the kernel change. Future hauls verify against those baselines so
regressions are caught at the parity test layer before reaching
integration.

## Tone of artifacts

Blocked docs and closeouts go straight to the point. Format:

```
# Phase 1 haul attempt N — [BLOCKED|CLOSEOUT]
**Halted at:** <iso8601>
**Halted on:** <gate-id>
## Root cause
<one paragraph>
## What ran
<bullet list with hashes>
## What attended work unblocks
<one paragraph + concrete files to inspect>
## Followups
<list>
```

No prose padding, no apologetic tone, no "we should consider…". The
audit trail is for the next session to act on, not to read.

## On disagreement with this contract

If a haul situation seems to require breaking a rule (e.g., "the
fix is one line, why not just do it?"), the answer is always: write
a blocked doc with the rule violation as a *proposed* unblock. The
next attended session reviews and either adopts the change into the
contract or rejects it. The contract evolves through explicit
attended changes, never through autonomous "just this once"
exceptions. Phase 0 of dismantle taught us that mid-flight scope
creep (e.g., "while I'm here, fix the tokenizer too") is how 4-hour
hauls become 12-hour debugging spirals.
