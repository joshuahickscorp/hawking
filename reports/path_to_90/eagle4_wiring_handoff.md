# EAGLE-4 wiring handoff — for a fresh session

**Audience**: a new session (fresh context window) that picks up after
the user has dropped the eagle4 codebase into dismantle. Everything you
need to act is in this doc.

## What's already landed (read these before touching anything)

Two commits on `claude/dreamy-golick-d54ff8`:

- `2852ede` — orphan-worktree recovery, brief realignment, max-chars fix.
- `370c47f` — eagle4 prep: `DraftHead` trait generalized for multi-hidden,
  `Eagle4Head` skeleton at `crates/dismantle-core/src/speculate/eagle4_head.rs`,
  `reports/path_to_90/eagle4_convergence.md` with the full integration plan.

The skeleton compiles. `cargo test -p dismantle-core --lib speculate::` is
19 tests, all pass. Read `reports/path_to_90/eagle4_convergence.md`
end-to-end first — it has the contract, NPZ key naming, forward
pseudocode, masked-verify kernel signature, and order-of-work.

## What this handoff is about

The user is dropping the standalone `eagle4` repo into
dismantle as a top-level `eagle4/` subdirectory. Reason: public-repo
optics. The whole stack (head training + inference runtime) should be
visible in one place.

After the drop you need to:

1. Refresh path references in dismantle's docs/code so they point at
   in-tree paths instead of the old `eagle4/...`.
2. Gitignore the large artifacts (checkpoints, training data, frozen
   weights, mlx caches) without losing the source code or the small
   results files (bench/tau JSON).
3. Wire a parity test that loads `eagle4/checkpoints/best.npz` and
   diffs the Rust head forward against a Python-side reference at
   atol=1e-3 fp16.
4. Commit.

## Step 1 — verify the drop

After the user has done the rsync/copy, expect:

```
eagle4/                         (top-level, peer to crates/, tools/)
├── ARCHITECTURE.md
├── LICENSE
├── README.md
├── bench.py
├── bench_results.json
├── bench_results.md
├── capture.py
├── eagle4.py
├── pyproject.toml
├── q4_parity.py
├── tau_eval.py
├── tau_results.json
├── tests/
│   └── test_smoke.py
├── checkpoints/                ← gitignored, ~49 GB
├── data/                       ← gitignored, ~22 GB
├── models/                     ← gitignored
├── v2lite_frozen.npz           ← gitignored, 800 MB
├── .venv/                      ← gitignored
└── __pycache__/                ← gitignored
```

Sanity-check the drop landed cleanly:

```bash
ls eagle4/eagle4.py eagle4/capture.py eagle4/ARCHITECTURE.md eagle4/bench_results.md
ls eagle4/checkpoints/best.npz  # should exist locally; gitignored
```

If the user dropped it somewhere else (e.g. `tools/eagle4/` or
`crates/eagle4-core/`), adjust the paths below — but `eagle4/` at top
level is the recommended location and the rest of this doc assumes it.

## Step 2 — gitignore preflight

Dismantle's `.gitignore` already covers `/reports/` (we force-add).
The `eagle4` large-file paths need their own entries. **This commit
adds them pre-emptively** (see "Pre-staged in this handoff" below) so
the drop doesn't spam `git status` with 70 GB of files.

If for any reason those entries didn't make it in, append to `.gitignore`:

```gitignore
# eagle4 large artifacts (regeneratable, document-but-don't-commit)
/eagle4/.venv/
/eagle4/__pycache__/
/eagle4/checkpoints/
/eagle4/data/
/eagle4/models/
/eagle4/v2lite_frozen.npz
/eagle4/*.log
/eagle4/.pytest_cache/
/eagle4/.ruff_cache/
```

Eagle4's own `.gitignore` (its repo-internal one) gets dropped along
with the rest of the tree. That's fine — git respects nested
`.gitignore` files. We add the dismantle-side entries as a belt-and-
suspenders measure for paths that absolute against dismantle's repo root.

## Step 3 — what to commit from the drop

```
# Commit (source code + small artifacts)
git add eagle4/eagle4.py eagle4/capture.py eagle4/bench.py
git add eagle4/tau_eval.py eagle4/q4_parity.py
git add eagle4/tests/
git add eagle4/pyproject.toml eagle4/LICENSE
git add eagle4/README.md eagle4/ARCHITECTURE.md
git add eagle4/bench_results.md eagle4/bench_results.json
git add eagle4/tau_results.json
git add eagle4/.gitignore  # eagle4's own
# Force-add if V3.md / V4.md exist (might not, depending on drop)
git add -f eagle4/V3.md eagle4/V4.md 2>/dev/null || true

# Verify nothing huge slipped in
git status --short
git ls-files eagle4/ | xargs ls -la | awk '{print $5, $9}' | sort -n | tail
```

The last line shows the biggest files staged. If any are > 1 MB,
inspect — probably should be gitignored.

## Step 4 — find/replace pass on in-tree paths

After the drop, four files in dismantle reference the old external
path. The find/replace list (pre-grepped at handoff time):

| File | Lines |
|---|---|
| `reports/path_to_90/eagle4_convergence.md` | 4, 172, 277, 311 |
| `reports/path_to_90/NEXT_SESSIONS_BRIEF.md` | 8, 20, 22 |
| `crates/dismantle-core/src/speculate/eagle4_head.rs` | 4, 40, 56, 69, 168 |
| `crates/dismantle-core/src/speculate/draft_head.rs` | 20 |

Most are simple `s,eagle4,eagle4,g` (the convergence doc
references `eagle4/checkpoints/best.npz` → `eagle4/checkpoints/best.npz`).

Recommended approach:

```bash
# Dry-run: see what would change
grep -rln "eagle4\|eagle4" reports/ crates/

# Apply
find reports/ crates/ -type f \( -name "*.md" -o -name "*.rs" \) \
  -exec sed -i '' 's|eagle4|eagle4|g' {} \;
find reports/ crates/ -type f \( -name "*.md" -o -name "*.rs" \) \
  -exec sed -i '' 's|eagle4|eagle4|g' {} \;

# Verify
grep -rn "eagle4" reports/ crates/  # should print nothing
```

There are also a few references that don't have the `~/Downloads/`
prefix but talk about "the eagle4 repo" or "standalone repo" — those
are now wrong; eagle4 is no longer standalone, it's in-tree. Re-grep
and rewrite as appropriate:

```bash
grep -rn "standalone repo\|eagle4 repo\|sibling private repo" reports/ crates/
```

## Step 5 — wire the parity test

A test stub is pre-staged at `crates/dismantle-core/tests/eagle4_parity.rs`
(see "Pre-staged in this handoff" below). It currently `#[ignore]`s
itself and contains the contract the new session needs to fill in.

The test's job: confirm dismantle's Rust head forward matches eagle4's
Python head forward at atol=1e-3 fp16 on a small fixed-seed input.
Mechanics:

1. Load `eagle4/checkpoints/best.npz` via `Eagle4Head::from_npz()`
   (needs implementing — that's the next-session work).
2. Pick a deterministic 10-record subset from any eagle4 data shard
   (or a generated synthetic fixture if eagle4's data isn't checked in).
3. Run Rust forward on those records → emit token_logits, mask_logits,
   calib_logit per record.
4. Run Python reference forward via `python eagle4/eagle4.py eval
   --ckpt eagle4/checkpoints/best.npz --frozen eagle4/v2lite_frozen.npz
   --parquet <fixture>.parquet --max-records 10 --json-out /tmp/ref.json`
   in a subprocess.
5. Diff outputs at atol=1e-3 fp16. Token argmaxes must match exactly
   (they're discrete). Mask logits and calib_logit are fp32 — atol 1e-3.

The test gates on env `EAGLE4_PARITY_TEST=1` so CI without the
checkpoint locally still passes. Run locally:

```bash
EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --test eagle4_parity -- --ignored
```

## Step 6 — commit

```bash
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' commit -m "
path-to-90 eagle4 drop: source code in-tree + path refresh + parity wiring

EAGLE-4 (formerly in eagle4) is now in-tree at eagle4/ for
public-repo optics. Source code + small results files committed; large
artifacts (checkpoints, training data, frozen weights) gitignored —
regeneratable via eagle4/eagle4.py frozen + eagle4/capture.py.

  - Drop: eagle4/{eagle4.py, capture.py, bench.py, tau_eval.py, q4_parity.py,
    pyproject.toml, README.md, ARCHITECTURE.md, bench_results.md,
    bench_results.json, tau_results.json, tests/, LICENSE}
  - Gitignore: eagle4 large artifacts (49 GB checkpoints, 22 GB data,
    800 MB frozen)
  - Path refresh: eagle4 → eagle4 in convergence doc, brief,
    Eagle4Head module comments
  - Parity test: crates/dismantle-core/tests/eagle4_parity.rs (gated on
    EAGLE4_PARITY_TEST=1)
"
```

Do NOT add Co-Authored-By or Generated-with lines per the user's
project rule.

## Pre-staged in this handoff

To save the new session some keystrokes, this commit also lands:

1. **`.gitignore` entries** for eagle4 large artifacts. Pre-emptive; the
   drop won't spam `git status`.
2. **`crates/dismantle-core/tests/eagle4_parity.rs`** — test scaffold,
   `#[ignore]`'d, with the contract the new session fills in. cargo
   check passes.
3. **This doc** itself, so the next session has a single starting
   point.

After the drop, the new session's flow is:

```
1. read reports/path_to_90/eagle4_wiring_handoff.md
2. read reports/path_to_90/eagle4_convergence.md
3. follow steps 1-6 above
4. ship.
```

## Estimated wall time for the new session

- Steps 1-4 (verify drop, gitignore, commit source, path refresh): ~30 min
- Step 5 (parity test wiring + implementing `from_npz`): ~half day
- Step 6 (commit): ~5 min

After this, the next session works on `Eagle4Head::propose` forward
pass (CPU first for parity, then Metal). That's a separate half-day.

## Sanity checks after this commit

- `cargo check` clean (10 sec, no codegen)
- `cargo test -p dismantle-core --lib speculate::` shows 19 tests pass
- `cargo test -p dismantle-core --test eagle4_parity` shows 1 test
  ignored (the parity test gated on env)
- No churn in `training_data/c2_hidden/eagle3_v0/` (the paused capture's
  log file is still in working dir but not staged)

## Status of running work (so it's not lost in transit)

- The eagle3 capture (PID 40658 → killed cleanly during pause) is at
  ~85/100000 samples on disk. Resumable via
  `./tools/training/launch_main_capture.sh` per the recovery commit.
  **Decision pending** (see eagle4_convergence.md § "Decision points
  outstanding"): cancel or finish? Eagle4 already has the eagle3
  baseline at 75.84% accept, so finishing doesn't add new info.
- `tools/training/mlx_eagle/` is the dismantle-side training stack
  that eagle4 now supersedes. Pending retirement decision.

Both are punted to the user; neither blocks the wiring work.
