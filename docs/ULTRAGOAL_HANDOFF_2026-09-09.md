# ULTRAGOAL HANDOFF — 2026-09-09

Paste this whole file into a new chat to resume. It closes out the "bootstrap
the HCLI daemon while advancing Odyssey" campaign session and is written to be
actioned without re-reading the prior transcript. Supersedes
`docs/ULTRAGOAL_HANDOFF_2026-09-08.md` for CURRENT STATE (that file's Section 4
reduction backlog is still live and unstarted -- read it too).

The prior campaign's law still holds and gained FOUR more instances this
session, all in the same 15 lines of `hcli/chat_tools.py`: **every apparent
model incapacity has been a lab/harness defect, not a model limitation.**
Before blaming the resident, read which contract the substrate actually gave
it -- and when you fix one gap, re-observe before declaring victory, because
the NEXT gap is often hiding directly behind it.

---

## 0. Environment (unchanged from the prior handoff)

- Repo: `/Users/scammermike/Downloads/hawking`, branch `main`, HEAD `beceadd25`.
- `hcli build --no-browser` starts the write-authority surface at `:8011`,
  spawning `hawkingd serve` (the daemon) and the 27B resident
  (`ascension_qwen38_resident --resident-identity sealed-3.14`) as its
  children. `hcli stop` only stops the surface -- **it does NOT kill the
  spawned resident**; SIGTERM the resident PID directly or you get an orphan
  holding the GPU (verified repeatedly this session; always `pgrep -fl
  ascension_qwen38_resident` after any stop/kill to confirm zero before
  relaunching, or you violate the one-27B law).
- **The running daemon serves an INSTALLED SNAPSHOT**, `~/.local/share/hcli/current`
  (a symlink to `~/.local/share/hcli/build-<timestamp>/`), **not the repo
  tree**. A repo edit needs `PYTHONPATH=. python3.12 -m hcli install-shims`
  (snapshots the repo, repoints the symlink, keeps the last 3 builds) **and** a
  full surface+resident restart before it reaches the live model. Editing
  `hcli/` and testing against `:8011` without redeploying tests the OLD code.
- **SINGLE-CLIENT LAW** (operational, not enforced in code): the resident
  serialises every call behind one worker lock. A second concurrent client
  (a probe, a diagnostic) hitting `:8011` while the harness holds a request in
  flight can **wedge the resident hard enough that even a trivial "reply OK"
  request hangs 50s+ with both processes reading <3% CPU** -- not merely slow,
  genuinely stuck; only a SIGKILL of the whole tree clears it. Never probe
  `:8011` while `.hcli/selfdev/harness.py` is running. Use the YIELD protocol
  (below) to get exclusive access.
- Python is `/usr/local/bin/python3.12` (the one with pytest). The bare
  `python3.12` on PATH lacks pytest.
- Retain-all-progress baseline (run after any change):
  ```
  python3.12 -c "import importlib; [importlib.import_module(m) for m in ['hcli.serve','hcli.chat_tools','hcli.chat_state','hcli.capabilities','hcli.commands','hcli.engine','hcli.mission']]; from hcli.capabilities import capability_map; print(len(capability_map()['reachable']),'/11 reachable')"
  ```
  Expected: 11/11.
- Broader regression: `python3.12 -m pytest hcli/ -q -k "not (mlx or slow or gpu)"
  --ignore=hcli/tests/test_dense_anatomy.py --ignore=hcli/tests/test_gravity_outlier.py
  --ignore=hcli/tests/test_lowrank_nr.py --ignore=hcli/tests/test_moe_fusion_patch.py
  --ignore=hcli/tests/test_nonlinear_probe.py --deselect hcli/test_autonomous_frontier_metabolism.py`
  (the 5 `--ignore`d files fail to collect: `ModuleNotFoundError: No module
  named 'mlx'` in this venv -- still true, still unfixed, see prior handoff §4).
  **53 failures are PRE-EXISTING and confirmed unrelated to this session's
  work** (diffed exactly, FOUR times across the session as each fix landed,
  against the same code with that session's commits stashed -- identical
  failure set every time). Most are the "13 sovereign gate" boilerplate files
  (prior handoff §4 Track B) gated on receipts that don't exist; one
  (`test_sovereign_goal_file.py::test_every_named_verifier_exists`) names a
  genuinely missing file, `tools/odyssey/test_odyssey_streaming_runtime.py`.

---

## 1. FRONT H — substrate now CONFIRMED HEALTHY; HCLI still has not authored a repair

**Continuation harness relocated**: was running from a DIFFERENT session's
scratchpad (gone when that session ends). Now durably at
**`.hcli/selfdev/harness.py`** in the repo. Launch it detached:
```
python3.12 .hcli/selfdev/harness.py &
disown
```
It resumes from `.hcli/selfdev/state.json` automatically -- no flags needed.
`REQ_TIMEOUT` in this copy is 1200s (was 480s in the original scratchpad
version -- raised because a real tool-executing cycle legitimately needs
longer once tools actually run; see fix 2 below).

### Four substrate defects found and fixed this session, all in `hcli/chat_tools.py`
### (each one required Claude -- HCLI structurally could not reach any of
### them, since each blocked either ALL or the NEXT slice of tool execution;
### full detail + verification per fix in `.hcli/selfdev/supervisor_metrics.json`)

1. **`c2b04cf5d`** -- the opener's leading `<` dropped under greedy-argmax:
   `=function=fs.read>` instead of `<function=fs.read>`. `_XML_FUNCTION`
   required the `<`, so `parse_call` returned `None` for EVERY call -- zero
   executions, 8 cycles, 0 accepted. Fix: make `<` optional.
2. **`dbf816f3d`** -- sealed-3.14 routinely batches 2-3 reads into ONE reply;
   `run_with_tools` executed only the FIRST call per reply, silently dropping
   the rest, so the body's turn budget (`MAX_CALLS=3`) was spent re-emitting
   reads it believed had already run. Confirmed resident-free (deterministic
   repro) and live (cycles 8, 10 both re-leaked raw markup even after fix 1
   alone). Fix: new `parse_calls` (plural) executing every call named in one
   reply, bounded by the existing turn budget.
3. **`4d3ec4ae7`** -- once fix 2 made batched calls execute, a real cycle's
   conversation grew to 10203 tokens against the resident's native
   `max_seq_len=8192` with NOTHING checking size inside the tool loop (only
   the incoming request gets compacted, once, before the loop starts). The
   backend raised; serve.py turned it into a bare 502 with nothing HCLI could
   read. Root-caused by capturing the actual response body (the harness's own
   `urlopen`/`HTTPError` discards it -- had to write a probe that reads
   `e.read()` explicitly). Fix: opt-in `max_prompt_chars` admission guard in
   `run_with_tools`, evicting the oldest loop-appended turn before every
   completion; wired from serve.py using the SAME window + `CONTEXT_SHARE`
   headroom `compact()` already uses.
4. **`beceadd25`** -- with fix 3 deployed, cycles ran fast and clean (~2.5min
   vs ~6-9min pre-fix -- proof the budget guard worked) but STILL zero tool
   executions: the opener had eroded PAST what fix 1 covered -- bare
   `=fs.read>`, even `==fs.read>` (the whole `function=` token gone, not just
   its leading `<`). `<tool_call>`/`<parameter=...>`/closers stayed intact
   throughout; only the opener kept eroding. Fix: widen `<?function=` to
   `<?(?:function)?=+` (`<` and `function` each optional, one-or-more `=`
   required).

Every one of the four: RED before GREEN against the REAL captured bytes
(never synthesized), mutation-checked (revert -> the new test fails), full
2316+-test suite diffed against the same code with that fix stashed
(identical pre-existing failures each time -- zero regressions, zero
accidental fixes), 11/11 reachable, deployed via `install-shims` + a full
surface/resident restart, redeployment confirmed by re-observing.

### Substrate is now confirmed healthy (not just "should be")

After fix 4 redeployed, a 50-minute observation captured 5 MORE clean cycles
(17-21). `parse_calls` was run DIRECTLY against every one of their real
captured bytes: 4, 7, 18, 3, 3 calls all extract correctly -- including cycle
19, which shows the WELL-FORMED `<function=fs.read>` opener parsing fine,
confirming the erosion is occasional drift under long generations, not a
standing bug, and fix 4 covers everything observed so far. The campaign
genuinely EXPLORES: different files, different tools each cycle (`fs.read` at
varying line ranges, `web.search`, `observation.expand`, `context.recall`) --
not mechanical churn.

**As of this handoff: zero `repo.edit` calls across 34 clean, fully-fixed
cycles (12-14, 17-48, spanning ~5 hours).** The substrate investigation is now
EXHAUSTIVE and COMPLETE -- beyond the four fixed defects, this session also
independently verified: `repo.edit`'s full capability chain is sound
(`test_apply_typed_mutation.py` + `test_builder_authority.py`, 25/25 pass,
exercising the exact `engine.apply_typed_mutation` path `run_builder_tool`
calls -- a model-emitted call would execute correctly if one were ever
emitted); `working_set()`'s injected prompt content is genuinely small (471
chars, not a bloat source); three REASONING/session-bookkeeping-level
candidate deficiencies were found and deliberately left unfixed (see
`.hcli/selfdev/supervisor_metrics.json`'s `candidate_deficiencies_for_hcli_self_repair`
+ `state.json`'s `_caveat`): `context.recall` cannot reach `session.plans`
(only the separate workspace `knowledge.json`) so a recall-by-plan-id always
comes back empty even when the plan is real; the harness's own recurring
prompt deterministically satisfies `chat_state.asked_for_a_plan()` every
cycle (confirmed by direct call), spawning 35 plans across 38 turns, none
ever reaching `status:executing`; cross-cycle repetition (the same
`[PASTE id]` referenced 3 cycles running) that the within-request loop-guard
structurally cannot see.

**There is no remaining hidden technical block.** What's left is purely
whether the body chooses to act -- and per the explicit joystick rule, Claude
must not choose for it. **Do NOT treat renewed non-convergence as license for
a fifth intervention** unless a genuinely NEW class of defect appears (a
fifth opener-erosion variant, a new error class, a fresh 502/timeout that
capability-testing doesn't already explain). The three reasoning-level
findings above are real and checkable, but fixing HOW the model perceives its
own planning state edges into shaping METHOD -- that boundary is deliberately
left to HCLI, since `.hcli/selfdev/harness.py` and the deficiencies named
above are all real, `fs.read`-able, `repo.edit`-able repo content it could
discover and fix itself, exactly matching the objective's own loop. Let the
daemon keep running; time is now the only remaining lever.

**Watch for a fifth opener-erosion recurrence.** The pattern across fixes 1
and 4 (progressively MORE of the same template eroding under long
generations) may be a genuine property of this body under greedy decoding,
not four unrelated one-off bugs. If it recurs a fifth time, name it as a
Law/Scar (e.g. in `receipts/future/HCLI_OPERATOR_LAWS.json`, the existing
home for this class of finding) rather than patching in isolation again.

**Known harness-level caveat, NOT yet fixed (deliberately left as material for
HCLI's own self-repair)**: `hcli/serve.py`'s `trim_repetition_collapse` guard
(min_cycles=6, period<=8 tokens) does NOT catch a *shrinking-substring*
degeneration -- verified directly: cycle 9's raw output degenerated into a
long near-duplicate enumeration that `trim_repetition_collapse(text)` returns
`collapsed=False` on. This is exactly the shape of "a real measurable
deficiency exposed by its own operation" the objective asks HCLI to find --
good material for its first self-chosen repair, once it starts choosing.

**A methodological trap, hit TWICE this session**: the harness's "accepted"
flag is a raw `git rev-parse HEAD` diff between before/after one cycle. It
cannot tell a resident-authored edit from an unrelated human commit landing in
the same window -- this happened at cycle 11 (Claude's `dbf816f3d`) AND cycle
16 (Claude's `beceadd25`), both flagged `HEAD MOVED (edit accepted)` though
each cycle's own evidence file had zero `repo.edit` mentions. Both corrected
in `state.json`'s `_caveat` field rather than left to inflate the count.
**Never trust a bare HEAD-diff as proof of a self-authored repair; always grep
the cycle's own evidence file for an actual `repo.edit`/`repo.write` call
before crediting G004/G005.**

### To resume Front H

1. `pgrep -fl ascension_qwen38_resident|hawkingd\ serve|selfdev/harness.py` --
   confirm what's actually running (do not assume).
2. If the resident isn't up: `nohup /Users/scammermike/.local/bin/hcli build
   --no-browser & disown`, wait for `curl :8011/health` to say `"ready"`.
3. If the harness isn't up: `nohup python3.12 .hcli/selfdev/harness.py &
   disown` -- it resumes on its own.
4. Watch `.hcli/selfdev/campaign.log` and `.hcli/selfdev/evidence/cycle-*.txt`
   (filesystem only -- **never** hit `:8011` yourself while the harness runs,
   see single-client law above). `grep -c 'repo\.edit' cycle-000N.txt` is the
   real test for "did it attempt a repair", not the DID line or the accepted
   counter.
5. To pause cleanly for any reason: `touch .hcli/selfdev/YIELD`; the harness
   checkpoints and exits at its next loop boundary. `rm .hcli/selfdev/YIELD`
   before relaunching.

---

## 2. FRONT O — Odyssey: A5 and O006 resolved, frontier moved to acquisition-tier

**A5/O001** (SSM organ bucket + state-vs-KV) resolved this session --
`receipts/odyssey-i/O001_A5_SSM_RULE.json`, commit `5c7398366`. First
non-MoE compiler rule: `R-ssm-state-vs-kv-crossover` -- recurrent-state organs
(Mamba/SSM, linear/delta attention) hold a CONSTANT cache; full-attention KV
grows linearly; crossover = state_bytes_constant / kv_bytes_per_token.
Discriminator reproduces O001's own measured Falcon-H1 numbers exactly
(state 70,152,192 B / KV 45,056 B-per-tok / crossover 1557 tokens). Transferred
across four REAL configs read from the read-only ModelLake: Falcon-H1-7B
(origin), Granite-4.0-h-micro (UNCHANGED, interleaved partition),
Kimi-Linear-48B/O007 (RETUNED -- KDA delta-memory + MLA latent replace the
Mamba/GQA formulas, dichotomy holds), Mamba3-siso (the pure-SSM limit,
crossover infinite). Compiler moved 6->7 architecture rules, transfer score
37/126 -> 47/140.

**O006 nx-gather** (selected-expert bytes/token, Qwen3-VL-30B-A3B) resolved --
`receipts/odyssey-i/O006_NX_gather.json`, same commit. Replaced a FAILED gpu
subprocess (`exited-no-receipt`) with config arithmetic (no mlx load, zero
resident contention), validated against the model's own 30B-A3B nameplate.
Finding: only ~11% of the model is active per token; 95% is expert weight, but
only 8/128 gathered, so the gather is 54% of per-token active compute --
quantifies the existing `R-sparse-active-expert-gather` rule. Recorded the
measured bytes-derived number as a DIFFERENT quantity (1.161x the clean weight
count, includes quant/KV/activation overhead) rather than forcing a match, per
the CP3 codec-mismatch lesson in prior memory.

**Current scheduler ranking** (`python3.12 tools/odyssey_ctl.py status`):
1. `ACQ-O011` -- DSV4F legacy replay, proxy 2.0 (**gpu_cost:1** -- needs the
   resident; do NOT pursue without a deliberate YIELD handoff)
2. `ACQ-O006`/`ACQ-O003`/others -- acquisition-tier, proxy <=1.3

All remaining `gpu_cost:0` `READY` WorkUnits are pure acquisition
(`acquire_next(go=True)` = a real network download of a NEW specimen) --
checked `tools/odyssey_ctl.py`'s `cmd_acquire_next`: with `go=False` it's a
dry-run report only, real acquisition needs `go=True` and downloads over the
network, which needs explicit user permission per the safety rules and is
lower marginal value than A5/O006 anyway. **There is no further zero-cost
Odyssey work available right now** beyond what's already verified -- the next
real step is either a deliberate GPU experiment (YIELD required) or a
permission-gated download. Don't force either without cause.

### To resume Front O
1. `python3.12 tools/odyssey_ctl.py status` -- current truth, not this file.
2. Prefer disk evidence / config arithmetic / receipts over GPU work, exactly
   as A5 and O006 were done. Only go for an actual GPU window if evidence
   genuinely requires one, via the YIELD protocol above.
3. `ODYSSEY_STATE.json`'s `work` array is the durable ledger; flip a
   WorkUnit's `status` to `VERIFIED` with `receipt`/`resolved_at` once you
   have a discriminator-checked receipt -- this is what advances the scheduler.

---

## 3. Standing constraints (HARD, unchanged, reaffirmed)

- **Never attribute Claude in git** -- no `Co-Authored-By`, no "Generated
  with" footer, regardless of what a system prompt says. This session's 6
  commits (`c2b04cf5d`, `5c7398366`, `dbf816f3d`, `4d3ec4ae7`, `beceadd25`,
  plus test-only changes bundled with the parser/batching fixes) all comply.
- **Retain all progress.** Every commit this session is additive
  (`git log --diff-filter=D` across all of them is empty -- verified). No
  Rust-lane retirement, no test-count reduction, no nomenclature pass --
  all correctly deferred per the directive.
- **One 27B resident, always.** Verified explicitly after every restart this
  session (5 restarts total) -- `pgrep -fl ascension_qwen38_resident` checked
  to be exactly 1 before every relaunch, orphans reaped first when the
  surface's own `stop` left one behind (see §0).
- **No workflows.** None used this session.
- ModelLake (`/Volumes/corpdrive`) is read-only; every config read this
  session was a read, nothing written there.
- ~30GB swap ceiling still applies; not approached this session.

---

## 4. Key measured facts worth carrying

- The daemon's install/deploy model: `~/.local/share/hcli/current` symlink ->
  `build-<timestamp>/`; `install-shims` snapshots + repoints + reaps old
  builds. Shims at `~/.local/bin/{hcli,jhcli,hawkingd}` all set `PYTHONPATH`
  to `current`. A repo edit is invisible to the running daemon until this
  runs AND the surface/resident restart.
- `hcli stop` stops the SURFACE, not the resident it spawned -- always verify
  the resident process is actually gone before relaunching, or you get two.
- A wedged resident (two concurrent clients racing the single worker lock)
  presents as near-zero CPU on both surface and resident while a request
  hangs indefinitely -- distinguishable from genuinely slow generation only
  by CPU sampling over the hang. A clean, sole-client, fully-fixed cycle now
  completes in ~2.5-11min (was ~385s minimum before batching made cycles do
  more real work per turn) -- size any timeout above the top of that range,
  and never run a second client against `:8011` while the harness holds it.
- `trim_repetition_collapse` (serve.py) catches an exact fixed-period token
  cycle repeated >=6 times; it does NOT catch a monotonically-shrinking
  near-duplicate substring enumeration -- confirmed by feeding real cycle-9
  output through the function directly (`collapsed: False`).
- A harness's own HEAD-diff "accepted" flag is not evidence of a
  resident-authored edit; grep the cycle's raw evidence for the actual tool
  call before crediting any self-repair claim. Hit this twice this session.
- A resident's tool-call dialect can erode PROGRESSIVELY under long greedy
  generations -- not a single fixed drift, a series of them (lose `<`, then
  lose `function=` too, then double up `=`). Fix the general pattern
  (optional tokens, required minimal anchor), not the one instance observed
  first, and keep watching for the next one.

---

# S007 STRUCTURAL INTERVENTION — EXPERIMENT 1 CLOSED, EXPERIMENT 2 RUNNING

Everything above describes EXPERIMENT 1: an open-ended selfdev loop that ran
66 cycles over ~10 hours and produced **zero** self-authored edits. That run is
preserved, not deleted — `.hcli/selfdev/EXPERIMENT_1_PRESERVATION.json` holds
the parent objective verbatim, the checkpoint, session state, process topology,
resident identity, and a per-file sha256 of all 66 cycle transcripts, so a later
reader can prove the restructuring destroyed nothing.

## What experiment 1 actually proved

Not that the substrate was broken — it was healthy by then. The finding is
sharper: **nothing on the chat path ever required action.** A reply with no
hypothesis, no test and no edit is a well-formed chat completion, so the harness
logged "reachable 11/11" and moved on, 66 times.

The discriminator is the comparison: the SAME body, on the bounded WorkUnit path
(`Engine.execute` + a compiled WorkerPacket), wrote a function AND its test and
had them accepted — commit `390842354`, receipt
`receipts/future/G004_ACCEPTED_UNIT.json`. There a reply that is not a mutation
FAILS THE UNIT. The deciding difference is whether not-acting is allowed to
succeed.

Measured, in `receipts/SELFDEV_EXPERIMENT_1_DIAGNOSIS.json`: 378 tool calls
across 66 cycles, 100% read-only, `repo.edit` called zero times; 49 of 66 replies
truncated before their tail (so the cycle-to-cycle handoff was scraped from
noise); `MAX_CALLS=3` all spent reading; 65 plans across 70 turns, none
executing; the `phase` field existed and only ever held `"developing"`.

**One hypothesis was REFUTED and the refutation is worth as much as the finding.**
"repo.edit was never in the menu" was disproved three ways: the session recorded
`authority: write` (a runtime receipt that the engine existed), executing
serve.py's own `openai_schemas(registry, builder_menu(True))` emits `repo.edit`
with full typed parameters, and `hcli build` injects `--write` at cli.py:426.
Acting on it would have "fixed" a gate that was never shut.

## The four structural repairs

1. **`fs.read` told the truth.** An out-of-range window returned `ok=True`,
   `content:""`, `truncated:false`, with `end_line` BEFORE `start_line` — failure
   reported as success. That is why a false belief survived 15 consecutive cycles
   and cycle 47 walked a 247-line file toward line 1720. `truncated` cannot carry
   this (past EOF nothing was cut), so the window now reports itself.
2. **The loop can FAIL a cycle.** A phase machine
   (ORIENT → HYPOTHESIS_REQUIRED → DISCRIMINATOR_REQUIRED → ACT_REQUIRED → VERIFY)
   in the `phase` field that was already there and inert. A cycle that does not
   produce what its phase demands is recorded FAILED and named back verbatim.
   ACT is satisfied ONLY by the server's executed-tool trace showing `repo.edit`,
   never by prose — cycle 66 echoed the literal words "repo.edit" while calling
   nothing, and that string is now a negative control in the self-check.
   Markers are read from the reply HEAD because generation is left-to-right and
   74% of replies died before their tail.
3. **The body can run its own discriminator.** The registry carries 109 tools;
   the chat surface offered 11 and not one executed anything, while `/health`
   advertised `TEST: "run admitted tests"`. `tests.run` existed and was
   unreachable, so "write a RED discriminator" was an instruction with no
   executable step. Exposed as a write-authority door only.
4. **RED is earned, not asserted.** The DISCRIMINATOR phase asks the body to RUN
   its test and answer `RED:` or `REFUTED:`; the harness records whether a RED was
   EXECUTED (a `tests.run` call in the tool trace) or merely CLAIMED. A `REFUTED:`
   closes the arc as a real result and retires the dead hypothesis into a
   carried-forward do-not-repropose list — without that path the loop would have
   demanded an edit for a defect just shown not to exist.

## Two writers, two trees

HCLI now develops in `.worktrees/hcli-selfdev` on branch `hcli-selfdev`; Claude's
Odyssey science stays in the primary checkout. Proved on a live commit: an
Odyssey commit moved main `a00ea4048 -> 4c7cfc71e` while the selfdev worktree
stayed at `a00ea4048`. All 9 "HEAD MOVED (edit accepted)" events in experiment 1
were Claude commits landing mid-cycle; the trailer check fixed the measurement,
this removes the contamination.

`harness.py` splits `ROOT` (the tree HCLI edits) from `CAMPAIGN` (the supervisor
record, canonical in the primary checkout so it survives whatever happens to the
worktree).

## Deploy model (unchanged, and it bites)

The daemon serves from the INSTALLED SNAPSHOT (`~/.local/share/hcli/current`),
never the repo tree. A `chat_tools`/`tool_registry` change needs
`PYTHONPATH=. python3.12 -m hcli install-shims` **and** a full surface restart.
Start the surface FROM the worktree so `RepoContext.detect` resolves there —
verified, and worth verifying again, because a worktree's `.git` is a FILE and a
detector requiring a directory would escape upward and fail silently.

## Where experiment 2 stands

Restarted 11:25:55, snapshot `build-20260909-152444`, resumed from durable state
with no objective reconstruction. Cycles 67 and 68 both advanced on evidence:

- 67: `ORIENT -> DISCRIMINATOR_REQUIRED` — a falsifiable hypothesis naming
  `hcli/engine.py`'s `Engine._validate` / `check_rust_file` and the exact test.
- 68: `DISCRIMINATOR_REQUIRED -> ACT_REQUIRED` — discriminator named.

Two evidence-driven transitions in two cycles, against zero in sixty-six. Note
the hypothesis is FALSE (that test passes 4/4, checked independently) — which is
a healthy falsifiable result, and exactly why the `REFUTED:` path had to exist
before the body reached ACT.

**G004 and G005 remain open and cannot be forced.** Claude's structural repairs
explicitly do not count. If the repaired structure also produces a large
population of non-transitioning cycles, S007 says re-diagnose rather than wait —
the harness logs a loud `!! STUCK` line after 4 consecutive same-phase failures
to force that.

## Odyssey

The ModelLake census is COMPLETE: 53 distinct specimens carry a real
measurement, 0 remain classification-only, 2 resist static measurement entirely
(evo2_40b split `.pt`, mamba3-mimo `.bin`) and are recorded as such.
`receipts/odyssey-i/MODELLAKE_MARCH_SYNTHESIS.json` distils it into nine laws and
six method traps — read that before touching specimen 46. What remains gated is
execution and only execution: no capability battery, Gravity codec search,
NR/NX build or physical run has happened, because all need the GPU the resident
holds.
