# labd — the conductor pattern as a tool (Tier 3b)

_Extracted 2026-06-10 from the proven artifact set: `ops/conductor.sh` (the orchestrator that
ran the quant marathon), `ops/watchdog-runbook.md` (the cron backstop's standing orders),
`ops/pod-chain-v2.sh` and `ops/pod-watch.sh` (the remote chain and the superseded poller).
The incident record is will.md's update log (2026-06-10 entries 12:20 / 18:00 / 20:50).
`ops/labd.sh` is the prototype engine. **The hand-written conductor stays in production —
labd replaces nothing until it has run a marathon of its own.**_

## 1. The pattern, named

One supervisor process, three properties:

1. **Stateless-resumable tick loop.** Every decision is re-derived from the filesystem each
   tick. The process can be killed at any instant and relaunched with one command; nothing is
   lost because nothing lived in memory.
2. **The mechanical/judgment split.** Actions whose policy was decided in advance (recipes,
   pre-authorized in calm context) run inline. Anything carrying *new information* prints an
   `EVENT:` line and exits the process — the tracked launcher wakes an LLM session to
   interpret. The loop never improvises.
3. **A backstop above, sentinels below.** An hourly cron-driven LLM (with a runbook as its
   standing orders) re-arms the loop and audits what the loop cannot see — its own death.
   Sentinel files give the owner a one-touch control plane; babysitters handle one-off
   boundary-timed actions the 60 s tick is too coarse for.

The tier ladder this project actually climbed:

| tier | artifact | shape |
|---|---|---|
| 1 | ad-hoc `until`-loop watchers | one concern each; die with the app; re-armed by hand |
| 2 | the watcher zoo (guardian / governor / pod-watch) | named scripts, still one concern each; double-wake noise |
| 3 | `conductor.sh` (2026-06-10) | ONE loop, ordered rule blocks, mechanical/judgment split, judgment exits |
| 3a | `watchdog-runbook.md` + hourly cron | the LLM backstop that re-arms tier 3 and audits the invisible |
| **3b** | **labd (this)** | the same machine with rules as *config* instead of code |
| 4 | compiled engine (Rust) | not built — triggers in §13 |

## 2. The model — five parts, three clocks

| part | clock | intelligence | cost per activation |
|---|---|---|---|
| labd tick loop | 60 s | zero — policy pre-decided | ~free |
| judgment session (LLM) | on event | full | $$ — paid only when there is information |
| cron backstop (LLM + runbook) | 1 h | full, but narrow standing orders | $ bounded |
| babysitter | 5–15 s, for ONE boundary | zero | free; dies after acting |
| sentinel files | — | the owner's one bit | free |

The division of labor: the loop has reflexes, the session has judgment, the cron has
suspicion, the babysitter has timing, the owner has a veto.

## 3. Tick model

Each tick, in order:

1. **Re-read the config** (hot-edit: changes land next tick — no restart, no SIGHUP protocol).
2. **Pass 1 — mechanical rules** (`type = action`), in `order`. All of them. Conditions are
   shell, evaluated fresh against the filesystem; actions run inline and are logged with rc.
3. **Pass 2 — judgment rules** (`type = event`), in `order`. The FIRST one whose condition is
   true prints `EVENT: <name> :: <detail>` to stdout and exits 0.
4. Discard uncommitted detector stages, heartbeat-log every 10th tick, sleep.

**What persists, and where.** Nothing in memory. The spool directory holds exactly four kinds
of state, all plain files, all resettable with `rm` (one-touch, same philosophy as sentinels):

| spool file | meaning | reset |
|---|---|---|
| `cur.<rule>.<key>` | committed detector cursor (`changed`/`grew`) | rm → re-seeds silently |
| `done.<rule>` | oneshot latch | rm → rule is live again |
| `count.<rule>` | retry budget spent | rm → budget restored |
| `lock/pid` | single-instance lock | stale locks self-clear via `kill -0` (pid-based — no pgrep, no self-match) |

**Cursors vs the conductor's in-memory baselines — a deliberate upgrade.** The conductor
snapshots `B_JSON`/`B_RC`/`B_POD` at startup, so anything that changed *while it was down*
is silently absorbed into the new baseline (the cron-LLM was the only thing that could catch
it). labd's cursors live on disk and are committed only when a rule fires: an event that
happened while labd was down fires on the next launch. First-EVER sight of a watched path
seeds the cursor silently (otherwise the first launch would be an event storm over
pre-existing state). Detector writes are staged (`*.stage`) during condition evaluation and
committed only on fire, so a compound condition that evaluates `changed` but fails its other
clause does not advance the cursor.

**Exit is the only event mechanism.** `--once` runs a single tick (the test harness);
`--check` parses and lists rules (the config linter — run it after every edit).

## 4. The mechanical/judgment split — why it matters for LLM-in-the-loop ops

The split is the load-bearing idea; everything else is plumbing.

**Token cost.** An LLM polling at the conductor's 60 s tick is 1,440 wakes/day; each wake
re-derives state (will.md §8, logs, `ls`) at ~10–50k tokens → order 10⁷–10⁸ tokens/day of
"nothing happened". The marathon's observed judgment-event rate was ~5–15/day (count the
update-log entries). The split buys roughly two orders of magnitude — but the bigger win is
that the session's context window holds only *decisions*, never heartbeat noise.

**Judgment quality.** A pre-authorized recipe is judgment exercised ONCE, at write-time, with
full calm context — the runbook was reviewed, the kill threshold (35 min) was derived from
measured requant/eval durations (§5 of the runbook: requant ≈ 950–1000 s, eval ≈ 66–95 s).
The alternative — waking an LLM at 03:00 to re-decide "should I pkill this stalled arm?" —
re-spends judgment on a solved problem with cold context and nonzero improvisation risk. The
runbook line "Do not 'fix' performance by changing configs" exists because woken LLMs
improvise. Corollary: **the mechanical list IS the safety policy** — anything not on it is
structurally forced up to a brain.

**Exit-don't-callback.** Judgment events terminate the loop rather than notifying and
continuing, because:
- one event at a time: the session inspects a system that is no longer mutating itself;
- the relaunch is the acknowledgment — a running labd *proves* the last event was seen;
- the failure mode degrades gracefully: if the wake is missed, the loop is down, the hourly
  cron notices a dead loop + an unhandled `EVENT(...)` line at the log tail, and escalates.
  A missed event becomes a 1-hour-latency page, never a silent loss.

**The event is a doorbell, not a payload.** The `detail` line is a courtesy preview (capped
500 chars). The waking session re-derives everything from the filesystem — which it must be
able to do anyway (stateless-resumable applies to the brain too). This is why string events
are sufficient and JSON payloads are a rewrite trigger (§13), not a missing feature.

## 5. Event taxonomy

| class | conductor instance | meaning | session's first move |
|---|---|---|---|
| VERDICT | `qat-*.json` changed | a result landed | read it, update will.md, pick the next arm |
| FAILURE | `rc=` count grew in night3.log | a lane died abnormally | tail the log; classify OOM / EPERM / science |
| MILESTONE | pod ppl json count rose | remote progress banked | read the mirrored json |
| EXITED | `ladder=DOWN` | a watched remote process is gone | finished-vs-died triage |
| STALL_KILLED | the runbook-§2 kill fired | a pre-authorized kill ran | confirm crash-resume picks it up |
| EXHAUSTED | re-pass count hit max | a mechanical budget is spent and the condition persists | the recipe is wrong or the world changed — investigate; never just raise max |
| UNREACHABLE | 3 consecutive ssh failures | the probe path is dead | pod-dead vs IP-changed vs local-network |

Note what is NOT an event: an idle window. Idleness triggers *mechanical filler* (the speed
sweep, the low-LR ternary arm) — pre-authorized work queued for free compute. Its completion
then surfaces as a VERDICT/named event.

## 6. Sentinel files — the one-touch control plane

| kind | examples | semantics |
|---|---|---|
| human gate | `/workspace/SKIP-70B`, `SKIP-MP` | a future expensive phase pre-cancelled by one `touch`; works over any transport that can create a file (web terminal, scp, Finder) |
| once-latch | `.pod-ladder-down-noted`, `.ternary-lowlr-launched` | edge-detection made durable: fire exactly once across restarts (labd: `oneshot = true` → `spool/done.*`) |
| completion marker | `.night3-done`, `.quant-done` | idempotency keys — re-runs become no-ops; also the resume protocol |
| counter | `.repass-count` | retry budgets that survive restarts (labd: `spool/count.*`); reset = rm |

The inverse lesson (12:20 incident): **a marker is trusted state — write it only when the
thing is actually true, by the process that did the work, last.** The false `.night3-done`
(written because `scratch/` root stayed writable while the qwen-05b subtree was
sandbox-denied) cascaded into two downstream watchers reading it and exiting.

## 7. Babysitters — boundary-timed one-shots

Proven instance (will.md 2026-06-10 20:50): ladder threads 24→27 had to land *between* two
ladder configs — the running quantize-model had already read `--threads`, so the bump was
legal only in the seconds-wide gap at the config boundary (cgroup quota 27.2 made the value
itself pre-decided). A 60 s tick is the wrong shape for that: too coarse, and the rule would
fire once and be dead config forever.

A babysitter is: an `until`-loop at 5–15 s tick, watching ONE boundary signature, performing
ONE pre-staged action, then exiting with a log line. Spawned by the judgment session when it
schedules the boundary work; never managed by labd. The litmus test for "labd rule or
babysitter?" — **will this ever fire twice?** Yes → rule. No → babysitter.

## 8. The cron backstop

The hourly cron invokes an LLM session whose prompt is the runbook. Its duties, in priority
order: (1) re-arm a dead loop (runbook §1b: pgrep for it, relaunch as a tracked background
task — and read the log tail first for an unhandled EVENT); (2) audit what the loop cannot
see: its own death, the harness app dying (watchers die with the app; the nohup'd runs do
not — will.md §5.9), resource health with pre-authorized cleanups (§4), metrics sampling and
degradation thresholds (§5); (3) enforce the hard-limits list (§6) — the permission boundary
in prose. The 12:20 chat-deletion incident is the existence proof: the entire watcher layer
died with a deleted chat, and the hourly cron re-armed the stack in the successor session
within the hour. labd's contribution to the backstop is its heartbeat line (every 10th tick),
which gives the cron a dead-or-alive read without process inspection.

## 9. Failure modes already paid for

These three were bought with real incidents. labd hard-codes the first, documents the second
as a launch discipline, and provides idioms for the third. Do not re-derive them from scratch.

### 9a. Event order: mechanical before judgment

**Incident (will.md 2026-06-10 18:00: "event ORDER = mechanical before judgment — the first
lane-end proved why").** When a lane ends, the DONE marker and the new result json appear in
the same tick. If the VERDICT event is checked first, the loop exits to wake the session and
the re-pass relaunch never runs — the box idles for the entire wake-interpret-act latency
(unbounded if the wake is missed). Mechanical actions are precisely the ones safe to run
without a brain; starving them behind a judgment exit wastes the machine for nothing.

**labd:** two passes, hard-coded; pass 1 always completes. Stricter than the conductor in one
spot: the conductor's `REPASS_EXHAUSTED` exits from inside mechanical block 1, skipping the
stall check and pod poll that tick; labd defers exhaust escalations to the end of pass 1.

### 9b. Tracked vs orphaned launches (the session-lifetime trap)

**Incident (will.md 2026-06-10 ~12:20).** Processes spawned from an agent chat inherited
macOS sandbox grants that died when the chat was deleted: a requant 164/168 tensors deep
panicked EPERM on its output write; three sibling arms died within one second (could not even
open their logs); a false `.night3-done` propagated (root writable, subtree denied); two
watchers read the marker and exited. Reads and already-open fds kept working — the run limps
until its first fresh file-create, which makes the failure look like anything but what it is.

**The matrix, now policy:**

| what | launch as | why |
|---|---|---|
| science runs / anything long | ORPHANED: `nohup … & disown`, from a durable context (Terminal, or an unsandboxed exec from a living session) | must survive app restart AND chat deletion; grants must not be session-scoped |
| watchers / labd itself | TRACKED harness background task | the exit IS the wake mechanism; dying with the app is acceptable because the cron re-arms |
| never | a run as a watcher's child | inherits the watcher's lifetime and grants |

Tracked watchers should also heartbeat-exit on a deadline (pod-watch used 6 h) so a stale
watcher cannot zombie — the re-arm is the liveness proof. After any chat deletion or app
restart: assume EPERM-broken trees and relaunch the runs per the runbook.

### 9c. Self-match traps in pgrep/pkill

The process table is an API with platform-dependent semantics. Three sub-traps, all hit:

1. **Ancestor self-match — platform-asymmetric.** BSD/macOS pgrep excludes itself AND all its
   ancestors by default (man pgrep: "the current pgrep or pkill process and all of its
   ancestors are excluded"; verified on this box 2026-06-10). procps (Linux, the pod) excludes
   ONLY the pgrep process itself. Consequence: `ssh pod 'echo … $(pgrep -f strand-ladder.sh …)'`
   embeds the pattern in the argv of the sshd-spawned `bash -c` — pgrep's *ancestor* — so on
   the pod the probe can read `ladder=up` forever. The same line is safe on the Mac. The fix
   is one character class: `pgrep -f 'strand-ladder[.]sh'` (the pattern no longer matches its
   own literal occurrence) — pod-chain v2 line 39 carries exactly this fix for its
   ladder-drain wait, while the conductor's pod poll (line 109) and pod-watch.sh still embed
   the un-bracketed form; consistent with POD_LADDER_EXITED never having been observed to
   fire. Treat the bracket trick as mandatory idiom, both platforms, no exceptions.
2. **Sibling match.** A concurrently running process with the pattern in its argv matches on
   BOTH platforms (verified: a backgrounded `bash -c '… # tag'` is found by `pgrep -f tag`).
   The conductor's own ssh poll holds `strand-ladder.sh` and `pod-chain` in its local argv for
   up to 15 s per poll — any local probe for those names during that window false-positives.
   Mitigation: anchor patterns to paths that only the real target has, and bracket-trick.
3. **Under-match — the killer misses.** will.md §7: on the pod, `pkill -f` misses the eval
   because the python's argv lacks the script name. The argv you launched is not the argv
   you'll find — in both directions (verified 2026-06-10: single-command `bash -c
   'sleep 2 # tag'` exec-optimizes the wrapper away, so the tag never enters the process
   table; only the multi-command form `bash -c 'sleep 2; true # tag'` stays resident and
   matches). When the name is unreliable, kill by resource handle instead (GPU occupancy:
   process-level GPU monitors). Same family: the runbook's `pkill -f strand-qat.py`
   issued through a `bash -c` wrapper is ancestor-safe on macOS and would kill its own wrapper
   first on Linux.

labd's own discipline: the engine never pgreps for itself — the single-instance lock is a pid
file checked with `kill -0` (no pattern matching anywhere in the engine's own liveness logic).

## 10. Config format

Line-oriented, parsed by ~30 lines of bash. Full-line `#` comments only (no inline comments —
`run =` values legitimately contain `#`-able text). Globals first, then `[rule]` sections of
`key = value`. Values may contain `=`; keys/values are whitespace-trimmed; unknown keys are
ignored. One line per value — anything longer belongs in a recipe script.

```
tick  = 60                  # seconds between ticks
spool = scratch/labd        # cursors, latches, counters, lock
log   = scratch/labd.log    # actions, rcs, events, heartbeat
cwd   = /abs/path           # cd here at startup (relaunch-from-anywhere safety)

[rule-name]
type  = action | event      # default action
order = 10                  # evaluation order within each pass (default 50)
every = 10                  # evaluate only every Nth tick (default 1)
if    = <shell expression>  # condition; helpers below available; $RULE is set
run   = <shell>             # type=action: executed inline, logged, rc recorded
event = NAME                # type=event: printed as "EVENT: NAME :: <detail>", then exit 0
detail = <shell>            # optional payload preview (stdout, 500-char cap)
oneshot = true              # latch after first fire (spool/done.<rule>)
max   = 2                   # action budget; spent + condition still true -> exhaust_event
exhaust_event = NAME        # the judgment escalation for a spent budget
```

Helper predicates available inside `if`/`detail` (evaluated in a subshell, `$RULE` set):

| helper | semantics |
|---|---|
| `alive '<pattern>'` | `pgrep -f` — bracket-trick the pattern, always (§9c) |
| `quiet '<glob>' <secs>` | newest matching file's mtime is older than secs (also true when no file exists) |
| `changed <key> <glob...>` | `ls -l \| cksum` snapshot differs from the committed cursor; stages the new one |
| `grew <key> <file> <pattern>` | match count increased; reads `grep -c` stdout only (immune to the `0\n0` trap, §12) |

Engine guarantees: pass 1 completes before pass 2 can exit (including deferred exhaust
escalations); detector cursors commit only on fire; first sight seeds silently; config
hot-reloads every tick; one instance per spool (pid lock); heartbeat every 10th tick.
An exhaust escalation REPEATS on every relaunch until the budget is reset
(`rm spool/count.<rule>`) or the condition clears — the conductor's `REPASS_EXHAUSTED`
semantics exactly — and while it persists it preempts every pass-2 rule; that is the point
(an unsolved escalation should monopolize the doorbell), but order graver events lower anyway.

All guarantees verified by tick-level smoke 2026-06-10 (this box): `--check` on §11's config
(12 rules, mechanical 10–41 then judgment 50–90); cursor seed-silent / fire / commit-on-fire;
missed-while-down event fires on relaunch; same-tick mechanical-then-judgment (the action's
artifact exists despite the event exit); exhaust deferred behind an order-99 action; `grew`
fires from a zero-match seeded file (the `0\n0` trap dead); oneshot latches across restarts;
stale-lock takeover via `kill -0`; live-lock refusal rc=1; exit-on-event releases the lock.

Design stance: **the engine stays dumb.** Anything with a protocol (ssh probing, mirroring,
fail-streak accounting) is a recipe script that the config points at. Rules express WHEN;
recipes express HOW. If a condition needs more than one line, it is a recipe that writes a
status file, plus a rule that tests the status file.

## 11. Today's conductor, re-expressed (REFERENCE ONLY — the conductor stays in production)

Verified 2026-06-10: `bash ops/labd.sh <this-config> --check` → `OK: 12 rules, tick=60s`
(mechanical 10–41, then judgment 50–90 — the two passes visible in the listing).

```
# conductor.labd.conf — ops/conductor.sh (2026-06-10) as labd rules. REFERENCE ONLY.

tick = 60
spool = scratch/labd
log = scratch/labd.log
cwd = /Users/scammermike/Downloads/strand

# ---------- mechanical: every tick, in order, always before any judgment exit ----------

# conductor block 1 — night3 wrote DONE but the headline arm json never appeared
[repass]
order = 10
if = test -f scratch/.night3-done && test ! -f scratch/qwen-05b/qat-pv.json && ! alive 'strand-act2-night3[.]sh'
run = rm -f scratch/.night3-done; nohup caffeinate -dimsu ./scripts/strand-act2-night3.sh >> scratch/qwen-05b/night3.log 2>&1 & sleep 3; alive 'scratch/guardian[.]sh' || nohup ./scratch/guardian.sh >/dev/null 2>&1 &
max = 2
exhaust_event = REPASS_EXHAUSTED

# conductor block 5 — runbook §2 stall: qat alive, no requant, newest arm log silent >35 min
[stall-kill]
order = 20
if = alive 'strand-qat[.]py' && ! alive 'quantize-model' && quiet 'scratch/qwen-05b/qat-*.log' 2100
run = pkill -f 'strand-qat[.]py'

# conductor block 6 — pod poll + local mirror, every 10th tick. The recipe owns the
# ssh/scp protocol, writes scratch/pod-results/status.line + mirrored jsons/logs, and
# maintains scratch/labd/pod.fails (consecutive-failure streak, reset on success).
[pod-poll]
order = 30
every = 10
run = ops/recipes/pod-poll.sh

# conductor block 4 — marathon quiet => definitive speed sweep, exactly once
[idle-sweep]
order = 40
oneshot = true
if = test -f scratch/qwen-05b/qat-pv.json && test -f scratch/qwen-05b/qat-pv3bit.json && ! alive 'strand-qat[.]py|quantize-model' && quiet 'scratch/qwen-05b/qat-*.log' 180
run = cd ../strand-speed && mkdir -p research && { echo "== definitive idle sweep $(date '+%F %H:%M') =="; ./target/release/gate-interleave; echo; ./target/release/gate-decode-speed; } > research/speed-G0b-idle-bench.txt 2>&1

# conductor block 4b — owner-approved idle filler at the corrected restart LR (3e-5)
[ternary-lowlr]
order = 41
oneshot = true
if = test -f ../strand-speed/research/speed-G0b-idle-bench.txt && test -d scratch/qwen-05b/qat-ternary-kd-2k-hf && test ! -f scratch/qwen-05b/qat-ternary-lowlr.json && ! alive 'strand-qat[.]py|quantize-model'
run = PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85 nohup caffeinate -dimsu /usr/local/bin/python3 scripts/strand-qat.py --ctx 512 --eval-chunks 64 --eval-ctx 2048 --grad-accum 4 --batch 1 --device mps --grad-checkpoint --lr 3e-5 --kd --model scratch/qwen-05b/qat-ternary-kd-2k-hf --quant ternary --steps 500 --train-chunks 2000 --eval-every 250 --save scratch/qwen-05b/qat-ternary-lowlr.pt --save-hf scratch/qwen-05b/qat-ternary-lowlr-hf --out scratch/qwen-05b/qat-ternary-lowlr.json >> scratch/qwen-05b/qat-ternary-lowlr.log 2>&1 &

# ---------- judgment: first true rule prints EVENT and exits; the session interprets ----------

# conductor block 2 — any new/changed arm json
[verdict]
type = event
order = 50
if = changed jsons scratch/qwen-05b/qat-*.json
event = VERDICT
detail = n=$(ls -t scratch/qwen-05b/qat-*.json | head -1); echo "$n :: $(tr -d ' \n' < "$n" | head -c 400)"

# conductor block 3 — new rc= lines in the chain log
[failure]
type = event
order = 60
if = grew rc scratch/qwen-05b/night3.log rc=
event = FAILURE
detail = tail -3 scratch/qwen-05b/night3.log | tr '\n' '|'

# the stall KILL is mechanical (above); the NEWS of it is judgment — read back from labd's log
[stall-news]
type = event
order = 65
if = grew kills scratch/labd.log 'action stall-kill: firing'
event = STALL_KILLED
detail = ls -t scratch/qwen-05b/qat-*.log | head -1

[sweep-done]
type = event
order = 70
oneshot = true
if = test -f ../strand-speed/research/speed-G0b-idle-bench.txt
event = SPEED_SWEEP_DONE
detail = grep -E 'verdict' ../strand-speed/research/speed-G0b-idle-bench.txt | tr '\n' '|'

# pod milestones ride on the MIRROR: the recipe makes remote state local, cursors do the rest
[pod-milestone]
type = event
order = 80
if = changed ppl scratch/pod-results/ppl_*.json
event = POD_MILESTONE
detail = ls -t scratch/pod-results/ppl_*.json | head -3 | tr '\n' ' '

[pod-ladder-exited]
type = event
order = 85
oneshot = true
if = grep -q 'ladder=DOWN' scratch/pod-results/status.line 2>/dev/null
event = POD_LADDER_EXITED
detail = cat scratch/pod-results/status.line

[pod-unreachable]
type = event
order = 90
oneshot = true
if = test "$(cat scratch/labd/pod.fails 2>/dev/null || echo 0)" -ge 3
event = POD_UNREACHABLE
detail = echo 'pod ssh dead 3 consecutive polls — ask owner: echo "TCP=$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22"'
```

The recipe the config points at (sketch — note the BRACKETED remote probe, fixing §9c.1):

```bash
#!/usr/bin/env bash
# ops/recipes/pod-poll.sh — owns the pod protocol: status line, mirror, fail streak.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519 -p 40078 root@213.192.2.110"
F=scratch/labd/pod.fails
S=$($SSH 'echo "jsons=$(ls /workspace/strand-results/ 2>/dev/null | grep -c ppl_.*json) ladder=$(pgrep -f strand-ladder[.]sh >/dev/null && echo up || echo DOWN) chain=$(pgrep -f /workspace/pod-chain >/dev/null && echo up || echo idle)"' 2>/dev/null)
if [ -z "$S" ]; then n=$(cat "$F" 2>/dev/null); echo $(( ${n:-0} + 1 )) > "$F"; exit 1; fi
rm -f "$F"; mkdir -p scratch/pod-results
printf '%s\n' "$S" > scratch/pod-results/status.line
scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078 "root@213.192.2.110:/workspace/strand-results/*.json" scratch/pod-results/ 2>/dev/null
scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078 "root@213.192.2.110:/workspace/strand-{chain,gate,ladder}.log" scratch/pod-results/ 2>/dev/null
exit 0
```

Translation deltas vs the running conductor (each one deliberate):

| conductor | labd | delta |
|---|---|---|
| in-memory baselines (`B_JSON`/`B_RC`/`B_POD` at startup) | spool cursors | missed-while-down events fire at next launch; first-ever sight seeds silently |
| `sleep 180` re-check inside the idle branch | `quiet … 180` in the condition | debounce by mtime; the tick is never blocked |
| stall kill + `evt` in one block | `[stall-kill]` action + `[stall-news]` event via `grew` on labd's log | the kill happens even if a verdict exits the same tick; the news still fires that tick (pass 2) |
| ad-hoc latches (`.ternary-lowlr-launched`, `.pod-ladder-down-noted`) | `oneshot = true` | same semantics, no bespoke files |
| `.repass-count` | `spool/count.repass` + `max`/`exhaust_event` | unchanged; exhaust now defers to end of pass 1 (§9a) |
| `podfails` in-memory | recipe-owned `scratch/labd/pod.fails` | the streak survives relaunch |
| `ls -l … \| md5` snapshot | `cksum` | portable to the pod (md5(1) is macOS-only) |
| `grep -c … \|\| echo 0` | `grew()` stdout-only read | kills the `0\n0` silent-false (§12) |
| first pod poll at tick 10 (~10 min in) | `every`-N rules fire at tick 0 too | relaunch-as-ack ⇒ the waking session gets a fresh poll immediately |

## 12. What bash cannot express cleanly (measured, not modeled)

Two specimens verified on this box 2026-06-10, then the structural list.

**Specimen 1 — the untyped-string failure is silent and permanent.**
`s=$(grep -c 'rc=' file || echo 0)` on an *existing* file with *zero* matches yields
`s = "0\n0"` — grep prints `0` AND exits 1, so the `||` fires too. Every subsequent
`[ "$RC" -gt "$B_RC" ]` errors (rc=2, "integer expression expected") and reads as false —
forever, silently. The running conductor carries this exact pattern (lines 26/55): launched
against a clean night3.log, its FAILURE event can never fire. The engine's `grew()` takes
grep's stdout only and defaults empty→0, but the deeper point stands: in bash every
comparison is a parse of an untyped string, and the parse failure mode is "condition false",
indistinguishable from "nothing happened".

**Specimen 2 — the process-table API is platform-dependent** (§9c). BSD pgrep excludes
self+ancestors; procps excludes only itself; siblings match on both. The same probe line is
correct on the Mac and a permanent false-positive inside an ssh remote string on the pod.
`stat -f` vs `stat -c` and `md5` vs `md5sum` are the same disease in milder form.

Structural limits (no specimen needed — they are visible in the engine's shape):

- **No per-rule timeout.** A hung action blocks every later rule and the tick itself. macOS
  base has no `timeout(1)`; recipes must self-limit (`ConnectTimeout=15` is why the conductor
  survives network loss). Single-threaded ticking is fine at tick=60 with sub-second rules
  and one 15 s-bounded probe; it is the first thing to break if rules multiply.
- **`eval` is the parser.** Config injection IS code execution, by design. Acceptable for a
  single-operator lab where the config author, the recipe author, and the victim are the same
  person; disqualifying for anything multi-tenant.
- **No cross-tick state machines.** "N consecutive failures" already needs a recipe-owned
  counter file; "X then Y within 5 min" would need timestamped cursors and is where config
  stops being declarative and starts being a language. Resist; write a recipe.
- **Quoting landmines.** Helpers word-split their glob arguments (paths with spaces break);
  ssh remote strings need nested-quote surgery (the pod-poll recipe above was simplified to
  stay single-quoted); zsh-vs-bash word splitting already burned one smoke test (will.md §7).
- **No structured payloads.** Events are strings. By design (§4 — doorbell, not payload), but
  the moment a *tool* rather than an LLM consumes events, strings stop being acceptable.

## 13. When a rewrite earns itself

Triggers — any ONE is sufficient:

1. A second watched host (parallel probes with independent timeouts → async or threads).
2. Rules > ~15, or any condition wanting cross-tick temporal logic (state machines).
3. A tool (not an LLM) consuming events → structured JSON payloads with schemas.
4. The engine itself needs unit tests to be trusted (i.e., someone other than its author
   operates it).
5. The first night lost to an engine bug rather than a recipe bug.

What Rust buys: typed config (serde TOML — the labd.conf grammar maps 1:1), per-rule
async+timeout (tokio), transactional cursors, a testable rule engine, one static binary that
is identical on the Mac and the pod — the same determinism argument STRAND itself makes.
Roughly 500 lines and a day of work, *when a trigger fires*.

What Rust does NOT buy: the recipes stay shell (the actions ARE shell; inlining them into the
engine would be the mistake — the engine/recipe boundary is the whole design); the LLM still
reads and writes config and log as text; the judgment loop (exit-on-event, relaunch-as-ack,
cron backstop) is engine-independent and survives any rewrite.

Verdict: stay in bash. The engine is ~150 lines an LLM can hold in one Read — at this scale
legibility is the feature, and the cost of the rewrite is low precisely because the config
format and the pattern are now specified independently of the implementation. Revisit at the
first trigger, not before.
