# What became buildable in the last few hours

Survey basis: `.hcli/mission/events.jsonl` (event_sink.py, wired at
`hcli/agentos/resident.py:1548`), `hcli/hawkingd.py` + its shim, the
`tools/sovereign/` gate producers, and `receipts/sovereign/G009_reachability.json`.
Ranked by (value to an unattended overnight run) / (effort). Five entries;
the rest didn't clear the bar.

## 1. Process capability for the model -- BUILT this task

**What it is.** `processes.list`, `processes.summary`, `processes.orphaned`
registered in `hcli/tool_registry.py`, each a thin read-only wrapper around
an existing function in `hcli/processes.py` (`live_processes()`,
`summary()`, `orphaned_resident_bodies()`). No new logic -- the classifier,
the `phys_footprint` measurement, and the orphan-detection rule already
existed and already ran in production; only the registry entry was missing.

**What unlocked it.** `receipts/sovereign/G009_reachability.json` found
`hcli.processes` reachable from `hcli/runtime.py:136` (startup reaper) and
`hcli/commands.py:1475` (`/processes` command) but **unreachable from the
model**: `registry_probes` for `process.list`, `processes.live`,
`processes.status` and `hcli.processes` all came back `UNKNOWN_TOOL`, and
`shell.readonly` explicitly refuses `ps`. The live goal names processes as
authority and gave the resident no typed way to look at one.

**Effort.** Small. Three `ToolSpec` registrations plus three handler
closures (~75 lines), no changes to `hcli/processes.py` itself.

**What it would break if wrong.** Two ways this fix could have introduced a
new problem instead of closing the old one, both guarded against in
`hcli/test_tool_registry.py`:
- If the handler had reached `processes.reap_orphaned_bodies()` (SIGTERM)
  instead of `orphaned_resident_bodies()` (enumerate only), a model could
  kill its own resident body or an in-flight `hf download` through what
  looks like a read. `test_processes_orphaned_never_calls_reap` monkeypatches
  `reap_orphaned_bodies` to raise if it is ever called and asserts the tool
  still succeeds.
- If the input schema had accepted a `pid`/`signal` argument, a future
  handler edit could turn a "list" tool into a kill primitive without
  anyone noticing the schema allowed it. All three schemas are
  `additionalProperties: False` with zero properties;
  `test_processes_tools_are_read_only` asserts a kill-shaped payload is
  rejected by the schema, before the handler ever runs.

Reachability was verified the G009 way: `ToolRegistry.invoke(name)`, never
the handler directly. Mutation-checked by commenting out the three
`registry.register(...)` calls, confirming all three reachability tests
failed with `failure_class == "UNKNOWN_TOOL"` (G009's exact finding,
reproduced and then closed), and restoring the file.

## 2. `hawkingd` + `hcli install-shims` -> real process supervision

**What it is.** A `launchd` plist (`~/Library/LaunchAgents/...plist`) that
keeps the daemon running across a crash or a reboot, invoking it by the
stable name `hawkingd` instead of a hardcoded
`python3 -m hcli.agentos.resident --supervise <path>` that breaks the
moment the venv path changes.

**What unlocked it.** `hcli/hawkingd.py` gives the daemon a name independent
of which model it happens to be holding, and `hcli install-shims` already
installs `~/.local/bin/hawkingd` pointing at it with `PYTHONPATH` set.
Before this, nothing external could supervise "the Hawking daemon" as a
concept -- only "this exact python invocation."

**Effort.** Small-medium. The plist itself is a dozen lines; the real work
is choosing `KeepAlive`/`ExitTimeOut` so launchd doesn't fight the
supervisor's *own* internal respawn logic (`hcli/agentos/resident.py`
already restarts a dead worker under restart limits) -- two independent
restarters racing on the same `.hcli/resident/state.json` is worse than one.

**What breaks if wrong.** A `KeepAlive` policy that's too aggressive
double-supervises: launchd relaunches the supervisor while the supervisor's
own worker-restart loop is still recovering, multiplying concurrent
writers on `state.json` and risking two resident bodies loading at once
(the exact 11GB-orphan failure mode G170/G173 just fixed, reintroduced from
a different direction).

## 3. `events.jsonl` -> a real-time stall/anomaly watchdog

**What it is.** A small standalone reader that tails
`.hcli/mission/events.jsonl` by byte offset (not by polling) and alerts on
patterns a human isn't watching for overnight: no event for N minutes while
`phase: thinking` is stuck, three consecutive tool-call failures, a
`heartbeat` with `elapsed_s` that stops advancing.

**What unlocked it.** `hcli/agentos/event_sink.py`, wired at
`resident.py:1548` (`subscribe(on_any_event)`), makes the worker write
*every* bus event to disk -- today it is 1,851 lines and growing in this
live session. Before this landed, the only signals were the coarse
heartbeat field in `state.json` and prose-grepping `mission.log`; nothing
gave byte-precise, typed access to tool calls, phase transitions, and model
text as they happen.

**Effort.** Small for a naive tailer; the real cost is respecting
`EventSink`'s own rotation (`events.jsonl` -> `events.jsonl.1` past
`max_bytes=8MB`, per `event_sink.py`) -- a reader that doesn't handle
rotation silently stops advancing at the boundary.

**What breaks if wrong.** A watchdog that mishandles rotation reads a
frozen offset forever and either alarms on a false stall (daemon is fine,
the reader just fell off the file) or, worse, goes silent and reports
nothing while a real stall runs unattended all night -- exactly the failure
mode it exists to catch.

## 4. Compact-catalog surfacing as its own regression, not just registration

**What it is.** A test that asserts a read-only tool relevant to the live
goal's own wording actually appears in `_compact_tool_catalog`'s output for
that goal text, not just in `ToolRegistry.discover()`.

**What unlocked it.** G009-F2: `Engine._tool_catalog` (the full catalog) has
zero production call sites -- both callers of `_prompt_with_observations`
(`hcli/engine.py:1296`, `:1323`) pass `compact_catalog=True`. The compact
path calls `registry.describe(focus=...)`, which ranks tools by a keyword
score against the prompt text and returns only the top `max_results`
(default 12). That means *registered* and *shown to the model* are now two
different claims -- the same shape of gap G009 found between "registered"
and "has a call site," one layer up. `processes.list`'s name literally
contains the word the live goal uses ("processes"), so it should rank near
the top of `describe("...processes...")`, but nothing currently asserts
that for any tool.

**Effort.** Small -- call `registry.describe(focus=<live goal text>)`
(or `_compact_tool_catalog` directly) and assert the tool name appears,
the same shape as `test_forbidden_fruit.py`'s registry-reachability test
but one layer further down the real prompt-construction path.

**What breaks if wrong.** A tool can be fully registered, pass every
reachability test in isolation, and still never appear in a live prompt
because `describe()`'s scoring ranked it outside the top 12 for the
mission's actual wording -- reachable on paper, invisible in the one path
that matters, which is precisely what this task's own audit was looking
for.

## 5. A sweep over `tools/sovereign/*.py` gate producers

**What it is.** One command that re-runs every gate producer
(`g001_verifier_synthesis.py`, `g002_overhead.py`, `g009_reachability.py`,
`g010_modellake_retained.py`, `g011_streaming.py`, `g014_negative_science.py`)
and diffs the freshly generated receipt against the checked-in one under
`receipts/sovereign/`, flagging drift instead of trusting a receipt that
may have gone stale.

**What unlocked it.** These producers already regenerate their own receipts
on demand (confirmed for G009: its docstring states it derives its required
set from the *live* goal text and refuses to write if a derivation quote is
no longer present -- an anti-invention guard, not a cached report). That
means a drift check is now a pure read: run the producer, diff the output,
report.

**Effort.** Small, if scoped to producers confirmed read-only (G009 says so
explicitly in its own docstring; the other five need the same one-line
check before being swept blindly).

**What breaks if wrong.** Low blast radius by design -- worst case is a
noisy diff. The actual risk is scope creep: sweeping a producer that turns
out to mutate state (never verified here) would turn an audit script into
an unattended writer, which is exactly the kind of accidental capability
this whole campaign has been finding and closing.
