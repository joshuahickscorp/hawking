# Running HCLI

The short version, for a human at a terminal. Everything here is checkable on
the machine; nothing in this file is a remembered number.

## Enter

```bash
hcli
```

That drops into the interactive environment. Natural language is the primary
input — you do not need commands for normal work. Type `/` to see what commands
exist and `/help <command>` for what one does and whether it mutates state.

**This file deliberately does not list the commands.** The registry in
`hcli/command_registry.py` is the single source of truth: `/help`, completion,
and resident tool discovery all render from it. A list here would drift, and a
drifted list is worse than no list.

Single-shot instead of interactive:

```bash
hcli "keep working on sub-2 and tell me only if the capability cliff changes"
```

## Which code does `hcli` actually run

Two different install paths exist, and they are not the same code:

| how you installed | what `hcli` runs |
|---|---|
| `python3 -m pip install -e .` | the repo, live — always current |
| `python3 -m hcli install-shims` | a stamped snapshot under `~/.local/share/hcli/` |

The snapshot is frozen at the moment you installed it. It drifted six days
behind the repo once and nothing said so, which meant `hcli` and
`PYTHONPATH=. python3 -m hcli` silently ran two different codebases. Startup now
compares the snapshot against the source it was copied from and says so if they
differ. Refresh it with:

```bash
python3 -m hcli install-shims
```

**One more wrinkle: your working directory decides.** The shim sets
`PYTHONPATH`, but Python puts the current directory first, so `hcli` run *from
inside the repo* imports the repo and `hcli` run from anywhere else imports the
snapshot. That is also why the staleness line only appears outside the repo —
inside it, there is nothing stale to warn about. If you want one answer
regardless of where you stand, use the editable install.

## Where state lives

| root | what it holds | safe to delete |
|---|---|---|
| `<repo>/.hcli/` | missions, resident state, background jobs, locks | working state, rebuilt on demand |
| `<repo>/.hcli/pastes/` | cached pastes | **yes** — disposable by definition |
| `<repo>/receipts/` | receipts and evidence | **no** |
| `<repo>/civilization/` | roadmap and obligation state | **no** |
| `~/.local/share/hcli/` | stamped install snapshots | yes, except the `current` symlink target |

Context has three tiers. The next model request receives a bounded semantic
checkpoint (goal/invariants, verifier and mission state, steering, recent
outcomes, prior receipt claims/blockers/next actions, the goal bank, and staged
versus unstaged paths). `/compact` keeps four hot messages and archives older
message records as
`<workspace>/.hcli/sessions/<session-id>.history.jsonl.gz`. Session shutdowns
and compactions also update the workspace prior-knowledge index at
`<workspace>/.hcli/knowledge.json`; its older bounded records are kept in
`knowledge.jsonl.gz`. The hot index is reused by a new session or overnight
WorkUnit and ranked against the current goal/question before it enters the
prompt, while the gzip file is cold recovery material.
Compression saves SSD space and the archive preserves recovery material; it
does not increase a model's native context window. `/context` reports the
checkpoint and paste cache without reading the cold archive into the prompt.
When an older fact falls out of the hot index, the model can call the bounded
read-only `context.recall(focus=...)` tool to retrieve relevant archive records
without replaying the transcript.
This workspace points its cold archive at the mounted large SSD through
`.hcli/config.json` (`context_archive_root`); another machine can use
`HCLI_CONTEXT_ARCHIVE_ROOT` or omit it and keep the archive beside the index.

Future work can be banked without turning it into steering:

```text
/bank prepare the overnight production report
/bank mission run the multi-day verified production campaign
/bank                         # show queued/running/recent goals
/bank drop 2
/bank clear
```

In the TUI, the literal `\\bank ...` spelling is an alias for `/bank ...`.

The queue is workspace-scoped at `<workspace>/.hcli/goal-bank.json`, survives
new HCLI sessions, and starts FIFO after a successful active goal. A failed or
cancelled goal stops promotion; its remaining bank stays intact. A dead HCLI
owner (matched by PID and process-start identity when the host exposes it)
returns a claimed goal to `queued` on the next launch.

The resident supervisor watches the same file. This means a goal banked from a
one-shot command or another terminal wakes the resident after its current
mission is complete, even when no TUI is open. `auto` goals follow the current
runner in an interactive session; a resident promotes every banked goal as a
persistent Mission. Use `mode=mission` when queueing a multi-day goal whose
Mission/DAG/checkpoints must be the visible unit of work.

An obvious read-only directory question such as “what is in this folder?” uses
the typed `fs.list` capability directly, so a cold model does not have to load
just to discover how to list the current directory. The normal model/tool loop
still handles qualified or mutating requests. `fs.list` reports both files and
visible subdirectories, with a bounded result and a truncation marker.

`/clear` clears the display and conversational scratch. It does **not** forget
the mission. Dropping a cached paste removes the cache entry only; it cannot
reach receipts, mission state, or evidence — the paste cache validates every id
against a strict pattern and a resolved-parent check, so a crafted id cannot
escape its own directory.

## The resident

The resident is a supervisor plus a disposable worker. The mission and DAG are
disk state; a worker PID or a loaded model is not. See
[RESIDENT_DAEMON.md](RESIDENT_DAEMON.md) for the architecture.

The scheduler can run dependency-ready WorkUnits concurrently up to its
resource/runtime limits. Grok is callable through `/grok` and the bounded
`grok.swarm.*` tool pair (at most four audit/consult lanes); Grok output still
needs local verification. The resident can launch explicit shell-free child
jobs, but it does not create an unbounded tree of model clones: one disposable
model worker owns the mission, and parallelism stays inside the durable DAG and
its resource gates.

```bash
hcli agentos resident status         # no model is opened by this
hcli agentos resident start --goal "..."
hcli agentos resident bank --mode mission "run the overnight campaign"
hcli agentos resident stop
```

`hcli agentos status` prints the machine-level view: background jobs, resource
ownership, recovery state.

## Leaving without stopping the work

Closing the CLI is not stopping the mission. A CLI session, an HCLI mission, and
a bounded model call are three different lifetimes: one mission outlives many
sessions, and one session triggers many bounded calls. `/quit` leaves the client.
Stopping work is an explicit action, not a side effect of closing a terminal.

## If it will not run

The resident refuses to start when the host is genuinely under memory pressure.
That refusal is real and should be believed — but check what it is reading:

```bash
hcli agentos resident status
sysctl vm.swapusage      # `used` is a boot HIGH-WATER MARK, not swap in use
vm_stat | grep -i swap   # Swapouts flat between samples = not swapping
```

Admission reads bytes paged out *since the previous probe*, not the swapfile's
size. A host with a 30 GB swapfile and flat swapouts is not under pressure and
will admit. If it refuses anyway, free RAM is below the reserve — look at what
else is running. Concurrent ModelLake downloads are the usual cause; they and
the resident compete for the same RAM and do not currently share a budget.

## Recovery

A new session recovers the active mission from disk. There is no recap step and
no "how can I help you today?" reset while a mission is live. If a supervisor is
left orphaned by a driver that exited, it records why it stopped in its own state
file rather than polling forever.
