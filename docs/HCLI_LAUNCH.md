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

`/clear` clears the display and conversational scratch. It does **not** forget
the mission. Dropping a cached paste removes the cache entry only; it cannot
reach receipts, mission state, or evidence — the paste cache validates every id
against a strict pattern and a resolved-parent check, so a crafted id cannot
escape its own directory.

## The resident

The resident is a supervisor plus a disposable worker. The mission and DAG are
disk state; a worker PID or a loaded model is not. See
[RESIDENT_DAEMON.md](RESIDENT_DAEMON.md) for the architecture.

```bash
hcli agentos resident status         # no model is opened by this
hcli agentos resident start --goal "..."
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
