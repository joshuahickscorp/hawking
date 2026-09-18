# HAWKING delegation — operator runbook

You do not need a separate client session to use this. Everything below is five shell
commands and a file you can read with `cat`.

HAWKING is a **governed laboratory**. You hand it a bounded objective; it hands back
**evidence**, not conversation. The whole point of the surface is the line it
draws between *a command that actually ran* and *something a model said*.

---

## The five verbs

```
hawking run    --goal "<objective>" [--verify strict|standard] [--budget '<json>']
                                 [--resources NAME]... [--protect PATH]...
                                 [--constraint TEXT]... [--expect PATH]...
                                 [--root DIR] [--worker-mode reason_only|read_only_research]
                                 [--no-spawn] [--json]
hawking status <mission> [--json]
hawking steer  <mission> "<text>" [--kind knowledge|correction|constraint]
hawking result <mission> [--json]
hawking abort  <mission> [--reason "..."]
```

`<mission>` is either the id `run` printed, or the path to the mission
workspace. Both resolve to the same place; the filesystem is the registry, so
there is no index that can go stale.

> **Before your first run:** the `hawking` on your PATH is a shim pointing at a
> *snapshot* of the package under `~/.local/share/hawking/current`, not at the repo.
> A snapshot taken before the delegation verbs existed will answer
> `unrecognized arguments`. Refresh it once:
>
> ```console
> $ cd ~/Downloads/hawking && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m hawking install-shims
> ```
>
> Or skip the shim entirely and run from the repo with
> `python3 -m hawking <verb> ...`, which is what every example below does.

`run` writes the durable contract, then returns **immediately** with a mission
id after asking local Hawkingd (`:8011`) to own the bounded executor. If
Hawkingd is unavailable, the mission remains queued; HAWKING never falls back to
a detached client-side worker. Every other verb reads durable state from disk
and needs no live process.

### The old CLI still works

`hawking 4 "do the thing"`, `hawking --task ...` and `hawking --task-file ...` are
unchanged. The delegation verbs are dispatched only when they are the *first*
token. If your mission prompt genuinely begins with the bare word `run`,
`status`, `steer`, `result` or `abort`, pass it as `hawking --task "run the ..."`.

---

## Start to finish

```console
$ cd ~/Downloads/hawking
$ export HAWKING_ENDPOINT=http://127.0.0.1:8011/v1/chat/completions   # Hawkingd
$ hawking run --goal "the hawking test suite passes on python 3.12" \
           --verify strict \
           --protect civilization/ --protect receipts/ \
           --expect report.json
9f2c41a0b7e3

$ hawking status 9f2c41a0b7e3
mission 9f2c41a0b7e3  phase=delegated
objective: the hawking test suite passes on python 3.12
writer: pid=48213 alive=True
units: {}
steers pending: 0
envelope: False verdict=None

$ hawking steer 9f2c41a0b7e3 "the interpreter is /Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
steer 0c8f... queued (knowledge)

$ hawking result 9f2c41a0b7e3
```

Everything the mission writes lives under
`.hawking/delegations/<mission-id>/.hawking/mission/`:

| file | what it is |
|---|---|
| `delegation_spec.json` | the contract, written **before** any work started |
| `state.json` | durable mission state (`hawking.mission` owns this) |
| `dag.json` (one level up) | the work DAG, the authority on unit status |
| `pipeline_result.json` | raw verifier verdicts |
| `delegation_envelope.json` | the result envelope |
| `delegate_exec.log` | the worker's stdout/stderr |
| `delegation_cancel.json` | present only if someone ran `abort` |

### Deliberate limit: delegated workers do not run verifier shell commands

Both durable worker modes are restricted. `reason_only` receives supplied
evidence and executes no tools. `read_only_research` may use Hawkingd's small
server-owned read/search/receipt/history allowlist; it cannot write, start a
process, switch models, deploy, promote, or invoke a generic shell.

Consequently, a command proposed by the model is recorded as refused rather
than run, even if an embedding caller supplies a legacy `shell_runner`.
Supervisor-owned verification must occur outside the worker pipeline under a
separately reviewed fixed contract, with its result supplied as evidence. Do
not treat a model-proposed command or its prose as verified evidence.

---

## Reading a result: verified vs hypothesis

This is the part that matters. `hawking result` prints two lists and they never
merge.

```
mission 9f2c41a0b7e3  state=failed  VERDICT=BLOCKED
VERIFIED (1) — each backed by a deterministic artifact
  [VERIFIED] the package imports without a model server
             command: python3 -c "import hawking" (exit=0)
HYPOTHESIS (2) — nothing here is established
  [HYPOTHESIS] the hawking test suite passes on python 3.12
               why not verified: verifier FAILED: verdict=FALSE exit_code=1
  [HYPOTHESIS] I have completely verified this and I am confident there are ...
               why not verified: no deterministic artifact supplied
blocker: ...
next: clear the blocker, then steer or re-run: ...
```

**`[VERIFIED]`** — a claim that got in through one door only: a deterministic
artifact. Either a **command that was actually run**, printed underneath with its
exit code, or a **receipt path that exists on disk right now**. You can re-run
the command yourself. That is the whole guarantee.

**`[HYPOTHESIS]`** — everything else, including every word a model produced. The
prose can be as confident as it likes; there is no wording that promotes it. The
`why not verified` line tells you which it is:

* `no deterministic artifact supplied` — a model said it, nothing checked it.
* `verifier FAILED: ...` — something *did* check it, and it came back false. The
  matching entry under `refutations` carries the command and exit code.
* `verifier UNVERIFIABLE: no command settled it` — nothing could decide it.
* `receipt path does not exist on disk: ...` — the artifact it leaned on is gone.

Rule of thumb: **if you would not act on it without re-running the command
yourself, it is a hypothesis, and the envelope already says so.**

### Verdicts

| verdict | what it means | what to do |
|---|---|---|
| `ACCEPT` | at least one verified fact, no failed verifier, nothing outstanding | consume the verified facts; re-run the printed commands to re-check |
| `BLOCKED` | a required verifier failed, or a blocker was recorded | read `blocker`, clear it, then `steer` or start a fresh `run` |
| `INCONCLUSIVE` | nothing was settled: no verified facts, or the data was malformed | read `defects` and `remaining_uncertainty`; supply a real verifier and re-run, or `--verify strict` |
| `ABORTED` | someone ran `hawking abort` | nothing here is a finding; start a fresh mission if the work is still wanted |

A failed verifier **can never** produce `ACCEPT`, whatever the model reported.

### What BLOCKED does *not* mean

`BLOCKED` is not "everything failed". Independent work that did land still shows
up under `VERIFIED`. Check `remaining_uncertainty` for what is genuinely still
open — being blocked on one resource is not global completion, and the envelope
will not pretend it is.

### Fields that are `null`

A `null` is a real answer: *this was not established*. It is never a placeholder
for a value someone forgot to fill in, and nothing here will invent a plausible
number. `physical_measurements: null` means no physical measurement was taken.
`defects` lists anything that was missing or malformed, by name.

If a recorded number and an actual run disagree, **the run wins** — the envelope
shows both under `tests.candidates` and names the winner in `tests.authority`.
Likewise a newer durable disk state beats a stale receipt.

---

## Steering a live mission

```console
$ hawking steer <mission> "prefer the 1B; the 27B is serving something else"
$ hawking steer <mission> "add obligation: the CLI must stay backward compatible" --kind constraint
```

Steers are queued on disk and consumed by the worker **before its next
obligation**. They apply to future work only — a steer never rewrites work that
already completed and never marks anything VERIFIED. `hawking status` shows
`steers pending`; it drops to 0 once the worker has absorbed them.

Kinds: `knowledge` (context), `correction` (you were wrong about X),
`constraint` (a rule the work must satisfy).

---

## Aborting

```console
$ hawking abort <mission> --reason "machine needed for the fill"
aborted 9f2c41a0b7e3 (lock_free=True)
```

`abort` writes a cancel record, signals the worker (only when its pid is alive
*and* its process start token still matches, so a recycled pid is never killed),
releases the single-writer lock and writes an `ABORTED` envelope. `lock_free=True`
means the workspace is immediately reusable. Nothing in an aborted envelope is a
finding.

---

## When something looks stuck

```console
$ hawking status <mission> --json | python3 -m json.tool
$ tail -50 .hawking/delegations/<mission>/.hawking/mission/delegate_exec.log
```

* `writer.alive: false` with no envelope — the worker died. Its artifacts are
  still on disk and `hawking result` will list them, but they stay **unverified**:
  something existing is not something checked.
* `DelegationBusy` from `run` — a live writer already holds that workspace. That
  is the exclusivity working. Wait, or `abort` it.
* `defects` non-empty — a state or result file is malformed. The envelope refuses
  rather than guessing; the defect names the file.

---

## The model

The worker talks only to the local Hawkingd
`http://127.0.0.1:8011/v1/chat/completions` endpoint in normal operation.
Hawkingd owns the worker mode, resident lifecycle, and executor tree; an
arbitrary OpenAI-compatible server is not equivalent authority.

The current local assistant may remain behind Hawkingd while it is useful, but
that is temporary migration debt rather than native-runtime completion. Do not
start a separate llama.cpp, Ollama, MLX, or Python inference server as a
delegation fallback.

**No server up is not a failure mode that lies.** The worker records the
connection error as the `blocker` and the verdict comes back `BLOCKED`; a
delegation does not create an unmanaged worker instead.

---

## Safety rails

* A delegated worker never executes its model-proposed verifier command. It is
  recorded as refused, regardless of its apparent admissibility or a caller's
  legacy generic runner.
* The host-only `shell_runner` retains its protected-path and command
  admissibility checks for separately reviewed control-plane verification; it
  is not an authority channel for a delegated worker.
* One writer per workspace, enforced by a crash-safe lock that records the
  holder's pid *and* process start token.

---

## Tests

```console
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest hawking/test_delegate.py -q
```

The suite is fully offline — the model is faked at the `ModelCaller` seam, so it
never passes or fails because a server happened to be up. The one live
end-to-end check skips loudly with its reason unless `HAWKING_ENDPOINT` is set.
