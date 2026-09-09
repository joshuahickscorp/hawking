# HCLI: the Hawking operator

`hcli` is the cognitive shell of the Hawking computer. One organism, different
authority. You state an objective; it does the engineering.

## Commands (no flags, any directory)

```bash
hcli web                 # browser chat on the current resident, read/research
hcli build               # the same chat WITH repo-scoped write authority
hcli use                 # list 54 bodies; hcli use Qwen3-14B to switch live
hcli serve               # the OpenAI endpoint only (Open WebUI points here)
hcli report              # measure the loaded body: cold prefill, warm reuse, decode
hcli stop                # put down what web/serve/build started
hcli --help              # every verb, grouped
```

## What the body can do

`/health` reports the capability surface -- named as capability, not package,
and each probed against live machinery:

| domain | verb | deepest owned rung |
|---|---|---|
| INSPECT | inspect source and repository | text search (AST: not yet) |
| RESEARCH | research the public web | fetch a page (crawl: not yet) |
| BUILD | mutate source under authority | typed op + red-before-green + rollback |
| TEST | run admitted tests | targeted subset / full regression |
| PROFILE | profile prefill and wall | `hcli report` |
| PROCESS | own, adopt and reap processes | reap owned orphans (never blind kill) |
| DEBUG | reproduce, localize, diagnose | project frame + assertion + source window |
| FUZZ | throw malformed input, keep crashes | minimized reproducer |
| FORENSICS | preserve failure evidence | persisted triage snapshot |
| REVERSE | identify a file without running it | magic/format (disasm: not yet) |
| RECOVER | checkpoint, resume, roll back | resume with divergence check |

REPORT is the one domain named but not yet a single owned capability.

## Durability

A conversation is a session (keyed by its first turn). Ask for a plan and it
persists with steps and the commit it was written against; say "apply it" later
and it resolves the plan without you restating it. When the context window fills,
old turns are archived (retrievable, not destroyed) and the objective is
checkpointed; a new process resumes it, and if the repo moved it says DIVERGED
rather than replaying a stale action.

## The rule the whole thing is built on

Every apparent model incapacity this campaign chased was the lab, not the body:
an empty tool slot, a dialect that cannot express an integer, stale bytecode, a
cwd-relative preflight, a fallback that matched by luck. Before blaming a body,
read which contract it was following. The ten expensive lessons are in
`receipts/future/HCLI_OPERATOR_LAWS.json`.
