# Hawking in a browser

`hawking web` puts the admitted Hawking resident behind
[Open WebUI](https://github.com/open-webui/open-webui), and the dropdown in that
browser page talks to that one resident. The daemon owns the provider and
Open WebUI child; `hawking web` exits after opening the client surface, so a shell
launcher does not remain as another Python process.

## Cheat sheet

No flags needed for any of it. Works from any directory.

```bash
hawking web                    # browser chat, opens the tab
hawking use                    # admitted Gravity bodies only
hawking use KIMI_P0_OPERATIONAL # switch an admitted body when supported
hawking web KIMI_P0_OPERATIONAL # ... starting on an exact admitted body
hawking serve                  # endpoint only, no browser
hawking report                 # measure the loaded body, writes a receipt
hawking report KIMI_P0_OPERATIONAL # measure the admitted body
hawking stop                   # put down what web/serve started
hawking --help                 # the verbs, grouped
hawking                        # interactive session in the terminal
```

A body is named by any unambiguous prefix, so `hawking use qwen3-14` is enough.
An ambiguous one is refused with the candidates rather than resolved to a guess.

If `hawking` is not on your PATH, or prints a STALE warning, reinstall the shims
with the interpreter you want them to use:

```bash
cd ~/Downloads/hawking && /usr/local/bin/python3.12 -m hawking install-shims
```

That interpreter matters: the shims previously pointed at a venv with no `mlx`,
which would have failed most ModelLake specimens at load while `python -m hawking` from
the repo worked perfectly.

Logs: `~/.hawking/web/serve.log`, `~/.hawking/web/webui.log`. Reports:
`~/.hawking/reports/`.

## What talks to what

```
browser  ->  Open WebUI :8080  ->  hawking serve :8011/v1  ->  persistent resident
```

`hawking serve` is thin on purpose. `make_backend_for_model` already returns a
backend holding a persistent `ResidentProcess`, so the server is HTTP in,
`backend.complete` out — no second runtime and no second model resolution.

Two OpenAI surfaces already existed and neither can drive a chat:

* `crates/hawking-serve` wants `model-*.gravity` shards. Every body on disk is
  `.hq30uq4` / `.f32v2` / `.hgrafv01`, so it has nothing to serve.
* `tools/hawking_resident/serve_sealed.py` is seal-verified and correct, and its
  own docstring says the binary **reloads the model per call**. That is a
  measurement instrument, not a chat surface. It is unchanged.

## Switching bodies

`/v1/models` lists only bodies admitted by the Gravity artifact registry, so
Open WebUI cannot accidentally offer raw ModelLake specimens, archived
Ascension/Qwen3.8 artifacts, or an unqualified checkpoint. Research access is
separate and explicit; the normal dropdown is an execution surface, not the
scientific population.
`hawking models` reports raw ModelLake downloads separately as source specimens.
Selecting an admitted body binds its exact catalog path, revision, and supported
action before the switch endpoint is called.

Three rules the switch keeps:

* **One body at a time, old one stopped first.** These are 8-150 GB artifacts on
  a 103 GB machine. Spawning before stopping would page the box into swap, so a
  switch is stop-then-start under a lock, and a request arriving mid-switch
  waits rather than racing a half-loaded body.
* **A body that cannot fit is refused, not attempted.** `Qwen2.5-72B-Instruct is
  145 GB and this machine has 103 GB` -- named, with both numbers, instead of
  thrashing for ten minutes and then failing.
* **An unknown model name is never a silent fallback.** Asking for `gpt-4o`
  returns a 404 saying which body *is* loaded. Answering as a different model
  than the caller selected is the same lie as serving a different sampler.

The sampler policy follows the loaded body: sealed-3.14 refuses `temperature`,
and an MLX specimen that samples accepts it.

## Things this deliberately refuses

**Samplers, on a greedy resident.** sealed-3.14 decodes argmax. `temperature:
0.8` returns a 400 naming the accepted value and where in Open WebUI to set it,
because silently serving a different sampler than the caller asked for is how a
measurement ends up describing something else. Leave Advanced Params unset.

**Pretending to stream.** The connector returns a completed string; there is no
incremental hook yet. `stream: true` emits one content delta, and every
response and `/health` carry `streaming: "single-chunk"` so SSE frames are never
read as evidence of token-by-token decode.

## Measured, on this machine

Measured with `hawking report` against the resident that was loaded at the time
(sealed-3.14, greedy). Re-run it against whatever resident you are serving --
the numbers below are one body on one machine, not a property of HAWKING.

**Cold prefill**, each prompt distinct so nothing is cached:

| prompt tokens | wall to answer | prompt tok/s | ms per prompt token |
|---:|---:|---:|---:|
| 173 | 5.23 s | 33.1 | 30.2 |
| 598 | 17.05 s | 35.1 | 28.5 |
| 2188 | 88.6 s | 24.7 | 40.5 |
| 4313 | 147.1 s | 29.3 | 34.1 |

Decode, separately: 150 tokens in 4.89 s -- 30.7 tok/s, 32.6 ms per token.

**A cold prompt token costs what a generated token costs.** 28.5-40.5 ms to
absorb a prompt token against 32.6 ms to produce one. It is not superlinear --
the 4313-token point is faster per token than the 2188-token one, which rules
out the quadratic reading a three-point series suggested. It is flat, at the
wrong price: the signature of prefill stepping the prompt one position at a time
rather than batching positions into matrix-matrix work. That is the remaining
S033 §13 target, and it is worth roughly a 20-30x cut in time to first answer.

**Prefix reuse is wired, and it holds across conversation turns:**

| turn | prompt tokens | wall |
|---|---:|---:|
| 1 (cold) | 720 | 18.16 s |
| 2 (extends the cached prefix) | 739 | 0.51 s |
| 3 (extends again) | 758 | 0.51 s |
| control: different prompt, same size | 720 | 16.93 s |

So a conversation does NOT re-pay its history -- only the first turn pays, and
every turn after it starts generating in about half a second. The control is
what licenses that claim: a same-sized but different prompt goes back to
16.9 s, so the 0.51 s is a real prefix hit and not a warm-process artifact.

The usability shape that falls out: **time to first answer is the wall, and it
is a first-turn cost.** A 4 k-token opening prompt is ~2.5 minutes; every turn
after it is fast.

## Known limitations

* **Repo-locked launch.** `hawking` on `PATH` runs the deployed snapshot under
  `~/.local/share/hawking/current` (it prints a STALE warning itself), and the repo
  package is not importable from elsewhere, so these commands need
  `cd ~/Downloads/hawking`. `hawking install-shims` is the documented fix and has
  not been run here.
* **Tools are mode-bound.** Ordinary resident chat is not a repository writer.
  Explicit Hawking Goals and remote worker loops receive only the capability
  projection admitted for that Goal; the remote-worker surface above adds no
  provider-owned tool authority.
* **One resident, one request at a time.** The server threads, the connector
  does not.
* **The model's self-report is unreliable.** Asked what it is running through,
  it answered "Qwen3, developed by the Qwen team at Alibaba" in the browser and
  "GPT-4o" over curl. It is a 3.14-EBPW body confabulating its own identity;
  `/health` is the authority on what is loaded, not the model.

## Current remote-worker surface

### Hawking Auto

Open WebUI's default model on the minimal Hawking review surface is
`KIMI_P0_OPERATIONAL`, rendered to the user as **Hawking**. The artifact is
present on this device, but it is a frozen legacy MLX identity and remains
withheld from canonical hawkingd execution. Displaying it does not grant it a
runtime admission.

The logical role names `pulsar`, `magnetar`, and `themis` remain internal
classification vocabulary only; they are not model-picker entries. Hawking
Auto and the cloud worker roster remain available to explicit advanced
requests, but they are not presented as the default model list. When used,
Auto selects from the live, credential and gateway-policy-admitted cloud roster
using the request's task signals:

* tool/build/repository work prefers `deepseek/deepseek-v4.1-flash`;
* default and fast execution use `deepseek/deepseek-v4.1-flash`;
* review, verification, and explicit quality/deep profiles may select
  `moonshotai/kimi-k3`;
* explicit premium/independent profiles may select `moonshotai/kimi-k3`;
* multimodal requests prefer `qwen/qwen3.8-flash` when the live catalog admits it.

The current cloud Auto roster is intentionally only those three exact live
OpenRouter ids: DeepSeek V4.1 Flash, Kimi K3, and Qwen3.8 Flash. The local
logical roles (`pulsar`, `magnetar`, and `themis`) are classes, not provider
model ids, and can remain selectable when their native runtime is admitted.

The ordinary `/v1/models` response is therefore a small Hawking control
surface, not a mirror of every OpenRouter model. An explicit
`/v1/models?search=...` (or `?q=...`) searches the live provider catalog and
returns at most 40 matching rows. Goals, workers, tools, receipts, cost gates,
and runtime controls are Hawking-owned capabilities and are intentionally not
listed as fake models.

The response carries a sanitized `hawking.auto_route` receipt with the selected
model, task class, policy version, and live candidate set. Provider credentials
remain inside the Hawking resolver. The Open WebUI process is still supervised
by `hawkingd`, connects to the same `/v1` endpoint, uses the Hawking label
and OLED-dark theme for the local single-user surface, and carries no provider
model list by default.

H-Web is also the operator surface for Hawking-owned remote cognition. The
default model picker contains only the Hawking/Kimi P0 review identity;
explicit `/v1/models?search=...` queries can still inspect the live provider
catalog when an advanced operator asks for it. Selecting an OpenRouter id does
not create a provider-owned mission. A durable Hawking Goal and WorkUnit remain
the authority, while the remote model is an ephemeral worker lease.

The worker fabric is visible at `GET /v1/workers` and can be controlled locally
through `POST /v1/workers/action` with `status`, `pause`, `resume`, or `kill`.
Inside an explicit Hawking worker loop the model receives three additional
Hawking-owned doors:

* `hawking.worker.spawn` creates a bounded, read-only `worker_research` child
  through the existing Goal/WorkUnit owner;
* `hawking.worker.status` reads the current worker or a direct child;
* `hawking.worker.kill` releases the current worker or a direct child while
  preserving the checkpoint.

Child fan-out, delegation depth, concurrency, wall time, provider cost, and
the current WorkUnit's capability projection are enforced by Hawking. Child
workers cannot mutate the repository; a separate discrete Goal is the only
path that can reach the existing mutation engine and lock. Browser clients
never receive or supply provider credentials, and a provider conversation is
not a completion or acceptance authority.
