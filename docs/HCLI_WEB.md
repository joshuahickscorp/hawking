# Hawking in a browser

`hcli web` puts the sealed resident behind [Open WebUI](https://github.com/open-webui/open-webui).

## Cheat sheet

Every command below was run and is real. All of them need the repo as the
working directory (see *Known limitations*).

```bash
cd ~/Downloads/hawking
```

**Open the browser chat** — starts the resident, starts Open WebUI, opens the tab:

```bash
/usr/local/bin/python3.12 -m hcli web
```

**Just the endpoint**, to point your own client (or another UI) at it:

```bash
/usr/local/bin/python3.12 -m hcli serve --port 8011
```

**Ask one question from the shell:**

```bash
curl -s http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Explain DeltaNet in three sentences."}]}'
```

**Is it up, and what is it running?**

```bash
curl -s http://127.0.0.1:8011/health
```

**Measure whatever resident is loaded right now** -- writes a receipt under
`~/.hcli/reports/`:

```bash
/usr/local/bin/python3.12 -m hcli report
```

**Serve a different body** (any ModelLake specimen, not only the sealed one):

```bash
/usr/local/bin/python3.12 -m hcli web --model /Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-4B-Instruct-2507@cdbee75f17c0
```

**Stop it** — Ctrl-C in the `hcli web` terminal stops what that command started.
To stop a surface left running in the background:

```bash
pkill -f "hcli serve"; pkill -f "open-webui serve"
```

Logs: `~/.hcli/web/serve.log` and `~/.hcli/web/webui.log`.

## What talks to what

```
browser  ->  Open WebUI :8080  ->  hcli serve :8011/v1  ->  persistent resident
```

`hcli serve` is thin on purpose. `make_backend_for_model` already returns a
backend holding a persistent `ResidentProcess`, so the server is HTTP in,
`backend.complete` out — no second runtime and no second model resolution.

Two OpenAI surfaces already existed and neither can drive a chat:

* `crates/hawking-serve` wants `model-*.gravity` shards. Every body on disk is
  `.hq30uq4` / `.f32v2` / `.hgrafv01`, so it has nothing to serve.
* `tools/hcli_resident/serve_sealed.py` is seal-verified and correct, and its
  own docstring says the binary **reloads the model per call**. That is a
  measurement instrument, not a chat surface. It is unchanged.

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

Measured with `hcli report` against the resident that was loaded at the time
(sealed-3.14, greedy). Re-run it against whatever resident you are serving --
the numbers below are one body on one machine, not a property of HCLI.

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

* **Repo-locked launch.** `hcli` on `PATH` runs the deployed snapshot under
  `~/.local/share/hcli/current` (it prints a STALE warning itself), and the repo
  package is not importable from elsewhere, so these commands need
  `cd ~/Downloads/hawking`. `hcli install-shims` is the documented fix and has
  not been run here.
* **No tools in the chat.** The browser chat is the resident answering from its
  own weights. Repo inspection and public-web research exist in HCLI
  (`web.search`, `web.fetch`, and the repo tools) but are not wired through this
  surface yet.
* **One resident, one request at a time.** The server threads, the connector
  does not.
* **The model's self-report is unreliable.** Asked what it is running through,
  it answered "Qwen3, developed by the Qwen team at Alibaba" in the browser and
  "GPT-4o" over curl. It is a 3.14-EBPW body confabulating its own identity;
  `/health` is the authority on what is loaded, not the model.
