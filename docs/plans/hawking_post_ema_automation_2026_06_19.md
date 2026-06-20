# Hawking Post-EMA Automation

Date: 2026-06-19

This is the handoff rail for the current RWKV7 QAT arc. Its job is to avoid
manual babysitting after the training EMA crosses the first release-candidate
threshold.

## One Command

```sh
tools/training/hawking_after_ema.sh
```

Default behavior:

- wait for the latest training segment to reach `loss_ema <= 6.0`
- round up to the next 5-step checkpoint
- wait for `state_dict.pt` to become stable
- freeze the checkpoint into `artifacts/lowbit_rwkv7/hawking_arc/`
- stop the live QAT/autocycle rail to free MPS memory
- run deterministic short PPL eval against the base model and candidate
- run deterministic long PPL eval against the candidate
- generate deterministic prompt samples
- export an HF-shaped directory with `--no-gguf`
- write `manifest.json`, `ppl.jsonl`, `samples.jsonl`, `report.md`
- generate `frontier_queue.sh` for the next training branches

## Current ETA

Current observed state before this automation was added:

- step: 57/60
- EMA: 6.178893
- PPL EMA: 482.46
- recent step time: about 30 minutes

If the present EMA drop rate holds, `EMA <= 6.0` is plausible around steps
70-90. On the current 18 GB M3 Pro run, that is roughly 6-16 wall-clock hours.
The estimate is wide because swap pressure is the real limiter.

Post-trigger verification rough budget:

- freeze/publish: minutes
- base short PPL: 20-60 minutes
- candidate short PPL: 20-60 minutes
- candidate long PPL: 1-3 hours
- deterministic samples: 20-90 minutes
- HF export: 10-30 minutes

Overall from now to a verified first release candidate: roughly 8-24 hours.

## Next Frontier Queue

The generated frontier queue is intentionally a plan script, not an automatic
multi-branch launcher. On 18 GB, running several MPS branches at once would
destroy throughput. The intended order is:

1. fast learner: 512 context, grad accumulation 8
2. balanced continuation: 768 context, grad accumulation 8
3. quality anchor: 1024 context, grad accumulation 16

Promote only if the branch beats the frozen EMA-6 candidate on PPL and sample
sanity.

## Why EMA

Raw loss is a useful hot signal but too noisy for release decisions. PPL EMA is
`exp(loss_ema)`, so the EMA gate is the stable version of the PPL frontier.

The release-candidate gate is multi-signal:

- EMA crosses target
- PPL keeps falling
- recent raw loss remains hot
- token speed remains usable
- eval PPL transfers outside the training stream
- deterministic samples do not regress
