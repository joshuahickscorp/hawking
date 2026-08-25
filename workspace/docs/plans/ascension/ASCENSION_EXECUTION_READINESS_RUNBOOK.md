# Ascension execution readiness runbook

**Status:** preparation only. This runbook authorizes no model-body fetch, model
launch, cache mutation, or deletion. It turns the Ascension Bible's sequence
into an operator-facing stop/go checklist for HCLI.

## Live-operation rule

The DeepSeek-V4 gravity restream is the sole active heavyweight transfer. While
it is in progress:

- do not start GLM, Kimi, Qwen 30B, or Qwen 80B body downloads;
- retain the DeepSeek journal, metadata, and content-addressed chunks;
- reserve at least 25 GiB of free disk and stop admitting new ranges at a
  breach;
- supervise the downloader process-group below 5 GiB RSS and stop scale-up on
  swap growth or a resource-receipt failure.

“Use idle CPU” means admitted useful I/O/validation work, not artificial CPU
spinning. A network-bound transfer may correctly leave CPU idle.

## No-download preparation now

| Lane | Allowed now | Not allowed now | Evidence required before body fetch |
|---|---|---|---|
| Qwen 30B executor | Pinned source/config/revision manifest; tensor and loader design; metadata-only parity plan | Any weight or cache fetch | Source admission, exact loader/forward support, Gravity recipe, green resource receipt, artifact reservation, profiler parity, controller approval |
| Qwen 80B reviewer | Pinned source/config/revision manifest; hybrid DeltaNet/gated-attention design; metadata-only state ledger | Any weight or cache fetch | All 30B gates plus exact Qwen3-Next parser and hybrid state proof |
| HCLI agent OS | Interface contracts, fixture-backed scheduler/retrieval/tool/memory tests | Model execution or autonomous external action | Proto offload completion and controller/TG authorization |
| Kimi strategic lane | Corpus/evaluation design, residual hypothesis, receipt schema, metadata admission | Teacher stream, capture, adapter fitting, or a second body | Sealed Proto, forward gate, clean one-donor storage window, and residual-teacher admission |

## Kimi decision rule

`GLM -> DSV4F` remains the canonical structural/math transfer. Kimi is not a
second copy of that transfer. It can be admitted only as a **reversible,
separately-addressed strategic residual** after a same-membership held-out A/B
shows a unique gain in a named target (for example HCLI tool policy,
long-horizon planning, or agentic coding) with no protected regression.

Reject or defer a Kimi pass when it is justified only by “more distillation,”
when the GLM-only baseline is unsealed, when the evaluation membership differs,
or when it attempts to compensate for absent DSV4F loader/forward support.
This keeps architectural work separate from teacher overlap and makes rollback
possible.

## Promotion sequence

```text
DeepSeek restream sealed
  -> Proto-Frankenstein sealed, offloaded, hash-verified, out of active envelope
  -> Qwen source + runtime + Gravity preflight green
  -> Qwen 30B one-body fetch / pack / parity / HCLI executor gate
  -> Qwen 80B exact hybrid-runtime gate, then one-body fetch / pack / parity
  -> HCLI controller and TG3 approval
```

Every arrow is fail-closed. A plan or metadata seal is not a runtime,
throughput, artifact, or capability receipt.

## Operator receipt checklist

Before changing any model body state, record all of the following in a
content-addressed receipt:

1. Source repository, immutable revision, config digest, license/authority.
2. Disk reservation, current free bytes, active heavyweight-transfer lease,
   and eviction classification (never delete unclassified shards).
3. Process-group RSS high-water mark, swap delta, CPU/throughput observations,
   and the resource-supervisor result.
4. Exact runtime loader/forward capability and Gravity representation proof.
5. Held-out membership hash, baseline comparison, protected non-regression
   measurements, and controller decision.

If any entry is missing, the decision is `BLOCKED`, not an assumption that the
next model may be fetched.
