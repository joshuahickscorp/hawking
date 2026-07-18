# Doctor V5 post-120B mountain ladder

This is an inert capacity and campaign compiler for the work after the isolated
GPT-OSS 120B run. It does not download, quantize, launch a worker, register an
adapter, alter a runtime default, or mutate the live Doctor queue.

The compiler carries the exact 120B experiment shape into the three already
selected horizons:

| Horizon | Source lifecycle | Main purpose |
|---|---|---|
| DeepSeek-V4-Flash (nominal 284B) | full-local, then bounded stream | first post-120B architecture bring-up |
| Kimi-K2.6 (nominal 1.1T) | guarded full-local, then expert stream | largest conventional repository in the SSD lifecycle |
| DeepSeek-V4-Pro (nominal 1.6T) | immutable remote ranges only | largest parameter target; never full-installed |

Every horizon gets 40 blocked cell templates: the same ten physical rates and
four isolated Doctor branches as GPT-OSS 120B. Each also gets nine dependency
phases from architecture sealing through promotion. The phases contain no
commands and cannot become executable by merely rewriting a status field.

## What is carried forward

- one bounded source traversal fan-outs ten independently receipted rate outputs;
- branch evidence and artifacts remain isolated;
- 8/12/16/20-thread and 2/3/4/6-depth choices require same-host physical A/B;
- deterministic block merge and ordered phase overlap remain exact-output gated;
- RAM lanes, zero-swap admission, thermal stops, WAL recovery, CAS promotion,
  rollback, native parity, and zero-skip quality stay mandatory;
- progressive 1% → 5% → 20% → 100% coverage can reject weak work early, but it
  does not erase any cell identity or substitute for all 40 terminal receipts.

The proactive work allowed during the current run is metadata-only: architecture
inventory templates, source/range manifest preparation, disk projections,
dependency graphs, and future receipt schemas. Any source read, download, model
conversion, or physical benchmark remains behind the later authority gates.

## Storage and residency are separate gates

The compiler calculates both against the machine snapshot:

1. source/window + one candidate + 150 GB disk reserve + 32 GB cache reserve +
   32 GB stream workspace must fit the disk lifecycle; and
2. candidate including an explicit 8% planning overhead + 20 GB runtime working
   reserve must fit the 78 GB process budget.

Those overhead values are conservative planning reservations, not codec claims.
Measured artifact bytes and runtime peaks replace them at the storage canary.
With this envelope, Kimi 0.33 bpw is the safer resident target and V4-Pro 0.25
bpw is the first clearly resident planning target. V4-Pro 0.33 remains
borderline/non-resident once explicit runtime and artifact overhead are charged.

## Cheap commands

```sh
python3.12 tools/condense/doctor_v5_mountain_ladder.py inspect
python3.12 tools/condense/doctor_v5_mountain_ladder.py build
python3.12 tools/condense/doctor_v5_mountain_ladder.py verify
```

`inspect` writes nothing. `build` writes only the unbound JSON plan under
`reports/condense/doctor_v5_unbound/post120_mountain/`. `verify` recalculates
the matrix, capacity math, inertness gates, and plan hash.
