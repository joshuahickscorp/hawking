# Hawking Sandbox S0 foundation — 2026-09-17

`hawking.sandbox_world` adds the missing restart-readable S0 problem-world
record. It is state only: it does not launch workers, schedule providers, or
create a third user-facing mode.

## Implemented

- persistent S0 world under `.hawking/sandbox/worlds/`;
- disposable branches with parent/source/worktree identity;
- bounded findings with falsifier, action, evidence, and candidate claim level;
- independent evaluator registration across software, optimization,
  mathematics, research, and security categories;
- candidate and artifact provenance records that cannot self-promote;
- protected-controller-only finding assimilation;
- root/branch budget reservation and settlement;
- child failure isolation: failed branches leave the S0 root and sibling
  branches active;
- bounded event replay and restart-readable world snapshots;
- capability catalogue and Auto allocation metadata without a second scheduler.

The existing `execution_sandbox.ExecutionSandboxPolicy` remains the path/action
authority, and `worker.option_c.OptionCSandbox` remains the executor/reviewer/
controller lifecycle. S0 only supplies their durable problem-world context.

## Evidence

```text
python3 -m pytest -q \
  hawking/tests/test_sandbox_world.py \
  hawking/tests/test_worker_option_c.py \
  hawking/tests/test_worker_special_unit.py
98 passed
```

## Claim boundary

This is S0 persistent-state foundation evidence, not Sandbox release
acceptance. Tool/device discovery, Auto allocation execution, evaluator
integration with real runs, and crash/restart tests around live branches still
need an admitted P19 tranche after its graph dependency is accepted.
