# HCLI resident daemon

## Roadmap relationship

The daemon is Hawking infrastructure, not a separate primary project. The
canonical law is **the daemon serves Hawking; Hawking does not serve the
daemon**. Roadmap execution stays ahead of speculative daemon qualification:

- CUDA remains hardware-blocked while CUDA architecture is ported and improved
  for Apple Silicon/Metal.
- Heavy resident load, measured UMA self-evacuation, protected GPU unload/reload,
  multi-lane contention, and long mixed-workload runs are qualified only when a
  real Hawking work unit needs them.
- If roadmap work exposes a daemon defect, patch and verify that defect, then
  return immediately to the roadmap. Do not broaden the daemon architecture
  speculatively.

HCLI now has a durable resident control loop. The resident is split into two
process roles:

- a small supervisor that owns heartbeat, memory admission, restart limits,
  and process ownership;
- one disposable worker that may construct `AgentOS`, load the configured
  model, execute a bounded mission slice, checkpoint, and exit.

The mission and DAG are disk state. A worker PID or loaded model is not. This
means a resident can wait for a safe memory window, unload itself for a
protected experiment, and return without losing the mission.

Start it with an explicit goal:

```text
hcli resident start --workspace /path/to/workspace \
  --goal "continue the current bounded research mission" \
  --model /path/to/profile.json
```

Inspecting status does not construct a `Controller` or open model weights:

```text
hcli resident status --workspace /path/to/workspace
hcli resident stop --workspace /path/to/workspace
hcli resident clean-room --workspace /path/to/workspace \
  --reason "protected accelerator experiment"
hcli resident resume --workspace /path/to/workspace
hcli resident queue --workspace /path/to/workspace \
  --id source-check --role research \
  --description "verify the pinned source"
```

Use `hcli agentos resident ...` for the equivalent namespaced command.

When host pressure is high, free memory is below the configured reserve, or
swap exceeds the configured ceiling, the supervisor records
`WAITING_FOR_MEMORY`, asks only its owned worker session to evacuate, and waits
for the next probe. If no ceiling is supplied, the daemon uses the same
conservative 2 GiB default as HCLI's runtime `MemGate`; `HCLI_SWAP_CEILING_GIB`
may be used when the machine policy explicitly changes. It never scans or
kills unrelated applications. A model worker is only launched after the
memory preflight passes.

Worker output may include a bounded `child_workunits` proposal. The proposal is
persisted as compact evidence and can refill the DAG only when the parent has a
passing verifier result. Each child is independently scheduled and verified;
the model cannot certify either its own work or its descendants.

External control-plane work can be queued without loading a model. The next
worker cycle admits pending WorkUnits into the durable mission DAG. The CLI
form is shown above; the Python form is:

```python
from hcli.agentos import ResidentDaemon, WorkUnit

ResidentDaemon(workspace).enqueue_workunit(
    WorkUnit(id="source-check", role="research", description="verify the pinned source")
)
```

The supervisor records an explicit behavior decision on every heartbeat:
`DISPATCH_WORK`, `MONITOR_WORKER`, `WAIT_FOR_MEMORY`, `WAIT_FOR_CLEAN_ROOM`,
`RESTART_WORKER`, `ESCALATE_FAILURE`, or `WAIT_FOR_WORK`. Idle means no
model-generated busywork is authorized; new work or verified evidence is the
trigger for another cycle. The physical body registry records `CONFIGURED`,
`LOADING`, `LOADED`, and `UNLOADED` independently of the logical mission;
`LOADED` is emitted only after the controller reports an admitted runtime, not
merely because a worker process was constructed.

The lightweight resident qualification is
`hcli/tests/test_hcli_resident_daemon.py`. It uses a fixture engine and a
short-lived child process, so it does not load Qwen3.8 or consume GPU memory.

For a serving-only latency experiment, set
`HAWKING_QWEN38_SERVE_UNTIMED=1` before starting `genesis-resident`. This uses
the resident untimed token loop and plain Metal fence, so per-token GPU timing
vectors are intentionally absent; health and propose responses report
`untimed_resident_fast`. Leave it unset for measured qualification runs.

The guarded production-path smoke is
`tools/headless/hcli_resident_native_smoke.py`. It validates the sealed Qwen3.8
profile and performs a dry host/runtime admission first. On a packed host it
proves that native work remains queued and emits a passing safety-guard receipt
without opening weights. Only when both gates pass does it run one bounded
resident mission in a disposable workspace. It emits
`receipts/headless/HCLI_RESIDENT_NATIVE_DAEMON_SMOKE.json` and never overrides a
memory refusal.

Run the deeper qualification goal with:

```text
python3 tools/headless/hcli_resident_qualification.py
```

This runs a real source/test AgentOS mission, verifies evidence-derived child
continuation, then SIGKILLs only a disposable worker under the real supervisor.
It emits `receipts/headless/HCLI_RESIDENT_CRASH_RECOVERY.json`. The receipt is
explicitly control-plane evidence; it makes no model-quality or GPU-performance
claim.
