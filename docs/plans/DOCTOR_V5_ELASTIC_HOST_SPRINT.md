# Doctor V5 elastic phases and host sprint isolation

Status: implemented as an unbound, default-off contract. It is not imported by
the live Doctor queue, worker, adapters, registry, or runtime defaults.

## What this adds

`tools/condense/doctor_v5_elastic_phase_scheduler.py` defines a future
source-bound generation with four bounded roles:

- at most one heavy prepare;
- at most one primary encoder;
- at most one serial finalizer;
- at most one measured-idle companion.

The hash-sealed host probe derives the reviewed M3 Ultra topology from four
read-only `sysctl` receipts: 28 physical cores, 20 performance cores, and 8
efficiency cores. The scheduler never treats all 28 cores as homogeneous heavy
capacity. A qualified 20-thread primary encoder is exclusive. Heavy prepare and
primary encode are mutually exclusive in both directions.

Prepare and serial finalization may overlap only when both have hash-bound
resource specifications and three fresh, ordered, green pressure/thermal
samples prove that the aggregate CPU and RAM envelope fits. The second owner
must also fit the minimum measured idle CPU/RAM over that window. Both state
owners retain the same overlap-envelope hash. Samples are at least five seconds
apart and bind the signed host probe, state generation, owner lease, cell, role,
and exact PID/start/command identity. This pure reducer path remains useful for
fixtures; production CAS overlap stays fail-closed until the same three-sample
window is collected by the trusted local observer.

Companion admission requires all of the following at once:

- a qualified exact vendor 8/12/16/20 selection for its exact tier and rate;
- an active non-20-thread primary encoder with a bound process/start identity;
- three fresh, hash-sealed samples bound to the exact encoder generation,
  primary cell/process identity, and host-probe hash;
- normal memory pressure and nominal/fair thermal state in every sample;
- selected threads and RAM reservation within the minimum measured idle
  envelope, using the contract's CPU and RAM budgets;
- a green aggressive swap-controller output bound to the exact hash-sealed swap
  state and exact controller-policy hash.

Prepare, encoder, finalizer, and companion admission all require the same fresh
swap-state/controller artifact. Production CAS reads the confined persisted
controller prior under the state lock and advances it from direct local
pressure/swap; caller state or raw `{mode, allow_launch}` booleans are not an
admission authority. Production companion launch remains fail-closed until its
three-sample idle window also moves behind that observer.

Encoder return closes new lending immediately. An existing companion is marked
for checkpoint/preemption, and finalization cannot begin until that companion is
released. Every production completion or release binds the owner lease and
exact PID/start/command identity to both a trusted acquisition observation and
a new direct `ps` negative-membership observation. It also hashes a confined
physical output plus a worker JSON receipt that binds the exact contract, cell,
phase, process, lease, and output. Caller-provided empty active lists or
`{sha256, bytes}` metadata cannot clear an owner. Companion release additionally
requires the checkpoint document; an observed live exact identity cannot be
cleared.

`tools/condense/doctor_v5_local_observer.py` is the shared production trust
boundary. While the same state lock is held it reads persisted state, exact
PID/start/command process generations, campaign-wide Doctor/Appendix/MOP/
cognitive-corpus/native-probe owners, topology, pressure, swap, thermal state,
and power. Receipts bind the observer source, persisted artifact hash, wall and
monotonic clocks, state generation, and lock lease. The observer performs no
model, GPU, corpus, queue, or runtime mutation.

Each production start additionally requires a source-bound, phase-specific
invocation-manifest entry. The entry freezes the executable artifact hash,
exact `KERN_PROCARGS2` argv template and enumerated substitutions, canonical
workspace cwd, hashed environment allowlist, unlisted Doctor/Hawking/Metal
control-variable refusal, and the exact worker/output receipt schemas. The
observer reads executable/argv/environment and cwd directly under the state
lock (`KERN_PROCARGS2` plus `lsof`), while caller executable, argv, cwd, and
environment declarations are discarded. Exactly one entry for the exact phase
must match; nearest-phase and fallback launchers are forbidden. The staged
manifest is intentionally empty, source-bound, and blocked until a new reviewed
generation freezes real waiting launchers.

State transitions are hash chained. A regular-file lock plus state-hash and
generation compare-and-swap serializes persistence. Only the winning CAS commit
receipt authorizes heavy work to begin; a pure in-memory transition is not
launch authority. Owner leases persist the contract, role, cell, process,
lease-generation, state-generation, and acquisition-time identities. Crash
recovery matches exact process and lease identities, never a cell label alone.
The scoped persistence helpers
validate and atomically write state and crash/rollback receipts with file and
parent-directory `fsync`, then verify readback. Those receipts never mutate
completed evidence or delete parent sources.

## Host sprint boundary

`tools/condense/doctor_v5_host_sprint_plan.py` performs read-only probes and
stages proposals. It executes no process-priority or operating-system changes.
The legacy pure gate can evaluate fixtures, but is explicitly
`caller-attested-test-only` and can never grant production authority. The
trusted gate accepts only persisted state/controller paths, then obtains the
host probe, owner set, campaign-wide heavy-owner inventory, clock, pressure,
swap, thermal, and power itself under the state lock. It remains proposals-only
and executes no action after releasing the lock. Replayed probes, stale swap
state, raw booleans, caller owner-free snapshots, and owner-set drift fail
closed.
The proposal inventory covers a process-scoped `caffeinate` wrapper, lower
companion priority, background QoS, optional indexing isolation, and optional
backup isolation. Every item is default-off, reversible, and requires explicit
authorization plus a new owner-free promotion gate.

The Spotlight item is intentionally unsupported as an executable command:
`mdutil` is volume-oriented and is not treated as proof of per-folder semantics.
It remains a manual System Settings proposal unless a later read-only preflight
can prove an exact reversible path contract. Fan/SMC writes, `launchctl`
mutation, automatic Spotlight/backup changes, negative-nice escalation, and
runtime-default changes are forbidden.

## Promotion and benchmark claim limit

These files do not promote themselves. A later owner-free checkpoint must
rebuild the bindings, pass the exact thread-profile and source gates, and create
a new source-bound generation. The present aggressive overlay still lacks the
qualified production thread profile, and the inert invocation manifest has no
reviewed phase launchers, so the staged elastic contract correctly remains
blocked.

Cheap tests and synthetic benchmark receipts prove contract behavior only.
They do not establish a speedup and cannot change the campaign ETA. An ETA
change requires a randomized, interleaved, owner-free, physical, real-artifact
full-stack A/B receipt under
`tools/condense/doctor_v5_single_device_benchmark.py`, with exact program,
input, output, receipt, invocation, and semantic identities plus wall/CPU/GPU,
RSS, scratch, disk, swap, memory-pressure, and thermal evidence. Component
speedups are never multiplied.

Practical expectation: a 20-thread encode remains exclusive. Safe gains can
come from eliminating phase bubbles, a measured prepare/finalizer overlap, or a
qualified small companion during a demonstrably idle non-20-thread phase. No
12–15 day estimate is claimed until the real full-stack receipt exists.

## Cheap verification

```sh
python3 -m unittest \
  tools.condense.tests.test_doctor_v5_elastic_host_sprint -v
python3 tools/condense/doctor_v5_host_sprint_plan.py stage
python3 tools/condense/doctor_v5_elastic_phase_scheduler.py stage
```

The stage order matters: the elastic contract binds the already staged,
hash-sealed host probe. Both outputs remain below
`reports/condense/doctor_v5_ultra/staged_acceleration/`.
