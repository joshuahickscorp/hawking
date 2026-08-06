# Ramanujan — local scaffold, blocked on Hawking completion

Ramanujan is a buildable **non-authorizing scaffold**.  It is deliberately
separate from a live research system: Hawking completion plus the required
owner and production evidence must arrive before Ramanujan research, parent
restream, teacher-trace acquisition, or production launch can begin.

The local boundary is [HAWKING_COMPLETION_GATE.json](governance/boundary/HAWKING_COMPLETION_GATE.json).
Neither this folder nor any Ramanujan controller may flip it.

```
ramanujan/
├── scaffold/    fixture-only runnable code, data checks, tests, and local guards
├── container/   byte-pinned Q0 clean-replay environment
├── governance/  current boundary declarations and non-authorizing contracts
├── records/     audits, intake records, and runtime snapshots
└── docs/        dependency and local check-in guidance
```

`scaffold/ramanujan/container` is a narrow compatibility link for the
byte-pinned Q0 prover; it exposes only the container and does not duplicate
records or documentation.

## Safe local work now

```bash
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m ramanujan.status
python3.12 -m pytest -q ramanujan/scaffold/tests ramanujan/scaffold/data/tests
python3.12 -m ramanujan.gen_data_matrix --check
python3.12 -m ramanujan.odyssey --fixture-selftest
python3.12 -m ramanujan.odyssey --fixture-rehearsal  # disposable accelerated T0--T12/F0--F12/Q0--Q12 run
python3.12 -m ramanujan.odyssey --proto-plan          # future strict-V4 Flash program; no model mount
python3.12 -m ramanujan.odyssey --proto-footprint     # conservative 1-BPW artifact/RAM prediction
python3.12 -m ramanujan.odyssey --proto-gravity-plan  # future-only render contract and required gates
python3.12 -m ramanujan.odyssey --proto-condense-spec  # capability-first Condense promotion contract
```

These commands inspect or exercise fixture-only code.  They do not authorize
research and do not contact, download, or launch a parent model.
`PYTHONDONTWRITEBYTECODE=1` keeps these routine checks from adding cache
folders to the compact scaffold view.

`ramanujan.odyssey` is the compact T0--T12 / F0--F12 / Q0--Q12 pre-sandbox
control plane.  Its self-test exercises only the fixture T0--T2 boundary; the
full runners require injected fixtures, sealed hashes, and external review
evidence, and cannot enable research authority.

The future `ramanujan-proto` plan fixes DeepSeek V4 Flash as the student and
uses three **sequential** teacher passes—V4 Pro for structured planning,
GLM Math for critique/repair, and Kimi K3 for independent alternatives.  It
passes GLM only a hash-bound, verifier-dispositioned compact V4 trace reference,
then admits data only if all three teachers bind to the same formal statement.
It mixes only verifier-dispositioned trace data, never model weights, and
refuses the independently trained layerwise route that the Flash cascade
evidence already ruled out.  The render command creates a contract only: a
real DeepSeek-V4 Gravity adapter, source admission, and math-retention receipts
are still mandatory before any `.gravity` artifact can exist.

The future Condense spec is deliberately capability-first: it locks router and
shared-path precision, allocates routed-expert precision only from measured
evidence, arbitrates the three teachers on the same formalized statement, and
requires independent statement, Lean/exact, repair, false-lemma, transfer, and
reload receipts at every rung (4 → 3 → 2 → 1.5 → 1.25 → 1 BPW).  It cannot
reuse a passing result: each rung receipt binds the exact parent state,
candidate state, and frozen evaluator suite.  It cannot invent a TPS number:
runtime throughput remains an explicit Hawking receipt after a real artifact
and runtime exist.

See [Hawking dependency](docs/HAWKING_DEPENDENCY.md) for the boundary and
[local check-ins](docs/LOCAL_CHECKINS.md) for the repeatable development loop.
