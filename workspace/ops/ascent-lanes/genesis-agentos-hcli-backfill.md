# LANE: genesis-agentos-hcli-backfill

## Class: CPU_ONLY / pre-100-TPS backfill

Read and verify the complete Genesis contract set before acting:

- `contracts/genesis/QWEN38_GENESIS_SYSTEM_DIRECTIVE.md`
  `881ae469e0287cf386467002d3fc7951524b47054ac6d7f753b94a8e4e3ceff7`
- `contracts/genesis/GENESIS_CONTINUITY_DIRECTIVE.md`
  `c4a58bc06575effb8f759dbb22c49abfc65e1957910b18917d45d02592d1fdbc`
- `contracts/genesis/GENESIS_OUTPUT_LAW.md`
  `9679490e8ae623a6fdb408fd906a15d676bc55926580f6d7ed60e9ea610c9ada`

Their exact combined binding is
`3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f`.
If a binding does not verify, stop with `GENESIS_SYSTEM_CONTRACT_INTEGRITY_FAILURE`.

## Objective

Use existing Hawking AgentOS/HCLI foundations to make the HCLI special-unit
path able to consume the *already resident* Genesis body as an explicit,
opt-in worker adapter. This is a CPU-only integration/test task while the
protected DeltaNet kernel lane owns the performance front.

The relevant existing seams are:

- `lab/hcli/special_unit.py` (`NativeQwen38Backend` and its receipts)
- `tools/agentos/genesis_resident.py` (verified resident client protocol)
- `lab/lineage/continuity.py` (durable worker identity / rebind semantics)
- `lab/tests/test_special_unit.py`

Do not create a parallel server, a new model loader, a new resident protocol,
or a new task ontology. Reuse these seams or honestly report why an adapter is
not legal with the current worker-role contract.

## Acceptance criteria

1. The opt-in path uses only the existing resident client protocol and can be
   driven entirely with an injected/stub client in tests. It must never require
   a cold Qwen binary invocation when the resident option is selected.
2. The caller must explicitly select an existing worker session (`child_a` or
   `child_b`); do not silently repurpose `parent` or `protected_test`, and do
   not add an ungoverned fifth role.
3. It preserves the existing refusal when a protected GPU lane is active. This
   lane itself must not acquire `gpu_lane_lock.sh`, load weights, run a native
   decode, run a benchmark, or request any live Genesis generation.
4. A resident reply lacking the existing verified contract provenance, a live
   body indication, usable text, or valid zero-fallback evidence must be
   rejected. Never convert a refusal into a synthetic model result.
5. It does not nominate a child, alter a lineage slot, invoke a promotion gate,
   modify protected tests, or claim a TPS/BPW change. A worker is not a child.
6. Add focused CPU-only tests proving the successful injected path and each
   important refusal. Existing native-binary behavior must remain covered.

## Scope

EDIT `lab/hcli/special_unit.py`.

EDIT `lab/tests/test_special_unit.py`.

EDIT `lab/tests/test_genesis_continuity.py` only when an adapter-level
continuity assertion is genuinely required; otherwise leave it untouched.

Do not edit the resident body, contracts, promotion gates, GPU lock,
`tools/ascent_daemon.py`, or Qwen Metal/runtime code.

## Verification

VERIFY `lab/tests/test_special_unit.py` and `lab/tests/test_genesis_continuity.py`.

Acceptance: `python3 -m pytest -q lab/tests/test_special_unit.py lab/tests/test_genesis_continuity.py` exits 0.

Run only CPU-safe checks, at minimum:

```text
python3 -m pytest -q lab/tests/test_special_unit.py lab/tests/test_genesis_continuity.py
```

## Completion report

End with exactly:

```text
LANE: genesis-agentos-hcli-backfill
STATUS: SHIPPED | PARTIAL | BLOCKED
BASELINE_NS_PER_TOKEN: N/A (CPU-only integration; no timing claim)
RESULT_NS_PER_TOKEN: N/A (CPU-only integration; no timing claim)
REPS: N/A
CORRECTNESS: focused stub/injection tests + exact command
FILES: <paths>
RECEIPT: <path or N/A>
NEXT_BOTTLENECK: <truthful next action>
```
