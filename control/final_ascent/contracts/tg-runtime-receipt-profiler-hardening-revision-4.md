# Temporal Gravity runtime receipt/profiler hardening — Revision 4

This addendum follows Revision 3 and closes live-binding/public-reseal defects
found by immutable audit. Apply it only after the Revision-3 canonical
Rust/Python payload and runner-median corrections are present. Preserve every
earlier source-body-free, false-fence, fixed-TG-threshold, physical-evidence,
and no-promotion requirement.

The only authorized Revision-3 predecessor is the following exact six-file
set:

- `crates/hawking-core/examples/gravity_glm_tps.rs`
  blob `5175ab5c64e5e70c49312b479613181f844fa1bb`;
- `tools/condense/glm52_runtime_speed_gate.py`
  blob `cbcd0108a68ff0afa732f3a0860b10cfae70ad68`;
- `tools/condense/gravity_profiler_acceptance.py`
  blob `2edce7bc98a21598af711f635293ff3fa0aade14`;
- `tools/condense/tests/test_glm52_runtime_speed_gate.py`
  blob `39c5a0dc43a4b493b7886fc2c6ff14d552dfa1a3`;
- `tools/condense/tests/test_gravity_profiler_acceptance.py`
  blob `4976cef26399eed8209523db2a7c37a87e6bc39d`;
- `tools/condense/hawking_receipt_canonical_vectors.v1.json`
  blob `f0cd3cbfa0e2b544417d0365af94a3d79ad86d60`.

Refuse before editing if any identity differs. Remove `.serena`; it is not a
deliverable.

## 1. Mandatory live evidence, not attacker-resealable structure

A public content hash plus public semantic seal proves consistency, not
provenance. Production validation must resolve and hash the mandatory live
authority files and compare their complete closed records with the receipt.
Embedded or caller-supplied substitutes cannot replace them.

Require mandatory path/hash/closed-schema bindings for:

- raw execution record;
- raw hardware probe;
- workload seed, prompt identity, tokenizer identity, and ordered output trace;
- executable and loaded-library/Metal identities;
- provider and exact resolved artifact map;
- physical instrumentation stream;
- artifact index, register, approval, and their predecessor chain.

The raw execution record itself must bind the workload and every reported
context/decode/sample/window/prefill/TTFT value. It must also bind request,
run, token order, output trace, provider, executable, hardware probe, and
physical instrumentation identities.

Switching a receipt from a named live record to internally consistent embedded
evidence refuses. Keeping an old `source_hashes.raw_execution_record` or
`raw_probe_record` while consuming different embedded evidence refuses.

## 2. Closed schemas and chain authority

Every JSON authority loader rejects duplicate keys, multiple JSON values,
NaN/infinity, missing fields, extra fields, unknown schema/version, and unknown
nested fields before treating a byte hash as authority.

Artifact index validation must enforce the exact frozen schema, version,
families, names, paths, hashes, and semantic seals. Register and approval must
enforce exact gate sets, verdicts, predecessor identities, and mutually
consistent `register_sha256`/`predecessor_register_sha256` fields.

## 3. Physical instrumentation parity

The speed gate and profiler must call the same validator over the same live
instrumentation record. An integer plus nonempty `source` and `method`, or an
arithmetically plausible attacker-created event list, is not evidence.

Bind instrumentation schema/version, exact source blob, executable/build,
device, provider, run, request, token identities, event order, counter source,
method, raw event bytes, terminal hash, and aggregate arithmetic. Profiler
production token validation must not use a weaker parallel helper.

## 4. Required public-reseal matrix

For each case, begin with an otherwise valid receipt, mutate one fact, recompute
every public hash and semantic seal available to the caller, and require a
specific live mismatch:

- Git commit or dirty state;
- artifact path/hash/family/index/seal;
- register path/hash/entry/gates;
- approval path/hash/seal/predecessor/gates;
- executable/build or loaded-library/Metal identity;
- hardware or raw probe;
- provider or resolved map;
- raw record path/hash/run identity;
- seed, prompt, tokenizer, or output trace;
- context/decode/sample/window/prefill/TTFT;
- bytes/ops/command buffers/memory/swap/thermal/attribution;
- any source path/hash.

Mandatory executable counterexamples:

- replace workload seed/prompt/tokenizer/output trace with valid attacker
  values and reseal;
- replace physical bytes with `777` and plausible events `333 + 444`, arbitrary
  source hash, request, run, and token identities, then reseal;
- keep a live `5.0 ms` raw record but replace embedded/receipt samples with
  `0.5 ms`, reseal, and attempt TG20 through TG1;
- keep the old raw-probe source binding but replace embedded CPU/probe with
  `Attacker CPU 9000`, then reseal;
- use an artifact index with unknown schema `attacker.unknown.v999` and an
  unknown nested field.

All must refuse in both speed gate and profiler.

## 5. Exact JSON parser matrix

Feed each authority loader:

```text
{"schema":"a","schema":"b"}
{"schema":"v","binding":{"path":"a","path":"b"}}
{"schema":"v","binding":{"value":NaN}}
{"schema":"v","binding":{"value":Infinity}}
{"schema":"v","binding":{"value":-Infinity}}
{"schema":"v"}
{"extra":1}
```

The last fixture is a byte stream with two newline-separated JSON objects.
Also test wrong schemas, unknown root/nested fields, missing register gates,
register/approval gate or predecessor disagreement, and extra approval gates.

## 6. Exit

Retain the exact six-file implementation/test/vector scope above. Return exact
blobs, SHA-256 identities, complete test counts, and a clean diff. No
`.serena` deliverable, model-body access, `BASE_TRUE_TPS`, TG milestone,
capable-provider claim, HIDE promotion, MOP action, or authorization
transition.
