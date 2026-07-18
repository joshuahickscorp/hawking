# The Appendix

The Appendix is the umbrella name for Hawking's additive compute-efficiency
work. It does not replace the condensation ladder, Doctor, Studio plans, or their
artifact hierarchy. The current ladder is the main corpus: its artifacts,
treatments, failures, receipts, and resource traces supply the models and evidence
that every Appendix experiment must use.

## One objective, five currencies

For a concrete workload and machine, inference time is bounded approximately by

\[
T \approx \max\left(
\frac{operations}{compute\ rate},
\frac{bytes\ moved}{memory\ bandwidth},
\frac{communication}{link\ bandwidth}
\right)
+ serial\ steps + queueing/coordination.
\]

The Appendix therefore manipulates five currencies:

1. arithmetic and useful FLOPs;
2. bytes moved and durable/resident state;
3. serial model or token dependencies;
4. CPU/GPU/device communication;
5. idle, launch, queueing, and coordination time.

The score is never nominal FLOPs or file bpw alone. It is quality-gated useful
output per second and per joule, with bytes moved, accepted/rejected work,
TTFT/TPOT, p50/p95/p99, pressure, and swap reported separately.

## Corpus rule

The active run is both the largest experiment and the largest Appendix corpus.
Appendix work may read its code and completed receipts, but must not stop it,
change queue state, hash bandwidth-heavy artifacts while it is active, or insert
new cells into its critical path.

After the run, build an immutable corpus index from:

- candidate and promoted artifacts plus SHA-256;
- exact ladder cell, bpw, codec knobs, Doctor treatment, and source commit;
- quality, acceptance, latency, byte, energy, pressure, swap, and thermal receipts;
- negative evidence and failure reasons, not only winners;
- prompt/workload/tokenizer hashes so cross-cell comparisons are real.

Every later Appendix receipt binds to that index. A result from an unrelated
FP16/GGUF parent does not prove behavior on the served TQ/Doctor distribution.

## Registry

| Appendix | Currency attacked | Current state | Canonical detail |
|---|---|---|---|
| A — corpus/compression | durable bits, quality per byte | active main run | existing ladder and Doctor documents |
| B — compute-for-memory | payload, metadata, codebook and scratch bytes versus integer ALU | four runtime modes scaffolded; device tests deferred | `tq_compute_for_memory_appendix_2026_07_14.md` |
| C — exact multi-token commit | serial target passes and repeated weight reads | TQ-native B=1..8 path and strict runner built; device parity deferred | `spec_decode_reentry_appendix_2026_07_14.md` |
| D — state and token avoidance | prompt tokens, KV bytes, repeated prefill | exact RAM/disk prefix cache shipped; KV precision active; semantic cache parked | `computational_efficiency_paradigms_2026_07_11.md` and active Bible |
| E — scheduling and physical execution | batching, launch/sync, occupancy, energy and thermals | partial Qwen/RWKV paths exist; Appendix receipt/rollup contract scaffolded, runtime sampling still deferred | this registry plus post-run receipts |
| F — architecture trained for deployment | model size, depth, experts, attention/state shape | research-only; never inferred as a runtime switch | `computational_efficiency_paradigms_2026_07_11.md` |

The lettering is an index, not a new sequence the main documents must adopt.

## Audit of the wider opportunity set

The attached 25-sector map is useful, but Hawking should not restart work that
already has evidence. The repo divides it into four buckets.

### Built or currently active

- Weight precision, TQ/STR2, RHT, outliers, and Doctor recovery are the main
  corpus.
- Exact RAM and disk prefix-state reuse are shipped; semantic-cache uplift is
  parked by its existing oracle.
- F16 and experimental INT4 KV paths, working-set capture, and state persistence
  already attack growing context state.
- Dense Qwen has batched prefill and multi-sequence scaffolding; RWKV has a real
  recurrent continuous-batch path.
- Q4_K predecode, fusion, buffer reuse, Metal command batching, and shape-specific
  kernels already cover much of the kernel/compiler sector.
- Energy, pressure, swap, and thermal probes exist, but are not yet one
  per-request capability ledger.
- MoE loaders/routing and expert paging exist, while several MoE execution ideas
  have explicit negative local evidence.

### Immediate Appendix build

1. Finish TQ runtime parity and byte-roofline receipts across stored, compact,
   hashed, computed, and layout-repacked lookup paths.
2. Device-prove the new TQ-native batched verifier and measure its B=1..8 curve
   before enabling any proposer.
3. Unify phase timing, bytes, joules, accepted/rejected work, pressure, and thermal
   state in one sampled ledger.
4. Measure Qwen continuous batching and chunked prefill as weight-reuse levers,
   not just kernel throughput. Sweep latency SLO as well as maximum throughput.
5. Add delayed exact shadow traces for model-free proposal arms and context-length
   strata for verifier/draft cost.
6. Treat prompt/RAG/output token budgets as a first-class baseline; a skipped
   token beats an optimized token, but quality and answer length must be held.

### Oracle-first, not implementation-first

- early exit/dynamic depth;
- activation-aware structured pruning or trained block sparsity;
- student distillation and small/large capability cascades;
- draft-model, parallel-head, block-diffusion, and block-iterative training;
- prefill/decode disaggregation, model parallelism, and unusual device placement;
- power caps and clock policies.

Each needs a corpus-bound oracle showing enough possible gain to pay its new
complexity. Architectural changes are not drop-in optimizations and belong after
the current dense TQ path is measurable.

### Do not casually resurrect

- Post-hoc block sparsity and static hot/cold MoE assumptions already failed on
  captured Hawking workloads.
- Semantic cache showed no median uplift beyond exact prefix reuse in its current
  proxy and stays parked until a better real corpus exists.
- A second conventional dense model is only a speculative control; dual residency
  and synchronization are charged.
- More GPUs/devices are not automatically faster; communication and conversion
  enter the same five-currency ledger.

## Cross-Appendix experiments

The largest gains can be compositions, so promotion happens on a factorial slice,
not isolated hero numbers:

- TQ runtime mode x batch size x context length;
- TQ runtime mode x exact proposer x draft length/tree shape;
- target precision x draft precision x acceptance-by-position;
- KV precision x context length x batch size;
- prefix/state hit rate x scheduler placement;
- power/thermal policy x accepted tokens/s and joules/token.

The matrix is pruned sequentially: cheap correctness and trace oracles first,
device microbench second, end-to-end composition last. A failed prerequisite
leaves dependent cells deferred, not falsely complete.

## Scaffolding

`tools/condense/appendix_scaffold.py` is the master, non-executing registry. It
prints deterministic experiment cells and can take a read-only snapshot of the
current corpus without opening or hashing large artifacts:

```sh
python3.12 tools/condense/appendix_scaffold.py --selftest
python3.12 tools/condense/appendix_scaffold.py --plan
python3.12 tools/condense/appendix_scaffold.py --status
```

The original 25-sector map is preserved without prose reinterpretation in:

```sh
python3.12 tools/condense/appendix_catalog.py --catalog
```

Cheap TQ geometry/byte data and its strict receipt can be rebuilt with:

```sh
python3.12 tools/condense/tq_runtime_probe.py \
  --write reports/appendix/tq_runtime_static_probe.json
python3.12 tools/condense/appendix_ledger.py \
  --wrap-static-probe reports/appendix/tq_runtime_static_probe.json \
  --output reports/appendix/tq_runtime_static_probe.receipt.json
python3.12 tools/condense/appendix_contract.py \
  --validate reports/appendix/tq_runtime_static_probe.receipt.json
python3.12 tools/condense/tq_runtime_matrix.py \
  --write reports/appendix/tq_runtime_device_matrix.json
python3.12 tools/condense/appendix_postrun.py \
  --write reports/appendix/appendix_postrun_plan.json
```

`appendix_postrun.py` is the two-ply bridge from the corpus into hardware/runtime
work and then speculative decode. It maps six existing compile/vendor gates and
the two artifact-bound Hawking adapters. Vendor gates remain candidate selectors;
only the strict Hawking runners can finalize receipts. `appendix_device_runner.py`
binds a real `.tq` tensor to decode/GEMV/counter evidence.
`spec_tq_runner.py` binds a full all-linear TQ Qwen target to non-skipping B=1..8
parity and cost-curve evidence. Both acquire the canonical heavy lease and recheck
owners. Hawking compiles Metal source through the runtime driver, so the absent
offline `xcrun metal` utility is diagnostic rather than an execution blocker.
`--status` exits 75 while owners remain or a required release probe is absent.

Post-run release probes are built explicitly; this command only compiles them and
does not open Metal or a model:

```sh
cargo build --release -p hawking --features tq \
  --bin hawking-tq-device-probe --bin hawking-tq-spec-probe
```

For seamless incorporation in another session, start at
`docs/plans/APPENDIX_HANDOFF.md` and run
`python3.12 tools/condense/appendix_handoff.py --audit`.

The corpus indexer is implemented but hard-interlocked against the live run:

```sh
# Safe now: names/counts only.
python3.12 tools/condense/appendix_corpus.py --preview

# Post-run only: refuses with exit 75 while any heavy owner exists.
python3.12 tools/condense/appendix_corpus.py \
  --build reports/appendix/corpus_index.json
python3.12 tools/condense/appendix_corpus.py \
  --verify reports/appendix/corpus_index.json
```

The stable hash checks a file before and after streaming it and aborts if its
identity, size, or modification time changes. Partial and negative evidence are
indexed alongside winners. Verification also enumerates the frozen tree and
rejects any file added after the index, so the binding is a complete snapshot
rather than a partial allow-list.

Speculative matrix detail remains in `spec_reentry_scaffold.py`, which still has
no execute command. P0/P1 execution lives only in `spec_tq_runner.py`; P2-P6 stay
deferred until their own artifact-bound runners and prerequisites exist.
