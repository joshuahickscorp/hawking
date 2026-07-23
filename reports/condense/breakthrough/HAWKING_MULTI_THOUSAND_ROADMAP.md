# True batch-1 multi-thousand TPS: what the measured numbers allow

Evidence grade on every figure below. MEASURED means a run on this box. DERIVED means arithmetic over a
sealed measured ledger. PROJECTED means a configuration whose enabling artifact does not exist. No
projection is reported as a measurement, and the moonshot's own law forbids calling anything true batch-1
TPS until a complete model emits a complete token.

Generator: `tools/condense/hawking_tps_budget.py`. Ledger: `GLM52_ACTIVE_BYTE_LEDGER.json`,
validated delta-0 against 13,017 real tensors across 63 sealed shards.

## 1. There are two budgets, and the forgotten one binds first

A TPS target sets a traffic budget and a depth budget. The design must satisfy both.

**Depth budget (MEASURED).** A command buffer costs 215.8 us on this machine, fixed, independent of the
work inside it. That number alone decides which submission models are alive:

| target | ms/token | one command buffer costs | verdict |
|---:|---:|---:|---|
| 100 | 10.00 | 2.2% of the budget | SUBMISSION_NOT_BINDING |
| 250 | 4.00 | 5.4% | REQUIRES_FEW_COMMAND_BUFFERS |
| 500 | 2.00 | 10.8% | REQUIRES_FEW_COMMAND_BUFFERS |
| 1000 | 1.00 | 21.6% | REQUIRES_ONE_COMMAND_GRAPH |
| 2000 | 0.50 | 43.2% | REQUIRES_ONE_COMMAND_GRAPH |
| 5000 | 0.20 | **107.9%** | **IMPOSSIBLE_WITH_PER_TOKEN_SUBMISSION** |

Two consequences follow directly, and neither depends on kernel quality.

The current runtime submits one command buffer per layer. 78 layers times 215.8 us is 16.8 ms of pure
submission, so **the shipped submission model caps at 59.41 tok/s** (DERIVED) before a single weight byte
is read. Every kernel improvement below that ceiling is invisible at the token boundary.

At 5,000 TPS the whole token budget is smaller than one command buffer. **No runtime that submits work per
token reaches 5,000 TPS on this machine.** It requires a persistent GPU causal loop where the CPU only
consumes emitted tokens, which is Lane G's own moonshot rather than an optimization of it. This is a
STRUCTURAL wall in the audit's sense: it survives unlimited compute, and only a different runtime shape
retires it.

**Traffic budget (MEASURED bandwidth 736 GB/s, not the 819 GB/s vendor figure).**

| target | at 100% BW | at 70% | at 50% |
|---:|---:|---:|---:|
| 1000 | 736 MB/token | 515 MB | 368 MB |
| 2000 | 368 MB | 258 MB | 184 MB |
| 5000 | 147 MB | 103 MB | 74 MB |

The sealed artifact moves **5,010 MB/token** (MEASURED ledger). 1,000 TPS at 70% bandwidth therefore needs
a **9.7x** reduction in active bytes. No kernel change moves that number: it is what the artifact stores.

## 2. Where the token's bytes actually are

| organ | MB/token | share | stored |
|---|---:|---:|---:|
| routed_expert | 2,481 | 49.5% | 0.876 bpw |
| attention | 1,409 | 28.1% | 0.876 bpw |
| indexer | 394 | 7.9% | **16.000 bpw** |
| shared_expert | 310 | 6.2% | 0.876 bpw |
| router | 236 | 4.7% | **16.000 bpw** |
| lm_head | 104 | 2.1% | 0.875 bpw |
| dense_mlp | 74 | 1.5% | 0.875 bpw |

Effective active BPW over the whole token is 0.995 bits per active weight (DERIVED: 5,010,218,784 bytes
for 40,297,758,720 active weights). That is the number the moonshot's Lane B definition asks for, and it is
worse than the artifact's headline 0.876 because of the next line.

**632 MB/token, 12.6% of the token, is stored at source precision.** The indexer, router, normalization and
router_control tensors are `CONTROL_SENSITIVE_CANDIDATE` and the packer deliberately carries them
uncompressed. Bringing them into the compact codec is the largest byte win available that is not a retrain,
worth 1.14x on its own. It is a packing decision owned by the science stream, not a runtime change, and the
protection presumably exists for a reason that this campaign has not evaluated.

## 3. The ladder, and where it stops

Every row except the control carries the labels `PROJECTED`, `NOT_MEASURED`,
`ASSUMES_70_PERCENT_BANDWIDTH` and its submission assumption, in the JSON itself rather than only in this
prose, so a row cannot be quoted out of context as a throughput result. Rows whose enabling artifact does
not exist additionally carry `ENABLING_ARTIFACT_DOES_NOT_EXIST`.

| config | lane | MB/token | tok/s | binds | vs shipped | submission | open? |
|---|---|---:|---:|---|---:|---|---|
| shipped-today | control | 5,010 | 59.4 | DEPTH | 1.00x | 78 CB | MEASURED ledger |
| A-kernels-only | A | 5,010 | **59.4** | DEPTH | 1.00x | 78 CB | **OPEN, and moves nothing** |
| A+G-few-buffers | A+G | 5,010 | 102.8 | TRAFFIC | 1.00x | 3 CB | **OPEN** |
| A+G-one-graph | A+G | 5,010 | 102.8 | TRAFFIC | 1.00x | 1 graph | **OPEN** |
| B1-compress-protected | B | 4,413 | 116.8 | TRAFFIC | 1.14x | 1 graph | blocked: science stream owns the protection |
| D2-top2-experts | D | 2,552 | 201.9 | TRAFFIC | 1.96x | 1 graph | blocked: teacher capsules |
| D3-top1-experts | D | 2,242 | 229.8 | TRAFFIC | 2.23x | 1 graph | blocked: teacher capsules |
| E2-depth-20 | D+E | 732 | 704.0 | TRAFFIC | 6.85x | 1 graph | blocked: teacher capsules |
| E3-depth-10 | D+E | 418 | **1,232.7** | TRAFFIC | 11.99x | 1 graph | blocked: teacher capsules |
| F-native-functional | F | 323 | 1,593.3 | TRAFFIC | 15.49x | 1 graph | blocked: teacher capsules |

Reading the ladder honestly:

- **Lane A alone moves nothing.** `A-kernels-only` is the control for that question and it returns the
  shipped 59.4 tok/s, still DEPTH-bound. Kernels can get arbitrarily better and the token does not change,
  because submission binds first. This is the row that orders the work.
- **100 TPS is reachable on open lanes**, and only once submission collapses: three command buffers per
  token already gets there at 102.8, and the binding constraint flips from DEPTH to TRAFFIC on the way.
  That flip is the milestone to actually go and measure.
- **250, 500 and 1,000 TPS are not reachable on any open lane.** The cheapest configuration that reaches
  250 already requires depth collapse. Every one of them is blocked on the same input.
- **1,000 TPS is arithmetically reachable**, at E3: 78 stages collapsed to 10, top-8 routing collapsed to
  top-2, protected organs compressed. All three simultaneously.
- **2,000 and 5,000 TPS are reached by no configuration in this ladder**, including the native functional
  model at 323 MB/token. 2,000 TPS at 70% bandwidth allows 258 MB/token; the most aggressive projected
  student is 323 MB. Closing that gap needs a smaller model than any candidate currently described, and
  5,000 TPS additionally needs the persistent-GPU-loop runtime from section 1.

## 4. The single blocker under lanes D, E and F

Every configuration above 230 tok/s requires training a student against teacher state: post-attention
state, pre-router state, weighted MoE output, post-block trajectory, next-token logits. The teacher
capsules for GLM-5.2 are written to

    ~/Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/capsules

which is inside the live campaign root with a capture actively running. This campaign checked only that
the path exists and read nothing. So the blocker is **access and ownership, not difficulty**: teacher
capture and scientific representation selection belong to the live GLM stream by the campaign split, and
lanes D, E and F are that stream's work, not this one's.

What would unblock them, stated concretely so it can be handed over: a sealed, immutable snapshot of
teacher capsules for at least the layers whose experts are already packed (layers 3 to 15 have all 256
experts on disk), plus a decision from the science stream on whether the protected organs may be
compressed.

Until then this campaign's honest ceiling is Lane A plus Lane G, and its job is to reach 102.8 tok/s and
prove it with a complete token rather than to project past it.

## 5. What is measured, what is not

MEASURED on this box: 736 GB/s sustained read, 17,703 GFLOP/s fp32 FMA, 215.8 us command-buffer cost,
0.71 us marginal dispatch, the per-organ active-byte ledger, 0.87633 packed BPW, and the absence of
codebook sharing across experts.

UNMEASURED, and therefore zero by the run report's rule: any batch-1 TPS at all. **No complete token has
ever been executed through this runtime.** There is no forward pass, the router is absent from every
sealed shard, real activations were not read, and `PRODUCTION_EXECUTION_ADAPTER_REGISTRY` is empty. Every
tok/s figure in this document is a budget, not a result, and none of them may be quoted as throughput.

The next honest milestone is not 1,000. It is one token.
