# What the semantic graph actually says

Built from `tools/graph/hawking_graph.py` over the tree at `6f3a1d2b`, after the repair
lane described at the end of this file. 25,069 nodes, 412,852 edges, 41s to extract,
byte-reproducible across runs. Agreement against the frozen measurement authority: LOC
exact, functions +3.4%, public symbols +2.8%.

Analyses run by `tools/graph/hawking_analyze.py` in 8.8s. Every number below is from the
repaired graph unless it is explicitly labelled as a before/after.

---

## The correction that has to come first

`HAWKING_RECOMPOSITION_CANDIDATES.json` ranks 91 candidates. Its top three, by a wide
margin, are:

```
RC-001  merge crate-level SCC of 18 crates          157,938 LOC
RC-002  merge file-level SCC of 464 files           133,287 LOC
RC-003  merge file-level SCC of 219 files            64,646 LOC
```

**All three are extraction artifacts and none of them is real.**

The first tell is that Cargo forbids dependency cycles, so an 18-crate strongly connected
component cannot exist in a workspace that builds — and this one does build. The cause is in
the extractor's own disclosure: unresolved calls are matched against the workspace by unique
name, and ambiguous matches are emitted at `confidence: 0.4`. Recomputing the components
with only high-confidence edges settles it:

```
edge set                          nodes    edges    SCCs>=2   largest
all calls + imports               14,217   85,267        75       411
confidence >= 0.8 only            12,063   24,269         4         7

(re-measured on the repaired graph below: 79 and 8 components, same largest of 7)
```

Sixty thousand of the 85,000 call edges — 71% — are the ambiguous kind. Drop them and the
"one giant cyclic core" disappears entirely: the tree's real strongly connected structure is
**eight components, the largest of which is seven nodes.**

That is a materially better architectural position than the artifact suggested, and it
changes the plan. There is no giant mutual-dependency blob to break apart. Boundaries can be
drawn where the design wants them.

It is also the campaign's own rule arriving on schedule: *graph tooling is an instrument, not
an authority*. Had the top three candidates been taken at face value they would have
authorised merging 355,871 lines' worth of "mutually dependent" code that is not mutually
dependent.

**Consequence for the candidate file:** RC-001, RC-002 and RC-003 are withdrawn. The
remaining 88 candidates are unaffected — they derive from fan-in, clone signature and
behaviour coverage, none of which uses the ambiguous call edges as evidence of a cycle.

---

## What the graph does say

### Communities — the folders are wrong, and by how much

157 communities over 1,386 files on the repaired graph. Sixteen span five or more
directories.

| community | files | LOC | dominant subsystem | dominant directory | directories spanned |
|---|---:|---:|---|---|---:|
| C-0000 | 228 | 114,389 | hawking | `crates/hawking-core/tests` | 21 |
| C-0001 | 133 | 45,929 | hawking | `crates/hawking-context/src` | 35 |
| C-0002 | 102 | 41,682 | hide | `crates/hide-backend/src` | 4 |
| C-0003 | 45 | 34,361 | laboratory | `tools/condense` | 15 |
| C-0004 | 64 | 25,108 | laboratory | `tools/bench` | 18 |
| C-0005 | 45 | 20,012 | hide | `crates/hide-kernel/src` | 6 |

C-0001 is the sharpest result in the whole analysis: 133 files that belong together by
coupling are spread across **35 directories**. C-0000's largest single home being
`crates/hawking-core/tests` says the same thing from the other side — the test tree is
where the runtime's real community is densest, which is what happens when tests are
organised by the code's layout rather than by the behaviour they prove.

C-0002 and C-0005, both HIDE, are the tidy ones at 4 and 6 directories. The prior arc's
HIDE consolidation is visible here and it held.

### Structural clones — there is no clone lever

400 families with two or more members, matched on control-flow signature rather than text.
The largest family is 24 members totalling **400 LOC**.

Combined with the prior arc's independent finding — 138 behavioural twins among 12,258
functions, 0.95% — the conclusion is consistent from two unrelated instruments: **this
codebase does not contain a large body of duplicated logic.** Whatever is going to be
removed will not be removed by deduplication.

### Fan-in — small, real, and worth taking

208 authorities carry four or more thin adapters, once the brace-match stubs that inflated
the first run are excluded. The largest ring is 63 adapters totalling 445 LOC — about seven
lines each, which is what a real thin forwarder looks like. The generate-bindings candidates
that follow are low risk and exactly what campaign section 7.2 describes. Worth taking, not
worth planning around.

### Betweenness and cuts

One broker-like node — high betweenness, low LOC, low complexity. 31 articulation points in
the largest component. 15 community-pair cut sets enumerated. The near-absence of brokers is
another way of saying the same thing the clone analysis said: this tree is not padded with
translation layers.

### Behaviour coverage — the number that matters

Joining the 210-behaviour constitution to the graph through `tools/graph/behaviour_map.py`,
under a deliberately generous depth-6 closure over `contains` and `calls`:

```
reachable from at least one behaviour contract     19,123 nodes
reachable from none                                54,590 LOC across 404 files
  laboratory    27,749
  shared        21,577
  hawking        3,531
  hide           1,733
```

The closure is generous on purpose — a behaviour naming `crates/hawking/src/main.rs` drags
in most of the runtime — so this **understates** the deletion candidates and the direction of
the error favours keeping code. The `shared` bucket is mostly root-level markdown. Twenty-six
of the 210 behaviours bind to no graph node at all, which is its own finding: those name
paths the graph does not have, and each one is either a stale path in the constitution or a
gap in the extractor.

The number moved from 22,634 to 54,590 when the imports repair landed, which is worth
stating plainly: **the first measurement was wrong in the direction of comfort.** With only
291 import edges the closure could not reach much, so more code looked covered than is. The
laboratory bucket in particular went from 525 to 27,749.

---

## The finding underneath all of them

The graph contains no queue of large, safe merges. The three biggest candidates were
artifacts; the clone families are tiny; the broker count is one; the fan-in rings are
hundreds of lines, not tens of thousands. Two independent instruments now agree that
duplicate logic is not where this tree's mass is.

**So the 183,505 lines between here and 250,000 cannot come from deduplication, and only
54,590 of them can come from deleting code no behaviour reaches.** They have
to come from re-expressing behaviour that is genuinely needed, in less code. That is exactly
the clean-room bet the campaign makes, and the graph's contribution is to have ruled out
every cheaper alternative before the building started.

What the graph does hand the rebuild is where to cut: 157 real communities, sixteen of them
scattered across five or more directories, and a strongly connected structure whose largest
genuine component is seven nodes. The boundaries are free to move.

---

## Instrument defects, three repaired and one outstanding

Three analyses produced nothing on the first run. A repair lane fixed all three; the numbers
above are from the repaired graph (25,069 nodes, 412,852 edges, byte-reproducible).

| defect | before | after |
|---|---|---|
| `imports` edges | 291 | **5,040** — Rust file-level 4,436 of ~4,472 `use` statements, Python 520 of 551 resolvable |
| dominator entry points | 0 of 252 | **253 of 268**; 80 nodes lie on the control path of two or more entries |
| co-change pairs above threshold | 0 | **59**; top pair `gravity_pq.metal` ↔ `gravity_glm_resident.rs`, 23 shared commits |

The co-change defect turned out to be in the analysis, not the schema: `weight` is defined as
`count / min(commits_a, commits_b)` and is therefore bounded by 1.0, while the analysis
thresholded at 5.0 — a bar nothing could clear. The threshold is now on raw co-commit count.

The fan-in "478 adapters at 1 LOC each" ring was, as suspected, brace-match stubs rather than
real forwarders. With `loc <= 1` stubs excluded the picture is 208 rings, largest 63 adapters
at 445 LOC — about 7 lines each, which is what a real thin adapter looks like.

**Still outstanding: cycle-derived analyses do not weight by confidence.** `analyze_scc` on
the repaired graph still reports a 473-file, 223,790-LOC component, and it is still an
artifact for the same reason as before. Re-measured on the repaired graph, now including
4,435 high-confidence Rust import edges:

```
edge set                       edges     SCCs>=2   largest
all calls + imports           91,574          79       411
confidence >= 0.8 only        29,594           8         7
```

The conclusion is unchanged and now rests on better data: 61,645 of the call edges are the
ambiguous `confidence: 0.4` kind, and without them the tree's real strongly connected
structure is eight components whose largest is seven nodes.

Until `analyze_scc` filters by confidence, **its output must not be read as evidence of a
cycle**, and any candidate derived from it is withdrawn on sight.
