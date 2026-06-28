# 05 · Codebase Intelligence — the Living Index

> **Purpose (one line):** A 24/7, crash-safe, incrementally-maintained, multi-layer model of *all* your code — parse trees, a symbol/reference graph, call/import/type/test/perf graphs, an optional dataflow CPG, and a lexical-first semantic index — kept always-fresh by a background daemon on idle Apple-Silicon GPU, so the agent (ch.02), the tools (ch.03), and the Context Compiler (ch.04) can ask precise structural questions over a million lines in milliseconds, forever, offline.

This is the subsystem the cloud literally cannot replicate. A SaaS coding assistant pays per token and runs on shared, ephemeral infrastructure; it cannot afford to keep a living, whole-history, cross-repo graph of *your* machine's code warm at all times. Hawking owns the whole stack, has free persistent local compute, and free private embeddings — so the index is not a request-time RAG bolt-on, it is a **standing organ** of the IDE.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art & its limits (cited)](#3-state-of-the-art--its-limits-cited)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [System map & module layout](#41-system-map--module-layout)
   - 4.2 [The parsing layer (tree-sitter, incremental, error-tolerant)](#42-the-parsing-layer)
   - 4.3 [The symbol graph (defs/refs/scopes — SCIP + stack-graphs hybrid)](#43-the-symbol-graph)
   - 4.4 [Call, import, type, test & perf graphs](#44-call-import-type-test--perf-graphs)
   - 4.5 [The optional code-property graph (dataflow/security)](#45-the-optional-code-property-graph)
   - 4.6 [The repo-map ranking algorithm (PageRank + signals, token-budgeted)](#46-the-repo-map-ranking-algorithm)
   - 4.7 [The semantic index (chunking, embeddings, lexical-first + re-rank)](#47-the-semantic-index)
   - 4.8 [Change detection (BLAKE3 merkle-DAG)](#48-change-detection-blake3-merkle-dag)
   - 4.9 [The Living-Index daemon (watch, debounce, idle reindex, crash-safe)](#49-the-living-index-daemon)
   - 4.10 [Storage architecture & schemas](#410-storage-architecture--schemas)
   - 4.11 [The query API surface](#411-the-query-api-surface)
   - 4.12 [Multi-repo, monorepo & million-line scaling](#412-multi-repo-monorepo--million-line-scaling)
5. [How we exceed cloud (the moat)](#5-how-we-exceed-cloud-the-moat)
6. [Failure modes & mitigations](#6-failure-modes--mitigations)
7. [Extensibility & plugin points](#7-extensibility--plugin-points)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions & dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)

---

## 1. Purpose & scope

The Living Index answers, with low latency and high precision, the structural questions an agent needs to act on a codebase:

- *Where is `X` defined? Who references it? Who calls it? What does it call?*
- *What implements this trait/interface? What overrides this method? What's the type of this expression?*
- *What is the shortest dependency path from module A to module B, and why does it exist?*
- *What tests cover symbol `X`? What changed since commit `Y`? What's the hottest function on the decode path?*
- *Given a natural-language task, which ~40 spans of code are most relevant — ranked, deduplicated, and fit to a token budget?*

It is consumed by:

- **ch.04 Context Compiler** — the single biggest consumer. The repo-map (§4.6) and the hybrid retriever (§4.7) are the Context Compiler's structural and semantic legs. This chapter *produces*; ch.04 *budgets and assembles*.
- **ch.03 Tools** — `find_definition`, `find_references`, `find_callers`, `find_implementations`, `path_between`, `tests_covering`, `changed_since`, `grep_symbol` are thin wrappers over the query API (§4.11). Tools should never re-parse; they query the index.
- **ch.02 Agent** — the agent's planning loop uses the index to scope edits, predict blast radius (reverse call graph), and verify it touched everything it needed to.

**Scope (strict, per the bible's scoping rule):**

- **Shell-first.** This chapter specifies the *shell*: the daemon, the schemas, the graph model, the query API, the ranking algorithm, the incremental-update machinery. These are the load-bearing, stable surfaces other chapters bind to.
- **The model layer is a stable localhost surface.** We consume `hawking-serve` over HTTP (`/v1/embeddings`, future `/v1/embeddings?role=code`). We do *not* design the model internals here. The runtime ground truth (below) constrains the design but the embedding model is swappable behind the HTTP seam.
- **HF / `.tq` / 32B are runtime-testing concerns, not shell-gating.** The index must work with the logits-proxy embedding that exists today and improve monotonically when a dedicated embedding role lands. Nothing in the shell blocks on a particular model.

### Runtime ground truth (verified against the repo, 2026-06-24)

This design is anchored to what `hawking-serve` actually exposes today, read from source:

- **`POST /v1/embeddings` exists** (`crates/hawking-serve/src/http.rs:166`, handler at `:939`). It accepts `input` as a single string or a batch, `encoding_format` must be `"float"`, returns OpenAI-shaped `{object:"list", data:[{embedding:[...]}], model, usage}`.
- **The current `embed()` is a logits/hidden-state proxy**, not a trained sentence encoder. From `crates/hawking-core/src/engine.rs:550`:
  ```rust
  fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
      let ids = self.encode_prompt_for_batch(text)?;
      let positions: Vec<usize> = (0..ids.len()).collect();
      let rows = self.forward_tokens_for_test(&ids, &positions)?;
      let last = rows.last()...;                 // last-token hidden state
      let norm = (Σ last_iᵢ²).sqrt().max(1e-8);
      Ok(last.iter().map(|v| v / norm).collect()) // L2-normalized
  }
  ```
  It runs a forward pass and L2-normalizes the **last token's hidden state**. This is a decoder-LM representation, **explicitly "not ideal for semantic search"**: it is dominated by next-token prediction signal, has no contrastive training objective, and the *last-token* pooling discards most of the sequence. **This is the single most important design constraint in this chapter.**

**Design consequence — lexical/symbol FIRST, embeddings as a re-ranker.** Because the embedding is a weak proxy *today*, the retriever's primary recall legs are **exact symbol resolution (the graph) + lexical search (BM25/trigram)**. Embeddings contribute a *re-ranking / semantic-expansion* signal, never the sole recall path. This also happens to be where the industry converged for code regardless of embedding quality (Sourcegraph, Anthropic, Augment — see §3). We design for a **future dedicated embedding role** (`/v1/embeddings?role=code`, a contrastively-trained code encoder served as a second model role) that drops in behind the same HTTP seam and is allowed to graduate from "re-rank only" to "first-class recall leg" via a config dial (§4.7, §9) — **without any schema change**, because we version the embedding model and store `model_id` + `dim` per vector.

- **Apple-Silicon / Metal / Rust workspace.** `hawking-core` (the engine + Metal kernels), `hawking-serve` (the HTTP surface), `hawking` (CLI), `hawking-bench`. The index lives in a **new crate, `hawking-index`** (§4.1), so it doesn't bloat the hot inference path and can be tested independently.
- **The front-end is Monaco-based.** Monaco gives us a `monaco.editor.IModel` with version-stamped change events (`onDidChangeModelContent`) and an `IModel.uri`. We tap these for *editor-buffer* freshness (the open file, before it's saved to disk) and to render results (go-to-def, references, the symbol outline).

---

## 2. Tenets

1. **Lexical & symbol truth first; embeddings re-rank.** Exact identifiers and structural resolution beat fuzzy vectors for code. Never let a weak embedding gate recall. (Forced by ground truth; vindicated by SOTA.)
2. **Always-fresh, never blocking.** Indexing is a background organ. A keystroke is never slowed by the index. Freshness is *eventual but fast* (sub-second for the edited file via the editor-buffer fast path; seconds for graph propagation).
3. **Incremental everywhere; O(changed), not O(repo).** Every layer — parse, symbol graph, vectors, repo-map — updates proportional to what changed, gated by a BLAKE3 merkle-DAG. Re-indexing the whole repo is the exceptional path (cold start / corruption recovery / model upgrade), run on idle GPU.
4. **Crash-safe by construction.** Append-only immutable segments + atomic manifest swap + monotonic generation counter. A crash at any instant loses at most the last uncommitted batch and recovers to the last good generation by truncating a torn tail. Never a corrupt index.
5. **Snapshot-consistent reads (MVCC).** A reader pins a generation and sees a coherent point-in-time view while the writer races ahead. No partial-update tears in query results.
6. **One writer, many readers.** A single background indexer is the sole writer (natural fit for SQLite WAL + segment manifests). Tools/agent/UI are readers.
7. **Determinism.** Same inputs → byte-identical index artifacts (modulo timestamps, which are segregated). Hashing is content-addressed; ordering is canonical. This is a Hawking family value (cf. Condense) and makes caching, diffing, and bug-repro trivial.
8. **Degrade gracefully, never silently.** Parse errors localize (tree-sitter ERROR nodes); huge/generated files get budgeted or skipped *with a recorded reason*; a dead language server falls back to tags. Every degradation is observable in the index's health surface.
9. **Whole-history, whole-machine.** Index everything: all repos the user opens, optionally all git history (blame/age signals, "what changed since Y" across the whole DAG), generated artifacts (flagged). The cloud cannot; we can.
10. **The model layer is a stable localhost surface.** We bind to HTTP, version the model, and never couple the index schema to a specific embedding model.

---

## 3. State of the art & its limits (cited)

A dense survey of what exists and where it breaks, with **difficulty** (effort to adopt/build) and **impact** (leverage for HIDE) tags, and **PROVEN vs SPECULATIVE** for the specific Hawking application.

### 3.1 Parsing — tree-sitter

**What it is.** A GLR incremental parser generator (Max Brunsfeld / GitHub) producing concrete syntax trees. Design goals verbatim: parse *any* language, fast enough to **parse on every keystroke**, robust to syntax errors, dependency-free C runtime.

**The data model / algorithm that matters:**
- **Incremental reparse:** keep the old tree, call `tree.edit(InputEdit{ start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point })` (byte **and** row/col on both axes), then `parser.parse(new_src, Some(&old_tree))`. The new tree **structurally shares** unchanged subtrees; cost ∝ size of the change. Interior nodes cache pre-`goto` LR state so whole subtrees are skipped. Reparse on a keystroke is sub-millisecond; full parse of a large file is tens of ms.
- **Error recovery:** unparseable spans are wrapped in `(ERROR)` nodes (damage localized, rest of file is real and queryable) and zero-width `MISSING` nodes are inserted to recover (absent `;`/`}`); `MISSING` is *not* matched by an `(ERROR)` query — query `is_missing` separately.
- **Query API:** S-expression patterns (`(function_definition name: (identifier) @name)`), captures `@name`, predicates (`#eq?`, `#match?`, `#any-of?`). The C core does *structural* matching only; **predicate evaluation lives in the host binding** — budget for it in Rust. `QueryCursor::set_byte_range` scopes a query to a window (the viewport).
- **`tags.scm`:** per-grammar convention capturing `@definition.{class,function,method,…}` and `@reference.{call,…}` plus `@name`/`@doc`. This is the cheap, language-agnostic, **approximate** (name-matched, no types) def/ref extractor that powers GitHub's search-based nav and Aider's repo-map.

**Who uses it:** Aider (repo-map), Zed (highlight, structural nav, copy-on-write trees off-thread), Neovim/Helix (highlight, textobjects, structural motions), GitHub semantic (jump-to-def/find-refs on push). Cursor reportedly uses it for structure (secondhand).

**Limits:** tags are *approximate* — no type resolution, can't disambiguate overloads or follow imports. The `rust` binding marshals predicates host-side. Cached `TSNode` handles go stale after an edit. Grammars vary in quality; some (C++) emit only defs in `tags.scm` (Aider backfills refs via Pygments — we backfill via the LSP or a `locals.scm` pass).

**Verdict for HIDE:** **PROVEN. Difficulty LOW–MED, Impact HIGH.** This is the bedrock parse layer. Non-negotiable.

### 3.2 Repo-map ranking — Aider

**What it is (exact algorithm, from `aider/repomap.py@main` + Gauthier's 2023 writeup):**
1. Extract `Tag(rel_fname, fname, line, name, kind∈{def,ref})` per file via tree-sitter `tags.scm` (Pygments ref-backfill for def-only languages).
2. Build three dicts: `defines: ident→set(file)`, `references: ident→list(file)` (list → multiplicity), `definitions: (file,ident)→set(Tag)`.
3. Build `nx.MultiDiGraph`, **nodes = files**, a multi-edge per `ident` **from referencer → definer**, `weight = use_mul · sqrt(num_refs)`. Weight multipliers (the secret sauce): `ident ∈ mentioned_idents → ×10`; distinctive long multiword name (snake/kebab/camel & len≥8) `→ ×10`; `_private → ×0.1`; defined in >5 files (generic) `→ ×0.1`; referencer is a chat file `→ use_mul ×50`. `sqrt(num_refs)` damps so one file can't dominate.
4. **Personalized PageRank** (`alpha=0.85`): personalization mass `100/len(files)` accrued to chat files, mentioned files, and files whose path components match `mentioned_idents`.
5. **Distribute each node's rank across its out-edges by symbol**, crediting `(definer, ident)` pairs. **The ranked unit is a `(file, identifier)` definition, not a file** — the most-misdescribed step.
6. **Binary-search** how many top definitions fit `map_tokens` (≈25 tok/tag seed, 15% tolerance), render as an **elided signatures-only tree** (`grep_ast.TreeContext`, bodies → `⋮`, lines capped 100 chars).

**Limits:** approximate (tag-based, no types); `networkx` PageRank is Python and not incremental; the elided rendering needs per-language scope queries. Personalization is the whole game — wrong signals → wrong map.

**Verdict for HIDE:** **PROVEN concept, our impl is a port + upgrade. Difficulty MED–HIGH, Impact HIGH.** We re-implement in Rust over our *persistent* graph (so PageRank runs on a maintained graph, not rebuilt per query), add **open-tabs / cursor-proximity / recency / git-age** signals on top of Aider's, and feed the result to ch.04. See §4.6.

### 3.3 Symbol indexing — SCIP, LSIF, stack-graphs, ctags

| System | Incremental? | Cross-file/precise? | Cost | Resolution |
|---|---|---|---|---|
| **SCIP** (Sourcegraph protobuf) | Batch-produced but self-contained per-doc → cheap incremental | **Yes, precise** (structured global *string* symbol IDs) | MED–HIGH (semantic indexer) | **String-equality lookup** (no traversal) |
| **LSIF** (Microsoft) | **Batch only** (opaque numeric IDs impose ordering → blocks fine-grained update) | Yes, precise (monikers + edges) | HIGH (verbose JSON, large in-mem) | Graph edge traversal |
| **stack-graphs** (GitHub) | **Incremental per-file** (isolated partial graphs, stitched at query time) | Yes at query; name-binding precise, imprecise vs a compiler | MED (tree-sitter + TSG, no build tools) | Path-finding with symbol/scope stacks |
| **universal-ctags** | Per-file regen, no stitching | **No** (textual name match) | VERY LOW | Binary search on sorted names |

**SCIP detail (we adopt its *storage/consumption* model):** `Index{Document[]}`, `Document{Occurrence[], SymbolInformation[]}`, `Occurrence{range (packed int32), symbol (string), symbol_roles (bitset: Definition=0x1, Import=0x2, WriteAccess=0x4, ReadAccess=0x8, Generated=0x10, Test=0x20, …)}`, `SymbolInformation{symbol, kind, Relationship[]}`, `Relationship{symbol, is_reference, is_implementation, is_type_definition}`. **Symbols are structured global strings** (`<scheme> <manager> <pkg> <version> <descriptors>` with `/` namespace, `#` type, `.` term, `().` method, `!` macro) → go-to-def and find-refs are **string-equality lookups over precomputed occurrences**, O(1)-ish, no graph walk. SCIP is ~4–5× smaller and ~3× faster to process than LSIF.

**stack-graphs detail (we adopt its *incremental production* model):** per-file **partial graphs** built independently from the tree-sitter CST via tree-sitter-graph (`.tsg` rules) with definition/reference/scope nodes; name resolution is **path-finding under a pushdown discipline** (a reference pushes a symbol, a definition pops it; you can't traverse a pop node unless its symbol matches the stack top). **Critical incrementality property (verbatim): "For each source file, we create an isolated subgraph without any knowledge of … any other file."** The full graph is the *union* of partials; cross-file resolution happens at query time by stitching partial paths. **A changed file recomputes only that file's partial graph** — exactly why it scales to an always-editing monorepo, unlike LSIF's batch reindex. Limit: name-binding only, *not* a type checker; per-language correctness depends on hand-written `.tsg`.

**Verdict for HIDE:** **PROVEN.** We use a **hybrid**: SCIP-shaped *storage* (string symbol IDs → O(1) nav, agent-readable) populated by a **stack-graphs-style incremental producer** for languages with a `.tsg`, falling back to tree-sitter `tags.scm` + LSP for the rest. ctags is the cold-start / exotic-language fallback. LSIF is import-only interop. **Difficulty MED–HIGH, Impact HIGH.**

### 3.4 Code-property graphs — Joern

CPG = AST ∪ CFG ∪ PDG (control-dependence + data-dependence) in one typed multigraph (Yamaguchi et al., IEEE S&P 2014). Enables **taint/dataflow/security** queries: `sink.reachableBy(source)`. Call edges pre-generated at build time (slow build, fast interprocedural query). **Cost is far higher than AST/tags** (full semantic + CFG + dependence + resolved calls, RAM-heavy, slow rebuild). Storage moved to columnar `flatgraph` (Joern 4.x, ~40% less RAM).

**Verdict for HIDE:** **PROVEN but scoped. Difficulty HIGH, Impact MED (HIGH for security tasks).** A CPG is *not* for keystroke-latency nav. We expose it as an **optional, on-demand, per-target overlay** (§4.5) built lazily for a function/file when the agent asks a dataflow/security question — never maintained repo-wide in the hot loop.

### 3.5 LSP for type info

Pull-model JSON-RPC; harvest `hover` (resolved type + doc), `definition`/`typeDefinition`/`implementation`, `references`, `documentSymbol` (per-file outline tree), `workspace/symbol`, `semanticTokens` (+`/delta` incremental), and prepare→resolve `callHierarchy`/`typeHierarchy`. Run real servers headless over stdio (rust-analyzer, gopls, pyright, clangd, tsserver), e.g. via `multilspy`'s subprocess/handshake abstraction. **Limits:** warm-up latency (servers return incomplete answers until indexed — the pull model silently reflects a half-built index), no bulk dump (N round-trips), one process per language (fleet lifecycle), per-language capability gaps, memory-heavy.

**Verdict for HIDE:** **PROVEN as the precise *type* source. Difficulty MED–HIGH, Impact HIGH.** LSP is the **type graph** producer (§4.4) and the precision upgrade over tree-sitter tags. We treat it as an *enrichment* layer that lands asynchronously — the index is useful from tags alone and gets *more precise* as LSP answers arrive.

### 3.6 Semantic code search & embeddings (2024–2026)

- **Chunking:** AST/symbol-aware beats fixed line-windows. **cAST** (EMNLP-Findings 2025): recursive split-then-merge over the tree-sitter AST (emit a node whole if it fits, split oversized, greedily merge small siblings) → +2.7–5.6 pts on RepoEval/SWE-bench/CrossCodeEval vs fixed chunks. Continue.dev does the same (collapse oversized functions to a signature with `{ … }`, recurse).
- **Models:** code-specialized embedders lead on CoIR (Qodo-Embed-1-7B 71.5, SFR/CodeXEmbed-7B 70.46, CodeSage-v2 64.2) vs generic (OpenAI v3-large ~65). nomic-embed-code / voyage-code-3 lead CodeSearchNet by language. *(Benchmarks differ — CoIR ≠ CSN ≠ MTEB; never cross-compare.)*
- **Hybrid + fusion:** BM25/sparse (exact identifiers, rare API names — matters **more** for code) + dense ANN, fused via **Reciprocal Rank Fusion** `RRF(d)=Σ 1/(k+rankᵣ(d))`, **k=60** (Cormack 2009; Elasticsearch/Weaviate default). Then a **cross-encoder re-ranker** over top-N (`~25→5`) — the biggest precision lever.
- **The key empirical finding for code:** *lexical/symbol retrieval often beats or matches **generic** embeddings.* Vendors that shipped on embeddings then **removed** them: **Sourcegraph Cody** ("we're leaving [embeddings] behind … replaced with Sourcegraph Search" — BM25-adapted ranking + query-understanding), **Claude Code** (Boris Cherny: "agentic search generally works better … fewer issues around security, privacy, staleness, reliability"), **Augment** ("'grep' and 'find' were sufficient" for SWE-bench). Counter-evidence (carry honestly): **CodeRAG-Bench** — *code-specialized* dense embedders beat BM25 and generic dense. So the blur is a *generic-embedder* problem; a good code encoder adds real value. **This directly validates our ground-truth-forced design:** lexical/symbol first; embeddings re-rank now, graduate later when the code role lands.
- **Products:** **Cursor** = merkle tree + server-side embeddings (Turbopuffer, obfuscated paths, chunk-hash cache → unchanged chunks near-free). **Continue.dev** = LanceDB vectors + SQLite FTS5 keyword + tree-sitter symbols, `nRetrieve=25 → rerank → nFinal=5`. **Sourcegraph** = keyword/BM25 + query-understanding.

**Verdict for HIDE:** **PROVEN architecture (hybrid + RRF + rerank + AST chunking). Difficulty MED, Impact HIGH.** Our twist: the *re-rank* stage is also where our weak local embedding lives until upgraded, and re-ranking can additionally use a **local LLM listwise pass** (free, private — §4.7, §8).

### 3.7 Change detection — BLAKE3 merkle-DAG

**BLAKE3** is internally a merkle tree over 1 KiB chunks → parallel/incremental hashing + verified streaming for free; XOF (any output length); SSE/AVX/**NEON** with runtime detection. On **Apple M3 ≈ 4.1 GB/s** single-thread NEON, **~5–8× faster than software SHA-256** (M-series lack SHA-NI, so SHA-256 falls to ~500–800 MB/s) — decisive on our exact hardware. Mature `blake3` Rust crate. Mesa adopted it for shader-cache hashing (direct analog).

**Merkle directory hashing:** leaf = hash(file bytes / serialized AST); directory node = hash of **sorted** child entries `(name, type, child-hash)` → root changes **iff** anything changed; **O(changed) tree diff** (compare roots; if equal stop; else recurse only into differing subtrees). This is Git's tree object / ostree / Nix NAR. **Verified streaming (bao / bao-tree):** fetch+verify an arbitrary byte sub-range of a large packed artifact against the root via a merkle proof, without reading the whole file.

**Verdict for HIDE:** **PROVEN — the single biggest incremental-indexing win. Difficulty LOW–MED, Impact HIGH.** §4.8.

### 3.8 Storage & graph engines

- **Vector (local, on-disk, churning 1M chunks on Apple Silicon):** the deciding axes are *deletion under churn*, *disk vs RAM residency*, *background-amortizable build*, *NEON*, *Rust embeddability*. **LanceDB/Lance** (Rust-native columnar, mmap, IVF-PQ/SQ, soft-delete + cheap append + background `optimize()`) is the best end-to-end fit (PROVEN, Difficulty LOW–MED, Impact HIGH). **usearch** (mmap'd HNSW + genuine NEON via SimSIMD + i8/binary quant; caveat: tombstone deletes degrade the graph under sustained churn) is the alternate. **sqlite-vec** (brute-force, but *trivially exact free deletes*, transactional) is the durable co-store / small-shard search path. **Avoid** HNSW-only libs that force tombstone-then-full-rebuild (hnswlib-rs has no NEON + RAM-resident graph; instant-distance is immutable; qdrant has no embedded Rust lib mode, 20K cap). **DiskANN/FreshDiskANN** is the *only* family designed for streaming insert+delete (PQ-in-RAM, disk graph, lazy delete + consolidation) but the Rust ports are young (Difficulty MED–HIGH).
- **Lexical/symbol store:** **SQLite** with WAL (single writer, concurrent readers, snapshot reads), **FTS5 + BM25** (per-column weights: identifier ≫ body), **`trigram` tokenizer** (substring/identifier search — `getUserId` via `serId`; 3-char min, so keep `unicode61` alongside for short symbols; measured >100× over `LIKE` scans), recursive CTEs for transitive closure (UNION to break cycles, index both edge columns, depth-cap). PRAGMAs: `journal_mode=WAL; synchronous=NORMAL; busy_timeout=5000; cache_size=-262144; mmap_size=256MiB; temp_store=MEMORY`. **Batched transactions are the dominant write lever (~600× over autocommit).**
- **Graph queries:** persist edges in SQLite (with **materialized reverse edges** — non-negotiable so "find all callers" is an index seek), load into **petgraph/CSR in-memory** for BFS/SCC/bidirectional-BFS at native speed (sidesteps recursive-CTE pain), materialize a **transitive-closure table** only for hot all-transitive queries. **CozoDB** (embeddable Rust Datalog, parallel Horn-clause recursion, built-in PageRank/shortest-path) is a strong option if we want recursion + incremental materialization pushed into the store. (Kùzu is technically ideal — columnar CSR, Cypher Kleene paths — but **abandoned Oct 2025**; mine its design, don't depend on it.)
- **Incremental-update spine:** one **monotonic generation counter** + **append-only immutable segments** + **atomic manifest swap** (temp → fsync → rename → **fsync the directory fd**) + tombstone deletes applied at read, reclaimed on throttled background merge + refcounted/pinned generations for reader MVCC. **Tantivy** (Rust Lucene-like) already implements this exact pattern (immutable UUID segments, opstamp sequence, `meta.json` atomic swap, `.del` tombstone bitsets, throttled merges, point-in-time `Searcher`); it's our model and a candidate dependency for the lexical leg.
- **CAS:** clone Git's object model (header `<type> <size>\0` + content, 2-hex fan-out dirs, tree-as-sorted-entries, loose→pack with bounded deltas, mark-and-sweep GC) but swap SHA-1 → **BLAKE3**.

**Verdict:** **PROVEN building blocks. Difficulty LOW–MED per component, Impact HIGH.** §4.10.

### 3.9 File-watching — Rust `notify`

`RecommendedWatcher` → **FSEvents** on macOS (path/dir-granular, coalesces). Always go through **notify-debouncer-full** (dedup + `FileIdMap` rename-stitching, single Rename event), window ~200ms–2s. **Renames are the hard part**: not atomic on most platforms (Windows = two events; many platforms = remove+create; move-out = bare remove; inotify cookie is racy). **Atomic-save-via-rename** (vim/Sublime/Kate/`sed -i`: write `.tmp` then rename over target) mis-reads as "deleted" by naive watchers. **Watch the parent directory** (stable inode), not the file (its watch dies on rename). **Reconcile against the merkle/content hash — the watcher is a *hint*, not truth** (absorbs duplicate/racy/dropped events, `IN_Q_OVERFLOW`, FSEvents history loss). Respect `.gitignore` via the `ignore` crate (ripgrep's), re-checking each event path. On macOS, fall back to `PollWatcher` for un-owned files.

**Verdict:** **PROVEN. Difficulty LOW–MED, Impact HIGH.** §4.9.

---

## 4. The Hawking design (concrete)

### 4.1 System map & module layout

The index is a **new crate, `hawking-index`**, plus a long-lived daemon binary `hawking-indexd` and a thin client used by tools/agent/UI. It depends on `hawking-core` only for the embedding HTTP client surface (it talks to `hawking-serve` over localhost), never the inference hot path.

```
crates/hawking-index/
├── src/
│   ├── lib.rs                 # public query API (the surface ch.02/03/04 bind to)
│   ├── daemon/
│   │   ├── mod.rs             # IndexDaemon: lifecycle, supervisor
│   │   ├── watcher.rs         # notify + debouncer-full; .gitignore via `ignore`
│   │   ├── scheduler.rs       # priority queues (editor > save > idle-reindex); idle/GPU detection
│   │   └── health.rs          # freshness, lag, error surface; /index/health
│   ├── merkle/
│   │   ├── mod.rs             # BLAKE3 merkle-DAG over the workspace tree
│   │   └── diff.rs            # O(changed) tree diff → changeset
│   ├── parse/
│   │   ├── mod.rs             # tree-sitter manager, grammar registry, error-tolerant reparse
│   │   ├── grammars.rs        # per-language Language + tags.scm + locals.scm + .tsg + chunk.scm
│   │   └── chunker.rs         # cAST-style AST-aware chunking (by symbol, signature-collapse)
│   ├── symbols/
│   │   ├── mod.rs             # SCIP-shaped occurrence/symbol model
│   │   ├── stackgraph.rs      # per-file partial scope-graphs, stitched at query
│   │   ├── lsp.rs             # headless LSP fleet: hover/def/refs/impl/callhierarchy/typehierarchy
│   │   └── resolve.rs         # name resolution: stack-graph path-find → LSP precision overlay
│   ├── graphs/
│   │   ├── mod.rs             # node/edge model; petgraph load; closure materialization
│   │   ├── callgraph.rs       # call edges (tags + LSP callHierarchy + CPG when present)
│   │   ├── importgraph.rs     # module/import/dependency edges
│   │   ├── typegraph.rs       # subtype/impl/override edges (LSP typeHierarchy)
│   │   ├── testmap.rs         # test → covered-symbol edges (heuristic + coverage import)
│   │   ├── perfmap.rs         # symbol → measured cost edges (profiler/trace ingestion)
│   │   └── cpg.rs             # OPTIONAL on-demand AST∪CFG∪PDG overlay (Joern-style)
│   ├── repomap/
│   │   ├── mod.rs             # persistent personalized PageRank over the reference graph
│   │   └── render.rs          # token-budgeted elided signatures-only tree (ch.04 feed)
│   ├── semantic/
│   │   ├── mod.rs             # hybrid retriever: lexical (FTS5) ⊕ symbol ⊕ vector → RRF → rerank
│   │   ├── embed.rs           # /v1/embeddings client; role-aware; model+dim versioning; cache
│   │   └── rerank.rs          # cross-encoder / local-LLM listwise re-rank
│   └── store/
│       ├── sqlite.rs          # WAL, FTS5 (trigram+unicode61), edges + reverse edges, CTEs
│       ├── vectors.rs         # Lance (primary) | usearch (alt) | sqlite-vec (co-store)
│       ├── cas.rs             # BLAKE3 content-addressed blob store (Git-model, 2-hex fanout)
│       ├── segments.rs        # immutable segments + manifest + generation counter (MVCC)
│       └── catalog.rs         # files/symbols/generation catalog; per-repo shards
└── bins/
    └── hawking-indexd.rs      # the daemon process

crates/hawking/ (CLI)            adds: `hawking index {status,reindex,query,explain}`
crates/hawking-serve/ (HTTP)     adds (future): /v1/embeddings?role=code  (dedicated code encoder role)
```

**Process model.** `hawking-indexd` runs as a user-level daemon (launchd agent on macOS), one per machine, multiplexing all open workspaces. It is the **sole writer**. Tools/agent/UI link `hawking-index` as a library for **read** queries against the same on-disk store (SQLite WAL + Lance + segments support concurrent readers), and talk to the daemon over a local Unix-domain socket for *control* (open/close workspace, force reindex, subscribe to freshness events). The Monaco front-end subscribes to push notifications (symbol-outline updates, diagnostics) over the same socket / a WebSocket bridge.

**Why a separate crate + daemon, not inside `hawking-serve`:** (1) the index must survive across editor sessions and outlive any single inference server; (2) it must not contend with the decode hot path for the Metal queue except during *idle* windows it explicitly schedules; (3) it has a very different dependency surface (SQLite, Lance, tree-sitter, notify) we don't want in the inference binary; (4) independent testability and crash isolation.

### 4.2 The parsing layer

**Grammar registry.** A `GrammarRegistry` maps file extension / shebang / Monaco language-id → a bundle `{ Language (tree-sitter), tags.scm, locals.scm, chunk.scm, optional stack_graph.tsg, lsp: LspServerSpec }`. Grammars are compiled in (statically linked tree-sitter parsers) for the core set (Rust, TS/JS, Python, Go, C/C++, Java, C#, Ruby, JSON/TOML/YAML, Markdown, SQL, Bash) and dynamically loadable for the long tail (§7).

**Per-file parse state.** For each tracked file we keep a `ParsedFile`:
```rust
struct ParsedFile {
    path: RepoPath,
    lang: LangId,
    content_hash: Blake3Hash,        // of the bytes that produced `tree`
    tree: tree_sitter::Tree,         // the live CST (structurally shared on reparse)
    source: Arc<[u8]>,               // the bytes (for byte-range slicing)
    generation: u64,                 // index generation this parse belongs to
    error_spans: Vec<ByteRange>,     // ERROR/MISSING regions (for the health surface)
}
```

**Two freshness fast-paths.**

1. **Editor-buffer path (sub-keystroke).** Monaco fires `onDidChangeModelContent(e)` with `e.changes: [{ rangeOffset, rangeLength, text, range }]`. We translate each change to a tree-sitter `InputEdit` (we maintain the byte↔(row,col) mapping incrementally — Monaco gives us positions and offsets directly), call `tree.edit(...)` for each, then a single `parser.parse(new_bytes, Some(&old_tree))`. This reparse is sub-millisecond and feeds **live** symbol outline, local-scope nav, and diagnostics for the *open buffer* — before anything is saved or committed to the durable index. This buffer-tree is kept in memory keyed by the Monaco model URI and is *not* persisted; on save it converges with the disk path.

2. **Disk path (debounced, durable).** On a debounced FS change (§4.9), the merkle diff (§4.8) yields the exact changed files. For each, we load bytes, look up the previous `ParsedFile` (if any) and its prior bytes; if we have both we apply the *byte-diff* as `InputEdit`s for an incremental reparse, else we do a full parse. The new tree → tags → symbols → chunks → graph deltas, all committed in one batched transaction at a new generation.

**Error tolerance.** We never reject a file for parse errors. ERROR/MISSING spans are recorded in `error_spans` and surfaced in `/index/health` (and as Monaco squiggles distinct from LSP diagnostics). Symbols extracted *outside* error spans are fully indexed; symbols *inside* are best-effort (tree-sitter still gives partial structure). A file that is 100% ERROR (binary mistakenly treated as code, or a grammar mismatch) is flagged `unparseable` with a reason and indexed lexically only (trigram FTS over raw bytes) so grep still works.

**Pseudocode — incremental reparse:**
```
fn reparse(path, new_bytes):
    prev = parsed.get(path)
    tree =
        if let Some(p) = prev where p.source available:
            edits = byte_diff_to_input_edits(p.source, new_bytes)  // Myers diff → InputEdits
            for e in edits: p.tree.edit(e)
            parser.set_language(registry[path.lang])
            parser.parse(new_bytes, Some(&p.tree))                 // structural reuse
        else:
            parser.parse(new_bytes, None)                          // cold full parse
    error_spans = collect ERROR|MISSING node ranges (walk tree, is_error||is_missing)
    parsed.put(path, ParsedFile{ tree, source:new_bytes, content_hash:blake3(new_bytes), .. })
    return tree, error_spans
```

### 4.3 The symbol graph

**The model (SCIP-shaped storage, hybrid production).** We store occurrences and symbols in the SCIP shape because it makes nav an O(1) string lookup and is directly agent-readable.

- **Symbol ID = a structured global string.** Format (Hawking dialect, SCIP-compatible): `hawking <lang> <repo>@<rev|wc> <pkg> <descriptors>`, e.g. `hawking rust hawking@wc hawking_core crate/model/qwen_dense/forward_token_greedy_tcb().`. `@wc` = "working copy" (current uncommitted state); a concrete rev is used for historical occurrences (§4.12 history). Descriptors use SCIP suffixes (`/` namespace/module, `#` type, `.` term, `().` method, `:` meta, `!` macro). This ID is **stable across files and edits** as long as the qualified name is stable — so references resolve by string equality, and a rename is a *single ID remap* (§6).
- **Occurrence:** `(file, byte_range, symbol_id, role_bits)`. `role_bits` reuses SCIP roles (`Definition=0x1, Import=0x2, WriteAccess=0x4, ReadAccess=0x8, Generated=0x10, Test=0x20`). Go-to-def = `WHERE symbol_id=? AND role_bits & 0x1`. Find-refs = `WHERE symbol_id=?`. Read/write split lets us answer "where is X mutated?" for free.
- **SymbolInformation:** `(symbol_id, kind, display_name, signature, doc, enclosing_symbol_id)`. `kind` ∈ {function, method, class, struct, trait, interface, enum, field, const, module, macro, type_alias, …}.
- **Relationship:** `(symbol_id, related_id, kind)` with `kind ∈ {implements, extends, overrides, type_definition, reference}` — drives the type graph (§4.4).

**Production — three tiers, async-converging precision:**

1. **Tier-0 (instant, always): tree-sitter tags.** On every reparse, run `tags.scm` → defs/refs with names + ranges + kinds. This gives an *approximate* symbol graph immediately (name-matched, no overload/import resolution). Refs that the grammar can't capture (def-only grammars) are backfilled from a `locals.scm` pass or an identifier sweep. This tier alone makes the repo-map (§4.6) and lexical+symbol retrieval work.

2. **Tier-1 (seconds, incremental): stack-graph name resolution.** For languages with a `.tsg`, build the file's **partial scope-graph** from its CST (independent of all other files). Resolution = path-finding under the symbol/scope-stack discipline over the *union* of partial graphs, stitched at query time. **A changed file recomputes only its partial graph** — the rest of the workspace's partials are untouched. This upgrades approximate name-matches to *scoped* resolution (correct shadowing, imports, locals) without invoking a compiler. Partial graphs are persisted (they're per-file and content-addressed by the file's hash, so a re-open with unchanged content is free).

3. **Tier-2 (background, precise): LSP overlay.** A headless LSP fleet (§4.4) provides compiler-grade `definition`/`references`/`implementation`/`typeDefinition`. When an LSP answer arrives it *overlays* (corrects/confirms) the tier-0/1 edges, tagged with provenance `{tags | stackgraph | lsp}` and a confidence. The agent can request "precise only" (LSP-confirmed) or accept approximate. LSP warm-up latency is hidden because tier-0/1 already answer; LSP just sharpens.

**Why this hybrid:** we get *instant* approximate nav (tags), *incremental precise-ish* nav (stack-graphs, the right model for live editing), and *compiler-precise* nav (LSP) **converging asynchronously** — never blocking, monotonically improving. This is strictly better than any single system: SCIP alone is batch-produced; LSIF can't update incrementally; ctags can't resolve; LSP alone is slow to warm and N-round-trip.

### 4.4 Call, import, type, test & perf graphs

All graphs share **one node/edge model** persisted in SQLite (`nodes`, `edges` with materialized reverse edges) and loaded into petgraph for traversal. Edge `kind` discriminates the sub-graph; a query can restrict to one kind or walk several.

**Node/edge model (the contract):**
```
Node   = { node_id: u64, symbol_id: TEXT, kind: SymKind, repo: RepoId, file: FileId }
Edge   = { src: node_id, dst: node_id, kind: EdgeKind, provenance: {tags|stackgraph|lsp|cpg|coverage|profile},
           weight: f32, attrs: JSON }     // + a stored REVERSE row (dst,src,kind) or a reverse index
EdgeKind ∈ { Calls, References, Imports, DependsOn,
             Implements, Extends, Overrides, HasType, ReturnsType,
             Tests, CoversSymbol, Costs, AllocatesAt, … }
```

- **Call graph (`Calls`).** Edge caller→callee. Tier-0 from tags (`@reference.call` resolved to a def). Tier-2 sharpened by LSP `callHierarchy` (incoming/outgoing). The **reverse** call graph ("who calls X?", blast-radius for an edit) is a stored reverse index → an index seek, not a scan. CPG (§4.5) provides interprocedural precision when present.
- **Import/dependency graph (`Imports`, `DependsOn`).** Module/file → imported module/symbol. From tree-sitter import nodes + package manifests (Cargo.toml, package.json, go.mod, requirements). Powers "what depends on this module", build-order, and the *file-level* node weights in the repo-map. Cycle detection via SCC (petgraph `tarjan_scc`) surfaces import cycles.
- **Type graph (`Implements`, `Extends`, `Overrides`, `HasType`, `ReturnsType`).** This is the **LSP-driven** graph: `typeHierarchy/{supertypes,subtypes}` for inheritance/impl, `hover`/`typeDefinition` for expression types. Answers "what implements this trait?", "what overrides this method?", "what's the type here?". Tier-0 gives a rough `Implements`/`Extends` from syntax (`impl Trait for T`, `class C extends B`) before LSP confirms.
- **Test map (`Tests`, `CoversSymbol`).** Edges from a test symbol to the production symbols it exercises. Two producers: (a) **static heuristic** — a test's call graph closure ∩ production symbols (a test `Calls`+ reaches X ⇒ candidate `CoversSymbol`); plus naming conventions (`test_foo`↔`foo`, `Foo.test.ts`↔`Foo.ts`, Go `TestXxx`↔`Xxx`); (b) **dynamic ingestion** — parse coverage artifacts (`lcov`, `coverage.py`, `cargo-llvm-cov`, Go `-coverprofile`) to attach *measured* line/symbol coverage, which upgrades the heuristic edge to `provenance=coverage` with high confidence. Answers ch.03's `tests_covering(symbol)` and the inverse `symbols_covered_by(test)`.
- **Perf map (`Costs`, `AllocatesAt`).** Symbol → measured cost (time/calls/allocations). Producer: ingest profiler/trace output (the runtime already emits dispatch traces and `gpu_us`; we also ingest `cargo bench`/`perf`/Instruments exports). Edge `attrs` carry `{ns_per_call, calls, pct_of_parent}`. This makes the index *performance-aware*: the agent can ask "what's the hottest function reachable from `forward_token_greedy_tcb`?" and the Context Compiler can *boost* hot symbols when the task is optimization (a Hawking-native signal — see ch.04). Stale perf data is timestamped and decays.

**Headless LSP fleet (`symbols/lsp.rs`).** A supervised pool of language-server child processes (rust-analyzer, gopls, pyright, clangd, tsserver, JDTLS, …), one per (language × workspace-root), spawned lazily on first need, idle-reaped after a TTL, restarted on crash with backoff. Initialize handshake → `didOpen` the relevant files → fire batched requests. We **never** block a query on LSP; LSP results land asynchronously and overlay the graph at the next generation. A capability probe records which methods each server supports (workspace/symbol, typeHierarchy, callHierarchy, semanticTokens) so we degrade per-server. We harvest `documentSymbol` for the per-file outline (also fed to Monaco), `semanticTokens/full/delta` for incremental token classification, and the hierarchy/def/ref/impl methods for graph edges.

### 4.5 The optional code-property graph (dataflow/security)

A repo-wide CPG is too expensive to maintain in the hot loop (§3.4). Instead we expose **on-demand, scoped CPG overlays**:

- **Trigger:** the agent (ch.02) or a tool (ch.03 `dataflow_paths`, `taint_check`) asks a dataflow/security question about a function, file, or change-set. Examples: "does user input reach this SQL string un-sanitized?", "what writes to this field?", "trace this value to all sinks".
- **Build:** for the targeted scope (a function + its interprocedural neighborhood, bounded by a depth/size budget), construct AST ∪ CFG ∪ PDG (control-dependence + data-dependence/reaching-def edges) using the tree-sitter CST as the AST and a per-language CFG/DDG builder (we ship builders for the core languages; for others we shell out to Joern if installed). Reuse the existing symbol graph for the resolved call edges so we don't recompute name resolution.
- **Query:** taint reachability `sink.reachableBy(source)`, def-use slicing, "all sinks reachable from this source". Results are paths through the CPG with witness nodes.
- **Cache & invalidate:** a built overlay is cached keyed by the BLAKE3 hashes of the files in scope; any edit to those files invalidates it (merkle diff). It is *not* persisted across model/grammar upgrades.
- **Cost control:** strict budgets (max functions, max depth, wall-clock cap). If the scope explodes, we return a partial result with a "scope truncated" marker rather than hang.

This gives Joern-class power for security/dataflow tasks **without** paying the repo-wide CPG maintenance cost — the cloud can't do this interactively over your private code at all.

### 4.6 The repo-map ranking algorithm

The repo-map is the structural leg of ch.04's Context Compiler: given the current task/context, produce a **token-budgeted, signatures-only, ranked tree** of the most relevant definitions across the (multi-)repo. We port Aider's algorithm onto our **persistent** graph and add Hawking-native signals.

**Inputs (signals):**
- `chat_files` — files referenced in the current agent conversation.
- `open_tabs` — files open in Monaco (a strong proximity signal Aider lacks).
- `cursor_file` / `cursor_symbol` — where the user's caret is right now (strongest proximity).
- `mentioned_idents`, `mentioned_files` — identifiers/paths named in the task prompt.
- `recently_edited` — files touched in the last N edits (git working-copy + editor history).
- `git_recency` / `git_age` — churned-recently files boosted; ancient stable files slightly damped (the index *has* whole history, §4.12).
- `task_kind` — if the task is "optimize", boost perf-map-hot symbols; if "fix test", boost test-map neighbors of the failing test.

**The graph.** We reuse the persistent reference graph (nodes = files *and* symbols; we run PageRank at the **file** granularity à la Aider, then distribute rank to `(file, symbol)` definitions). Edge weights mirror Aider's secret sauce, recomputed only on the changed subgraph:
- referencer ∈ `chat_files | open_tabs` → strong `use_mul` boost (chat ×50, open-tab ×20, cursor-file ×80);
- `ident ∈ mentioned_idents` → ×10;
- distinctive long multiword identifier (snake/kebab/camel, len ≥ 8) → ×10;
- `_private` → ×0.1; defined in > 5 files (generic) → ×0.1;
- `weight = use_mul · sqrt(num_refs)` (sqrt damps single-file domination).

**The algorithm (persistent, incremental personalized PageRank):**
```
fn repo_map(signals, token_budget) -> RankedTree:
    # 1. Personalization vector (mass to seed nodes)
    P = {}
    for f in signals.chat_files:        P[f] += BASE
    for f in signals.open_tabs:         P[f]  = max(P[f], BASE)        # don't double-count
    if signals.cursor_file:             P[cursor_file] += 2*BASE
    for f where path_component ∈ signals.mentioned_idents: P[f] += BASE
    for f in signals.recently_edited:   P[f] += RECENCY_DECAY(age)
    normalize(P)                        # sums to 1; uniform fallback if empty

    # 2. PageRank over the (maintained) reference graph, personalized
    #    Incremental: if only `signals` changed (not the graph), re-run power-iteration
    #    warm-started from the last rank vector → converges in 1-3 iters.
    R = personalized_pagerank(G, weight="weight", personalization=P, alpha=0.85, warm_start=last_R)

    # 3. Distribute each file's rank across its out-edges, credit (definer, ident)
    ranked_defs: Map<(file, symbol_id), f32> = {}
    for src in G.nodes:
        tw = Σ edge.weight for edge in out_edges(src)
        for (s, dst, e) in out_edges(src):
            ranked_defs[(dst, e.symbol_id)] += R[src] * e.weight / tw

    # 3b. Hawking-native re-weighting
    for (file, sym), r in ranked_defs:
        if task_is_optimize: r *= perfmap_hotness(sym)        # boost hot symbols
        if task_is_fix_test: r *= testmap_proximity(sym, failing_test)
        r *= recency_boost(file)                              # git/edit recency

    # 4. Binary-search the count of top defs that fit token_budget; render elided
    defs_sorted = sort_desc(ranked_defs)  (skip defs whose file already fully in context)
    lo, hi = 0, len(defs_sorted)
    best = None
    while lo <= hi:
        mid = (lo+hi)//2
        tree = render_elided(defs_sorted[:mid])      # signatures only, bodies → ⋮ (per-lang scope query)
        toks = token_count(tree)
        if toks <= token_budget: best = tree; lo = mid+1
        else: hi = mid-1
    return best
```

**Rendering (`repomap/render.rs`).** Per-language `chunk.scm`/scope queries identify the *signature lines* (def + enclosing scope headers) and collapse bodies to `⋮`, exactly like Aider's `TreeContext`, capped at ~100 chars/line (minified-file guard). Output is a compact elided tree grouped by file/module.

**Incrementality.** Crucially, PageRank runs on a **standing, maintained** graph. When code changes, we update only the changed subgraph's edges and *warm-start* the next PageRank from the prior rank vector (PageRank converges in 1–3 power iterations from a warm start when the graph barely moved). When only the *signals* change (user switches tabs, moves the caret, sends a message), the graph is unchanged and we just re-run the personalized iteration — milliseconds. This is the win over Aider's rebuild-per-query model at scale.

**Token-budget contract with ch.04.** ch.04 owns the *total* context budget and calls `repo_map(signals, token_budget)` with a budget slice. The repo-map returns the ranked elided tree *and* a structured `RankedSymbol[]` list (so ch.04 can interleave repo-map entries with retrieved chunks and de-dup). See §4.11.

### 4.7 The semantic index

**Retriever architecture (lexical/symbol-first, embeddings re-rank — per ground truth and SOTA):**

```
query (NL task or code) ──► [ Leg A: SYMBOL ]  exact symbol/name resolution (the graph): defs/refs of named idents
                       │
                       ├──► [ Leg B: LEXICAL ] SQLite FTS5: BM25 over identifier/body cols + trigram substring
                       │
                       └──► [ Leg C: VECTOR ]  ANN over chunk embeddings (Lance)   ◄─ weakest leg TODAY
                                   │
                          ┌────────┴────────┐
                          │  FUSE via RRF   │  k=60; legs A,B,C with weights (A,B ≫ C until code role lands)
                          └────────┬────────┘
                                   │ top-N (≈50)
                          ┌────────┴────────┐
                          │  RE-RANK         │  cross-encoder OR local-LLM listwise (free, private)
                          └────────┬────────┘
                                   │ top-k (≈8–12)
                                   ▼  → ch.04 Context Compiler
```

- **Leg A (symbol):** if the task names identifiers/paths, resolve them via the symbol graph (exact, precise). Highest precision; near-zero cost.
- **Leg B (lexical):** SQLite FTS5 with **BM25** and per-column weights (identifier column weighted far above body), plus a **trigram** index for substring/identifier-fragment search (`getUserId` via `serId`) and a `unicode61` index for short tokens trigram can't cover. This is the recall workhorse for code — it catches exact API names embeddings blur.
- **Leg C (vector):** ANN over chunk embeddings in Lance. **Weighted low** in the RRF fusion *today* (because `embed()` is a logits proxy), serving mainly as semantic *expansion* (catching paraphrase/concept matches legs A/B miss). When the dedicated code role lands, a config dial (§9) raises Leg C's weight and may promote it to a first-class recall leg — **no schema change**, because vectors carry `embed_model_id` + `dim`.
- **Fusion:** Reciprocal Rank Fusion, `RRF(d)=Σ wᵣ/(k+rankᵣ(d))`, `k=60`. Weights `{A: 1.0, B: 1.0, C: 0.3→1.0}` (C ramps with model quality). RRF is scale-free → no per-leg score normalization needed.
- **Re-rank:** over the fused top-N (~50), a precision pass to ~8–12. Two interchangeable rerankers: (1) a **cross-encoder** code reranker if a reranking role is served; (2) a **local-LLM listwise** re-rank — we hand the candidate snippets to the local model (`hawking-serve`) with a "rank these by relevance to the task" prompt. The listwise pass is *free and private* and uses the model we already host — a Hawking-native advantage the cloud charges for. The reranker is also where our weak local embedding's mistakes get corrected by a stronger signal.

**Chunking (`parse/chunker.rs`) — cAST / by-symbol.** We chunk **by AST symbol**, not fixed windows: each function/method/class is a chunk; a chunk that exceeds the embedding model's token budget is **recursively split** (and oversized function bodies are collapsed to a *signature + `{ … }`* with children recursed, à la Continue.dev), and small adjacent siblings are greedily **merged** to fill the budget (cAST). Each chunk carries `{ symbol_id, file, byte_range, lang, signature, enclosing_path }` so a retrieval result maps back to an exact span and the agent can expand to the full symbol. Chunk identity is content-addressed (BLAKE3 of the chunk text) → **unchanged chunks are never re-embedded** (the dominant incremental-embedding win, Cursor's chunk-hash cache).

**Embedding (`semantic/embed.rs`).** Talks to `hawking-serve` `POST /v1/embeddings` (batched — the handler already accepts a batch array). Each vector is stored with `embed_model_id` (e.g. `"logits-proxy:<model_id>"` today, `"code-role:<id>"` later) and `dim`. **Model/role versioning is first-class:** when the embedding model changes, we *don't* invalidate the index — we mark vectors with the old `embed_model_id` as stale and **re-embed lazily on idle GPU** (changed/queried chunks first), so search keeps working (mixed-model) and converges to the new model in the background. Embedding runs in **idle-GPU batches** (§4.9): when the user isn't decoding, we burst large batches of changed chunks through the Metal path (large batches amortize per-call overhead). On Apple Silicon's unified memory, the same mmap'd vectors feed CPU ANN search and GPU embedding with zero copy.

**The honest stance on embeddings.** Today, Leg C is a *re-ranking/expansion* signal, deliberately down-weighted, because the proxy embedding is weak. Legs A (symbol) and B (lexical) carry recall. This is not a workaround we're embarrassed by — it's where Sourcegraph, Anthropic, and Augment independently landed for code. The architecture is *designed* to absorb a real code embedding the moment it's served, with a single dial, no re-index, no schema churn.

### 4.8 Change detection (BLAKE3 merkle-DAG)

A workspace-wide **merkle-DAG** over the file tree is the gate for *all* incremental work. Nothing re-indexes unless its hash changed.

- **Leaves:** `leaf_hash(file) = BLAKE3(file_bytes)`. (For very large files we additionally keep the BLAKE3 *chunk tree* so we can verify/diff sub-ranges via bao-tree.)
- **Directory nodes:** `dir_hash = BLAKE3( sorted [ (name, type, child_hash) ] )` — sorted entries make it canonical/order-independent (Git tree object semantics). The single **root hash changes iff anything in the tree changed**.
- **O(changed) diff (`merkle/diff.rs`):** to find what changed between snapshot `S_prev` and current, compare root hashes; if equal → done (O(1)); else recurse only into child entries whose hashes differ, pruning matching subtrees. Cost ∝ number of changed paths, **not** repo size. Output = a `ChangeSet { added, modified, deleted, renamed }`.
- **Rename detection.** A rename changes a path → ancestor hashes ripple even if content is identical. We **pair the path-diff with content-hash identity**: a `deleted(p_old, h)` + `added(p_new, h)` with the *same leaf hash* in one changeset ⇒ a **rename** (`renamed(p_old → p_new)`), so the symbol graph does an ID *remap* instead of delete+reinsert (and unchanged content is never re-embedded/re-parsed). This is the correctness backstop that absorbs the watcher's racy/duplicate rename events (§4.9).
- **Snapshots & history.** Each committed index generation records its root hash. "What changed since Y" (§4.11) = diff the generation's stored tree against generation/commit Y's tree — O(changed), across the whole history if we've ingested it (§4.12).
- **Why BLAKE3 specifically:** ~4.1 GB/s on M3 (NEON), ~5–8× faster than software SHA-256 on Apple Silicon (no SHA-NI), internally a merkle tree (parallel + verified streaming for free), mature Rust crate. Hashing the whole tree on cold start is fast; per-edit we re-hash one file + O(depth) ancestor dirs.

### 4.9 The Living-Index daemon

`hawking-indexd` is the always-on organ. Its loop:

```
                ┌─────────────── notify (FSEvents) ───────────────┐
                │                                                  │
   ┌────────────▼───────────┐   ┌──────────────────────┐   ┌──────▼─────────┐
   │ debouncer-full (200ms) │──►│ merkle diff → ChangeSet│──►│  work queue    │
   │ rename-stitch, dedup   │   │ (O(changed))           │   │ (priority)     │
   └────────────────────────┘   └──────────────────────┘   └──────┬─────────┘
   Monaco buffer-change  ───────────────────────────────────────► │  (highest prio:
   (editor fast path)                                              │   open file)
                                                                   ▼
                                  ┌────────────────────────────────────────────┐
                                  │ incremental update pipeline (per changed f) │
                                  │  reparse → tags → stackgraph partial →       │
                                  │  chunk(dirty only) → embed(idle batch) →     │
                                  │  graph delta → repomap subgraph dirty →      │
                                  │  COMMIT one batched txn at generation g+1    │
                                  └───────────────┬──────────────────────────────┘
                                                  │ atomic manifest swap (g → g+1)
                                                  ▼
                                       readers advance to g+1 (MVCC)

   Idle/GPU-free detector ──► background full-reindex / lazy re-embed of stale-model vectors
```

**Watching (`daemon/watcher.rs`).** `RecommendedWatcher` (FSEvents) **on directories, recursively**, wrapped in **notify-debouncer-full** (window ~200ms; rename-stitch via `FileIdMap`; dedup). Paths filtered through the `ignore` crate against `.gitignore` + a built-in ignore set (`.git`, `node_modules`, `target`, `dist`, `build`, `.venv`, lockfile-noise). **The watcher event is treated as a hint** — the merkle diff decides the actual work, absorbing duplicate/racy/dropped events, FSEvents history loss, and editor temp-file churn (atomic-save-via-rename shows up as create-`.tmp`+rename; the merkle diff sees only the net content change). On macOS we fall back to `PollWatcher` for directories we can't observe via FSEvents (un-owned files).

**Scheduling (`daemon/scheduler.rs`).** A priority work queue:
1. **Editor-buffer changes** (the open file) — highest, sub-keystroke, in-memory only.
2. **Saved-file changes** — high, debounced, durable.
3. **Cross-file ripple** (a save invalidated other files' resolution / repo-map subgraph) — medium.
4. **Idle full-reindex / lazy re-embed / LSP warm** — lowest, only when idle.

**Idle & GPU detection.** The daemon detects (a) *user idle* (no edits/queries for T seconds) and (b) *GPU idle* (no active `hawking-serve` decode — it can query the serve process's status or simply detect no embedding/decode contention). During idle+GPU-free windows it runs the expensive background work: re-embedding stale-model vectors, building/maintaining the optional CPG cache for hot files, warming LSP servers, compacting segments (§4.10), and — on cold start or after corruption — a **full reindex**. All idle work is *preemptible*: a new edit or a decode request immediately pauses background GPU use (the daemon yields the Metal queue) and re-prioritizes the foreground change. This is the "living representation kept fresh by idle compute" moat — the cloud has no idle local GPU to spend.

**Freshness guarantees & the health surface (`daemon/health.rs`).** We expose, per repo, a `Freshness` record: `{ root_hash, generation, lag_ms (FS-event → committed), pending_files, unparseable_files, stale_model_vectors, lsp_status }`. The contract: **the open editor buffer is parsed within one keystroke; a saved file's *symbol* layer is committed within ~1–2 s; its vector/CPG/perf layers converge on idle.** Tools/agent can read `Freshness` and, for correctness-critical queries, *wait for a generation* (`query.min_generation = g`) so they never act on a stale view.

**Crash-safety (the spine — §4.10 details).** Every commit is an append-only segment + an **atomic manifest swap** at a new generation. A crash mid-commit leaves at most a torn segment tail, truncated on recovery; the last good generation's manifest is authoritative. On startup the daemon: (1) reads the manifest → last good generation; (2) truncates any torn tail; (3) re-hashes the workspace root and diffs against the generation's stored tree → reindexes only what changed while it was down. **Never a corrupt or partially-updated index.** The merkle root in the manifest also detects "user edited files while the daemon was dead" — covered by the same diff.

### 4.10 Storage architecture & schemas

Four cooperating stores under `~/.hawking/index/<workspace-id>/` (and a shared global store for cross-repo, §4.12):

```
store/
├── catalog.sqlite        # WAL — files, symbols, occurrences, edges (+reverse), generations, FTS5
├── vectors.lance/        # Lance dataset — chunk embeddings (IVF-PQ/SQ), soft-delete + optimize()
├── cas/                  # BLAKE3 content-addressed blobs (parse-tree snapshots, partial scope-graphs)
│   └── ab/cdef…          # 2-hex fan-out (Git layout)
├── segments/             # immutable segment files (lexical postings / graph deltas, tantivy-style)
└── MANIFEST              # append-only VersionEdit log; current generation pointer (atomic-swapped)
```

**(1) The catalog — SQLite (WAL).** The relational source of truth. PRAGMAs on every connection: `journal_mode=WAL; synchronous=NORMAL; busy_timeout=5000; foreign_keys=ON; temp_store=MEMORY; cache_size=-262144; mmap_size=268435456`. One writer (the daemon, `BEGIN IMMEDIATE`, **batched** commits per-file/N-rows — ~600× over autocommit), many readers (snapshot reads).

```sql
-- Repos & files
CREATE TABLE repo (
  repo_id     INTEGER PRIMARY KEY,
  root_path   TEXT NOT NULL UNIQUE,
  vcs         TEXT,                       -- 'git' | 'none'
  root_hash   BLOB                        -- current merkle root (BLAKE3)
);
CREATE TABLE file (
  file_id     INTEGER PRIMARY KEY,
  repo_id     INTEGER NOT NULL REFERENCES repo,
  rel_path    TEXT NOT NULL,
  lang        TEXT,
  content_hash BLOB NOT NULL,             -- BLAKE3 leaf hash
  size_bytes  INTEGER,
  parse_state TEXT,                       -- 'ok' | 'errors' | 'unparseable' | 'generated'
  generation  INTEGER NOT NULL,           -- generation this row reflects
  UNIQUE(repo_id, rel_path)
);
CREATE INDEX file_hash ON file(content_hash);

-- Symbols (SCIP-shaped)
CREATE TABLE symbol (
  symbol_id   TEXT PRIMARY KEY,           -- structured global string id
  kind        INTEGER NOT NULL,           -- SymKind enum
  display_name TEXT,
  signature   TEXT,
  doc         TEXT,
  enclosing   TEXT REFERENCES symbol(symbol_id),
  def_file    INTEGER REFERENCES file,
  def_range   INTEGER,                    -- packed start/end (or join to occurrence)
  generation  INTEGER NOT NULL
);
CREATE INDEX symbol_def_file ON symbol(def_file);
CREATE INDEX symbol_kind ON symbol(kind);

-- Occurrences (every def/ref/read/write)
CREATE TABLE occurrence (
  occ_id      INTEGER PRIMARY KEY,
  file_id     INTEGER NOT NULL REFERENCES file,
  symbol_id   TEXT NOT NULL REFERENCES symbol,
  start_line  INTEGER, start_col INTEGER, end_line INTEGER, end_col INTEGER,
  role_bits   INTEGER NOT NULL,           -- Definition=1,Import=2,Write=4,Read=8,Generated=16,Test=32
  provenance  INTEGER NOT NULL,           -- tags|stackgraph|lsp
  generation  INTEGER NOT NULL
);
CREATE INDEX occ_symbol ON occurrence(symbol_id);          -- find-refs: O(index seek)
CREATE INDEX occ_file   ON occurrence(file_id);
CREATE INDEX occ_def    ON occurrence(symbol_id) WHERE role_bits & 1;   -- go-to-def

-- The unified graph (call/import/type/test/perf)
CREATE TABLE node (
  node_id     INTEGER PRIMARY KEY,
  symbol_id   TEXT REFERENCES symbol,
  file_id     INTEGER REFERENCES file,
  kind        INTEGER NOT NULL
);
CREATE TABLE edge (
  src         INTEGER NOT NULL REFERENCES node,
  dst         INTEGER NOT NULL REFERENCES node,
  kind        INTEGER NOT NULL,           -- EdgeKind: Calls,Imports,Implements,Tests,Costs,...
  provenance  INTEGER NOT NULL,
  weight      REAL DEFAULT 1.0,
  attrs       TEXT,                        -- JSON (e.g. perf {ns_per_call,...})
  generation  INTEGER NOT NULL,
  PRIMARY KEY (src, kind, dst)
);
CREATE INDEX edge_fwd ON edge(src, kind);
CREATE INDEX edge_rev ON edge(dst, kind);   -- MATERIALIZED reverse: "who calls/depends-on X?" = seek

-- Chunks (semantic units; link to vectors.lance by chunk_id)
CREATE TABLE chunk (
  chunk_id    INTEGER PRIMARY KEY,
  file_id     INTEGER NOT NULL REFERENCES file,
  symbol_id   TEXT REFERENCES symbol,
  start_byte  INTEGER, end_byte INTEGER,
  chunk_hash  BLOB NOT NULL,              -- BLAKE3 of chunk text (dedup; skip re-embed)
  embed_model_id TEXT,                    -- which model produced its vector (NULL = pending)
  generation  INTEGER NOT NULL
);
CREATE INDEX chunk_hash ON chunk(chunk_hash);
CREATE INDEX chunk_file ON chunk(file_id);

-- Lexical search (FTS5): two tokenizers
CREATE VIRTUAL TABLE fts_body USING fts5(
  body, ident, path UNINDEXED,
  content='', contentless_delete=1        -- allow delete/update on re-index of changed files
);                                         -- query: ORDER BY bm25(fts_body, 5.0, 10.0)  (ident≫body)
CREATE VIRTUAL TABLE fts_tri USING fts5(
  body, tokenize='trigram', detail='none' -- substring/identifier-fragment search
);

-- Generations (MVCC anchor) & history
CREATE TABLE generation (
  generation  INTEGER PRIMARY KEY,
  root_hash   BLOB NOT NULL,
  git_rev     TEXT,                        -- the commit this generation corresponds to (if any)
  created_ms  INTEGER NOT NULL,
  status      TEXT                         -- 'committed' | 'in_progress'
);
```

Transitive-closure queries (callers/impls/path) use **recursive CTEs** over `edge` (UNION to break cycles, depth-cap, indexed on both columns) for ad-hoc depth, and an in-memory **petgraph/CSR** load for hot BFS/SCC/bidirectional-BFS. For *very hot* all-transitive queries we maintain a materialized closure table per component, recomputed only for the affected component on change (files change one at a time → cheap), or we let **CozoDB** own materialization if we adopt it.

**(2) Vectors — Lance.** A Lance dataset of `(chunk_id, embed_model_id, vector[dim])` with an IVF-PQ (or IVF-SQ/int8) index. Churn model fits an IDE: new chunks **append** (queryable immediately via flat-scan over the un-indexed tail), changed/deleted chunks are **soft-deleted** (tombstone, applied at query), and a periodic background `optimize()` (idle window) folds the tail into the index and prunes tombstones. Vectors are mmap'd (disk-resident, minimal RAM). **usearch** is the drop-in alternate (mmap HNSW + NEON + i8/binary) if we want lower query latency and accept periodic shard rebuilds under churn; **sqlite-vec** is the exact-delete durable co-store / fallback for small shards. The store is abstracted behind a `VectorStore` trait so the engine is swappable.

**(3) CAS — BLAKE3 content-addressed blobs.** Stores immutable artifacts by content hash: serialized parse-tree snapshots (for fast file reopen + history), per-file **partial scope-graphs** (content-addressed by file hash → unchanged file reopen is free), and large packed exports. Git's object layout (header `<type> <size>\0` + zlib, 2-hex fan-out `ab/cdef…`, loose→pack with bounded deltas), **BLAKE3 instead of SHA-1**. GC = mark-and-sweep from live generations (grace period to avoid the resurrection race).

**(4) Segments + MANIFEST — the crash-safe incremental spine (tantivy model).** The lexical postings and graph-delta logs are written as **immutable, UUID-named segments**; each index operation gets a **monotonic opstamp**; deletes are **tombstone bitsets** (`<segment>.<opstamp>.del`) applied at read and reclaimed only at **merge** (a throttled background pass — `LogMergePolicy`-style, throttled so it never starves the editor/decoder). The **MANIFEST** is an append-only log of `VersionEdit` records (segments added/removed); the "current" pointer is advanced by writing a temp manifest → `fsync` → `rename` → **`fsync` the directory fd** (atomic-or-nothing; the dir-fsync is the easy-to-miss durability step — skipping it can resurrect an old pointer after reboot, the exact RocksDB bug). A reader pins a **generation** (a `Searcher`-equivalent holding a fixed segment list + refcounts); segments are physically GC'd only when no pinned reader references them. `reload()` advances a reader to the newest generation atomically. **We strongly consider adopting Tantivy** directly for the lexical leg rather than re-implementing this — it *is* this design, battle-tested (backs Quickwit/ParadeDB).

**MVCC summary.** One generation counter is the spine: it orders writes, anchors snapshot reads (a query pins generation `g` and sees exactly the file/segment set live at `g`), and gates GC. Readers never see a half-applied update.

### 4.11 The query API surface

This is the **stable contract** ch.02/03/04 bind to. It is exposed three ways, all over the same engine: (a) a Rust trait (`hawking-index::Index`) for in-process callers; (b) a local Unix-socket JSON-RPC for the daemon's control + cross-process queries; (c) the CLI `hawking index query …`. All queries accept an optional `min_generation` (wait-for-freshness) and a `precise: bool` (LSP-confirmed only vs accept approximate). Every result carries `generation` and per-edge `provenance`+`confidence`.

```rust
pub trait Index {
    // ---- Navigation (O(1)-ish, occurrence lookups) ----
    fn find_definition(&self, sym: SymbolRef, q: Q) -> Result<Vec<Location>>;
    fn find_references(&self, sym: SymbolRef, q: Q) -> Result<Vec<Location>>;
    fn find_callers(&self, sym: SymbolRef, q: Q) -> Result<Vec<Edge>>;     // reverse call graph (seek)
    fn find_callees(&self, sym: SymbolRef, q: Q) -> Result<Vec<Edge>>;
    fn find_implementations(&self, sym: SymbolRef, q: Q) -> Result<Vec<Symbol>>;  // type graph
    fn find_overrides(&self, sym: SymbolRef, q: Q) -> Result<Vec<Symbol>>;
    fn type_of(&self, loc: Location, q: Q) -> Result<TypeInfo>;            // LSP hover-backed
    fn document_symbols(&self, file: FileRef) -> Result<SymbolTree>;        // outline (Monaco)

    // ---- Graph reasoning ----
    fn path_between(&self, a: SymbolRef, b: SymbolRef, kinds: &[EdgeKind], q: Q)
        -> Result<Vec<Path>>;        // bidirectional BFS over petgraph; witness path(s)
    fn transitive_callers(&self, sym: SymbolRef, depth: Option<u32>, q: Q) -> Result<Subgraph>;
    fn dependencies_of(&self, module: ModuleRef, transitive: bool, q: Q) -> Result<Subgraph>;
    fn dependents_of(&self, module: ModuleRef, transitive: bool, q: Q) -> Result<Subgraph>; // blast radius
    fn import_cycles(&self, repo: RepoRef) -> Result<Vec<Cycle>>;

    // ---- Test & perf maps ----
    fn tests_covering(&self, sym: SymbolRef, q: Q) -> Result<Vec<TestRef>>;
    fn symbols_covered_by(&self, test: TestRef, q: Q) -> Result<Vec<Symbol>>;
    fn hot_symbols(&self, scope: Scope, q: Q) -> Result<Vec<(Symbol, PerfCost)>>;  // perf map

    // ---- Dataflow / security (optional CPG, on-demand) ----
    fn dataflow_paths(&self, source: Location, sink_kind: SinkKind, budget: CpgBudget)
        -> Result<Vec<DataflowPath>>;
    fn taint_check(&self, change: ChangeRef, policies: &[TaintPolicy]) -> Result<Vec<Finding>>;

    // ---- Search (lexical-first hybrid; embeddings re-rank) ----
    fn grep_symbol(&self, pat: &str, q: Q) -> Result<Vec<Location>>;       // FTS5 trigram/BM25
    fn search(&self, query: &str, opts: SearchOpts) -> Result<Vec<Hit>>;   // Leg A⊕B⊕C → RRF → rerank
    //   SearchOpts { k, leg_weights, rerank: {none|cross_encoder|local_llm}, filter: {lang,path,...} }

    // ---- Context Compiler feed (ch.04) ----
    fn repo_map(&self, signals: RepoMapSignals, token_budget: u32) -> Result<RepoMap>;
    //   RepoMap { elided_tree: String, ranked: Vec<RankedSymbol{symbol_id, file, score, signature}> }

    // ---- History / change ----
    fn changed_since(&self, since: Rev, q: Q) -> Result<ChangeSet>;        // merkle diff of generations
    fn blame_age(&self, sym: SymbolRef) -> Result<AgeInfo>;               // from history ingest
    fn symbol_history(&self, sym: SymbolRef) -> Result<Vec<HistoricalDef>>; // def across revs

    // ---- Freshness / health ----
    fn freshness(&self, repo: RepoRef) -> Result<Freshness>;
    fn await_generation(&self, repo: RepoRef, g: u64, timeout: Duration) -> Result<u64>;
}

pub struct Q { pub min_generation: Option<u64>, pub precise: bool, pub repos: Vec<RepoRef> }
```

**ch.03 tool mapping (thin wrappers, no parsing):** `find_definition→find_definition`, `find_references→find_references`, `find_callers→find_callers/transitive_callers`, `find_implementations→find_implementations`, `path_between→path_between`, `tests_covering→tests_covering`, `changed_since→changed_since`, `grep→grep_symbol/search`, `dataflow→dataflow_paths`.

**ch.04 binding:** ch.04 calls `repo_map(signals, budget_slice)` for the structural leg and `search(task, opts)` for the semantic leg, interleaves+dedups `RankedSymbol`s with `Hit`s, and expands chosen symbols to full bodies via `find_definition` byte-ranges. The repo-map returns *both* a render-ready elided tree and a structured list so ch.04 controls final assembly.

### 4.12 Multi-repo, monorepo & million-line scaling

**Multi-repo & monorepo sharding.** Each repo (or, for a giant monorepo, each top-level *module/package*) is a **shard**: its own `file`/`symbol`/`occurrence`/`chunk` rows are tagged by `repo_id`, its own merkle subtree, its own Lance fragment-group, its own segment set. Sharding keeps incremental work and locking local (editing in package A never locks package B's index) and lets us load only the shards a query touches. Cross-shard edges (a symbol in A referenced from B) are stored in a **global edge table** keyed by the *string* `symbol_id` (SCIP IDs are global) — so cross-repo go-to-def is still a string lookup, no shard join needed. A **global symbol catalog** (just `symbol_id → (repo_id, def_file)`) indexes definitions across all shards for workspace-wide nav.

**Lazy & sharded loading.** Shards are loaded on demand: opening a file pulls its shard's hot tables into the SQLite page cache; petgraph subgraphs are loaded per-query-scope, not whole-repo. The optional CPG and full per-file scope-graphs are built/loaded lazily for the touched scope only.

**Memory budget.** The daemon runs under a configurable RAM ceiling. Levers: (1) vectors are **disk-resident mmap'd** (Lance/usearch) — near-zero resident; (2) f16/int8 vector quantization shrinks the mmap'd footprint; (3) parse trees for cold files are **dropped** (we keep only hashes + the durable symbol/chunk rows; reparse on demand from CAS-stored snapshot or source); (4) petgraph is loaded per-scope and dropped; (5) SQLite `cache_size`/`mmap_size` are bounded; (6) LSP servers are idle-reaped. **Working set, not whole repo, is resident.** Apple-Silicon unified memory lets the mmap'd vector pages serve CPU search and GPU embedding without copies, but we plan for OS page-cache weakness on ANN (MADV_RANDOM for the vector store, idle MADV_WILLNEED warming of likely-hot shards, PQ-in-RAM + bounded SSD round-trips) and keep RAM headroom to avoid the ~10× swap cliff.

**Million-line scaling, concretely.** Cold index of a 1M-line monorepo: parse + tags (sub-ms incremental, tens-of-ms full per file) and FTS5 trigram build (~126k rows/s measured) are the bulk; embedding is the long pole (idle-GPU, batched, chunk-hash-cached so re-runs are near-free). After cold start, *steady-state* cost is **O(changed files)** every time — a one-file edit re-parses one file, recomputes one partial scope-graph, re-chunks/re-embeds only changed chunks, updates the changed subgraph, warm-starts PageRank in 1–3 iters. The merkle gate guarantees we never touch unchanged code. Query latency: nav = index seek (µs–ms); `path_between`/closure = in-memory BFS over the per-scope petgraph (µs–ms for shallow code-dependency depths); `search` = FTS5 (ms) ⊕ ANN (ms over mmap'd Lance) → RRF → rerank (the rerank LLM call dominates, bounded by k). This is the scale Sourcegraph/Kythe operate at; we do it *locally and incrementally* because we own idle compute and never re-batch the world.

**Whole-history ingest (optional, idle-built).** Because we have the disk and the compute, we optionally walk the git DAG and index *historical* definitions: `symbol_id` carries `@<rev>`, `generation` rows carry `git_rev`, so `symbol_history`, `blame_age`, and `changed_since(old_rev)` work across the entire history — answering "when did this function last change and what did it look like 50 commits ago" without a `git` shell-out per query. Built incrementally on idle, capped by a history-depth dial (§9).

---

## 5. How we exceed cloud (the moat)

Each of these is a thing a per-token, ephemeral-infra SaaS assistant **structurally cannot do**:

1. **Always-on, idle-GPU living index.** The representation is *never stale* because a background daemon spends the user's idle GPU keeping it fresh — re-embedding, re-resolving, warming LSP, compacting. Cloud has no idle local GPU to burn per-user 24/7; it must (re)index at request time, pay for it, and rate-limit it. *(PROVEN building blocks; the always-on orchestration is the Hawking-native assembly.)*
2. **Whole-history, whole-machine index.** Every repo the user opens, all of git history, generated artifacts — indexed and cross-linked. "What did this function look like 80 commits ago and who called it then" is a local lookup. Cloud indexes a snapshot of *one* repo it's pointed at, gated by upload limits and privacy.
3. **Private, free embeddings (and re-ranking).** Code never leaves the machine. Embedding and **local-LLM listwise re-ranking** run on the model we already host — free and private. This is exactly the privacy/cost/staleness pain Sourcegraph and Anthropic cited when they *removed* cloud embeddings. We get the upside (semantic re-rank) without the downsides because the model is local.
4. **Cross-session forever-memory of the code.** The index persists across editor restarts and across days; the agent starts every session already knowing the whole codebase's structure, not re-discovering it. Combined with ch.02's memory, "what we learned about this code" compounds. Cloud assistants re-establish context per session.
5. **Determinism & integrity.** Content-addressed, canonically-ordered, BLAKE3-verified. Reproducible index artifacts make caching, diffing, and bug-repro trivial — a Hawking family value the cloud's shared mutable infra can't match.
6. **On-demand private CPG dataflow/security** over your proprietary code, interactively — a capability cloud tools can't offer over code you won't upload.
7. **Performance-aware retrieval.** The perf map makes the index *know what's hot* on the user's own hardware (we already capture `gpu_us`/dispatch traces). The agent retrieves hot symbols when optimizing. No cloud assistant has your machine's profiles.

---

## 6. Failure modes & mitigations

| Failure mode | What breaks | Mitigation |
|---|---|---|
| **Parse errors / half-typed code** | A file won't fully parse mid-edit. | tree-sitter ERROR/MISSING **localizes** damage; symbols outside error spans are fully indexed; error spans recorded in `parse_state='errors'` + surfaced to health/Monaco. Never reject the file. |
| **Fully unparseable (binary-as-code, grammar mismatch, novel syntax)** | No AST. | Flag `parse_state='unparseable'` with reason; index **lexically only** (trigram FTS over raw bytes) so grep still works; skip graph/chunk layers; retry on grammar upgrade. |
| **Huge files (minified JS, generated 100k-line files, vendored bundles)** | Parse/embed cost blows up; noise pollutes ranking. | Size threshold → **budgeted handling**: parse but cap chunk count; line-length cap (100 chars) in repo-map render guards minified files; mark `parse_state='generated'` and **down-weight** in repo-map (Aider's "defined in many files / generic → ×0.1" analog); optionally lexical-only above a hard cap. BLAKE3 chunk-tree (bao) lets us verify/diff sub-ranges of huge files without full reads. |
| **Generated code (protobuf stubs, OpenAPI clients, codegen output)** | Pollutes symbol graph & retrieval with machine noise. | Detect via `.gitignore`/`@generated` markers/path heuristics → `Generated` role bit (SCIP `0x10`) → **excluded from repo-map ranking by default**, included only if explicitly queried. Respect `.gitignore` via the `ignore` crate. |
| **Renames / moves** | Naive watcher reads delete+create → symbol graph churns, vectors re-embedded, history lost. | **Merkle content-hash identity**: same leaf hash on delete+add in one changeset ⇒ `renamed` ⇒ symbol-ID **remap**, no re-parse/re-embed, history preserved. notify-debouncer-full stitches the FS rename; merkle is the backstop. Watch the **parent dir** (stable inode), not the file. |
| **Watcher misses events (FSEvents history loss, inotify `IN_Q_OVERFLOW`, daemon was down)** | Index silently stale. | The watcher is a **hint, not truth**: on every wake (and on startup) we re-hash and **merkle-diff against the last generation's stored tree** → catch anything missed, O(changed). PollWatcher fallback for un-observable dirs. |
| **Atomic-save-via-rename (vim/Sublime/Kate/`sed -i`)** | Looks like "target deleted then a temp appeared". | Merkle diff sees only the **net content change** of the final file; temp-file create/rename churn is absorbed; `.tmp`/swap patterns in the ignore set. |
| **Embedding model is weak (today's logits proxy)** | Vector recall is poor. | **By design**: lexical+symbol carry recall; embeddings re-rank/expand at low RRF weight. A config dial raises vector weight when the **dedicated code role** lands — no re-index (vectors carry `embed_model_id`). |
| **Embedding model upgrade** | Old vectors incompatible. | **Mixed-model operation**: tag vectors with `embed_model_id`; search works across models; **lazy re-embed on idle GPU**, changed/queried chunks first; converge in background. No flush. |
| **LSP server crash / slow warm-up / missing capability** | Type/precise nav unavailable or wrong-while-warming. | Tier-0/1 (tags + stack-graphs) **always answer**; LSP only *sharpens* asynchronously. Capability probe → per-server degradation. Supervisor restarts crashed servers with backoff; `precise=true` queries wait for or skip unconfirmed edges. |
| **Crash mid-commit** | Partial/corrupt index. | Append-only segments + **atomic manifest swap** at a generation; recovery truncates torn tail, loads last good generation, merkle-diffs to catch in-flight changes. **Never corrupt.** |
| **WAL checkpoint starvation (always-on readers)** | `-wal` grows unbounded. | Keep read txns **short**; writer issues periodic `wal_checkpoint(TRUNCATE)`; `synchronous=NORMAL`. |
| **HNSW deletion under churn (if usearch chosen)** | Tombstones degrade the graph. | Prefer **Lance** (soft-delete + `optimize()` compaction) as primary; if usearch, schedule periodic **shard rebuilds** on idle. **sqlite-vec** co-store gives exact free deletes as a correctness anchor. |
| **Background indexing contends with decode/edit** | Stutter. | Idle+GPU-free gating; all background GPU work is **preemptible** — a decode request or edit immediately yields the Metal queue and re-prioritizes foreground. Merge/compaction throttled (never starves the editor). |
| **Symbol-ID instability (overloads, anonymous fns, macros)** | Refs fail to resolve / churn. | SCIP descriptor disambiguation (`().` arity, enclosing path, `:` meta) + stack-graph scoping; LSP overlay corrects; unresolved refs kept as *approximate* edges (provenance=tags) rather than dropped. |
| **Pathological graph queries (deep/cyclic closures)** | Runaway recursion / exponential path enumeration. | Recursive CTEs use **UNION** (cycle-break) + **depth cap** + `LIMIT`; in-memory traversals use SCC collapse (petgraph `tarjan_scc`) and **bidirectional BFS** with a meeting-level check + path-count cap; budgets on every traversal. |
| **Disk pressure (CAS/segments/history grow)** | Out of space. | Mark-and-sweep GC of unreachable CAS blobs (grace period) + segment merge reclaiming tombstones + history-depth cap + per-workspace size budget with LRU shard eviction (re-buildable on demand). |

---

## 7. Extensibility & plugin points

The index is built to absorb new languages and analyzers without core changes:

1. **New language = a grammar bundle.** Register `{ tree-sitter Language, tags.scm, locals.scm, chunk.scm, optional stack_graph.tsg, optional LspServerSpec }`. Tier-0 (tags) works immediately; tier-1 (stack-graphs) if a `.tsg` is provided; tier-2 (LSP) if a server spec is provided. No code in the core pipeline changes — it's data-driven off the `GrammarRegistry`. Grammars are statically linked for the core set and **dynamically loadable** (compiled `.so`/`.dylib` parsers) for the long tail.
2. **New analyzer = an edge producer.** Any analyzer that emits `Edge`s with a new `EdgeKind` and a `provenance` tag plugs into the unified graph (`graphs/mod.rs`) and is queryable via the generic graph API. Example future producers: an ownership/lifetime analyzer (Rust), an effect/exception graph, a security-rule pack feeding the CPG.
3. **New retrieval leg.** The hybrid retriever (`semantic/mod.rs`) fuses an arbitrary set of legs via RRF; adding a leg (e.g. a docstring-only index, a commit-message index, a "symbol-by-example" structural search) is a new leg with a weight.
4. **New embedding / reranker role.** Behind the `/v1/embeddings?role=…` and rerank HTTP seam — swap the model, bump `embed_model_id`, lazy re-embed. The dedicated **code embedding role** is the first planned addition; a **cross-encoder rerank role** the second.
5. **New vector engine.** The `VectorStore` trait abstracts Lance/usearch/sqlite-vec; a new ANN engine implements the trait.
6. **External tool ingestion.** Coverage parsers (lcov/coverage.py/llvm-cov), profilers (Instruments/perf/`gpu_us` traces), and SCIP/LSIF importers are ingestion plugins that emit occurrences/edges — letting us absorb the broader ecosystem's outputs.
7. **Query-API stability.** The `Index` trait (§4.11) is the versioned contract; new methods are additive. ch.02/03/04 bind to the trait, not the storage.

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by **(impact / difficulty)**; each tagged PROVEN-component vs SPECULATIVE-assembly.

1. **Local-LLM listwise re-rank as the default reranker.** *(Impact HIGH / Difficulty LOW. PROVEN technique, Hawking-native because the model is local & free.)* Use the hosted model for the precision pass over fused top-N. Free, private, no cloud reranker dependency. **Do this first.**
2. **Chunk-hash-cached, idle-GPU re-embedding with model versioning.** *(HIGH / LOW–MED. PROVEN — Cursor's chunk-hash cache + our idle-GPU and role versioning.)* Makes embedding upgrades free over time and keeps re-index near-zero-cost. Core to the living-index promise.
3. **Persistent, warm-started, signal-driven PageRank repo-map.** *(HIGH / MED. PROVEN algorithm; our incremental persistence + open-tab/cursor/perf signals are the upgrade.)* The structural heart of ch.04's context.
4. **Performance-aware retrieval (perf-map-boosted ranking).** *(HIGH / MED. SPECULATIVE assembly of PROVEN parts.)* We already capture `gpu_us`/dispatch traces; feeding measured hotness into ranking when the task is optimization is a moat no cloud tool has (it lacks your hardware's profiles).
5. **Stack-graph incremental precise nav for the core languages.** *(HIGH / MED–HIGH. PROVEN at GitHub scale.)* The right incremental model for live editing; upgrades approximate tags to scoped resolution without a compiler. Gated on per-language `.tsg` authoring.
6. **On-demand scoped CPG for dataflow/security over private code.** *(MED–HIGH for security tasks / HIGH. PROVEN (Joern); our scoping + caching make it interactive.)* Interactive taint/dataflow on code you'd never upload to a cloud.
7. **A dedicated, served code-embedding role (`/v1/embeddings?role=code`).** *(HIGH / HIGH. PROVEN models exist (Qodo/SFR/nomic/CodeSage); serving one as a role is the work.)* Graduates Leg C from re-rank to first-class recall. The single biggest *retrieval-quality* unlock; deferred behind the model layer per scope.
8. **Whole-git-history symbol index (time-travel nav).** *(MED / MED. SPECULATIVE assembly.)* `symbol_history`/`blame_age`/cross-rev `changed_since` without per-query `git`. Idle-built, depth-capped.
9. **Verified-streaming (bao-tree) packed artifacts for cross-machine index sync.** *(MED / MED–HIGH. PROVEN (BLAKE3/bao).)* Sync the index across a user's machines by content hash, verifying sub-ranges without full transfer. Only valuable multi-device; lower priority for single-machine.
10. **Embedding-of-the-graph (structure-aware retrieval).** *(MED / HIGH. SPECULATIVE.)* Embed not just chunk text but graph-context (neighbors, call paths) so semantic search is structure-aware. Research-grade; revisit after the code role lands.
11. **Adopt Tantivy for the lexical leg + Tantivy's segment/MVCC machinery wholesale.** *(MED / MED. PROVEN.)* Rather than re-implement immutable-segments+manifest+tombstones+merges, lean on Tantivy (backs Quickwit/ParadeDB) for the lexical leg and crash-safe spine. A pragmatic moonshot: less novel code, more reliability. Evaluate vs the bespoke segment store in §4.10.

---

## 9. Open questions & dials

**Dials (config surface, sane defaults):**
- `embed.role` = `logits-proxy` (default today) | `code` (when served). `search.leg_weights = {symbol:1.0, lexical:1.0, vector:0.3}` (vector ramps with role quality).
- `search.rerank` = `local_llm` (default) | `cross_encoder` | `none`. `search.k_retrieve=50`, `search.k_final=10`.
- `repomap.signals` weights: `{chat:50, cursor_file:80, open_tab:20, mentioned_ident:10, recency_halflife: 7d}`.
- `daemon.debounce_ms=200`; `daemon.idle_threshold_s`; `daemon.gpu_yield=true` (preempt on decode).
- `vectors.engine` = `lance` (default) | `usearch` | `sqlite_vec`. `vectors.quant` = `none|sq_int8|pq|binary`.
- `history.depth` = `working_copy` (default) | `N commits` | `full`. `index.generated = exclude` (default) | `include`.
- `chunk.max_tokens` (per embedding model); `huge_file.lexical_only_above` (lines/bytes).
- `lsp.enabled` per-language; `lsp.idle_reap_s`. `cpg.budget` (max functions/depth/wall-clock).

**Open questions (decide during build):**
1. **Tantivy vs bespoke segment store** for the lexical leg (§4.10/§8.11) — adopt the proven implementation, or keep full control? Lean: adopt Tantivy unless its model conflicts with the unified graph store.
2. **CozoDB vs SQLite+petgraph+materialized-closure** for graph reasoning — push recursion/incremental-materialization into Cozo, or keep the SQLite+in-memory hybrid? Lean: start SQLite+petgraph (fewer deps), adopt Cozo if closure maintenance becomes the bottleneck.
3. **Lance vs usearch** as the primary vector engine — Lance's soft-delete+optimize churn model vs usearch's NEON+mmap latency with periodic rebuilds. Lean: Lance primary, usearch behind the trait for latency-critical shards. *(Needs a local benchmark: measure both on a 1M-chunk index on the target Mac, including the exact random-4K SSD read latency — no citable M-series µs figure exists.)*
4. **When does Leg C (vector) graduate to first-class recall?** A measurable gate: on a held-out code-retrieval set, vector recall@k must exceed lexical+symbol recall@k by a margin before raising its RRF weight past parity. Until then, re-rank only.
5. **Per-file scope-graph languages** — which languages justify authoring a `.tsg` (cost) vs riding tags+LSP? Lean: the core set (Rust, TS/JS, Python, Go) first.
6. **History ingest scope** — full DAG (disk/compute cost) vs last-N. Lean: working-copy default, idle-built depth-N opt-in.
7. **Cross-repo symbol-ID collisions** — the `@wc` vs `@rev` discipline and package qualification must guarantee global uniqueness; needs a precise spec per language's package model.
8. **Generated-code detection precision** — heuristics (path, `@generated`, churn) will misfire; what's the false-positive cost and the override UX?

---

## 10. Cross-references

- **ch.04 · Context Compiler** — the primary consumer. Binds to `Index::repo_map(signals, budget)` (structural leg) and `Index::search(query, opts)` (semantic leg); interleaves+dedups `RankedSymbol`s and `Hit`s, expands via `find_definition` byte-ranges. **This chapter produces the ranked, budget-able structural + semantic context; ch.04 owns the total budget and final assembly.** The token-budget contract (§4.6, §4.11) is the seam.
- **ch.02 · Agent** — uses the index to scope edits and predict blast radius (`dependents_of`, `transitive_callers` reverse graph), verify completeness, and carry cross-session code knowledge (complements ch.02's memory). The agent's planning loop reads `freshness`/`await_generation` to avoid acting on stale views.
- **ch.03 · Tools** — `find_definition/references/callers/implementations`, `path_between`, `tests_covering`, `changed_since`, `grep_symbol`, `dataflow_paths` are **thin wrappers over the §4.11 query API**. Tools must never re-parse or re-walk the FS; they query the Living Index. The mapping table is in §4.11.
- **Runtime (`hawking-serve`)** — the model layer is a stable localhost surface: `POST /v1/embeddings` today (logits proxy), `POST /v1/embeddings?role=code` and a rerank role in future, behind the HTTP seam (`semantic/embed.rs`, `semantic/rerank.rs`). The index never couples to model internals.
- **Hawking family** — shares the determinism/content-addressing ethos with *Hawking Condense*; BLAKE3-keyed CAS and reproducible artifacts are the family value applied to the index.

---

*End of Chapter 05. The Living Index is the standing organ that makes every other subsystem smarter; it is the clearest expression of the local-first moat — a representation no cloud can afford to keep this fresh, this complete, or this private.*
