# Chapter 08 · Research & Knowledge Lab

> **Purpose (one line):** Turn HIDE from a coding environment into a *research laboratory* — a local-first engine that runs overnight multi-source literature sweeps for free, ingests papers/datasets/repos into a persistent private knowledge graph, and fuses research with code so findings become issues, equations become functions, and citations become queryable structure.

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the Art + Limits (cited)](#3-state-of-the-art--limits-cited)
4. [The Hawking Design](#4-the-hawking-design)
   - 4.1 [Module Layout](#41-module-layout)
   - 4.2 [The Knowledge Graph: Schema](#42-the-knowledge-graph-schema)
   - 4.3 [Storage Substrate & Provenance](#43-storage-substrate--provenance)
   - 4.4 [Ingestion: Adapter Interface](#44-ingestion-the-adapter-interface)
   - 4.5 [Ingestion: PDF / Equation / Table / OCR Pipeline](#45-ingestion-pdf--equation--table--ocr-pipeline)
   - 4.6 [The Research Pipeline: State Machine](#46-the-research-pipeline-state-machine)
   - 4.7 [Source-Quality Scoring & Adversarial Verification](#47-source-quality-scoring--adversarial-verification)
   - 4.8 [GraphRAG-Style Incremental Build & Query](#48-graphrag-style-incremental-build--query)
   - 4.9 [Citation / Literature Mapping Workflows](#49-citation--literature-mapping-workflows)
   - 4.10 [Research → Issues → Code Pipeline](#410-research--issues--code-pipeline)
   - 4.11 [Experiment Planning & Tracking](#411-experiment-planning--tracking)
   - 4.12 [Note-Taking & Synthesis Surfaces](#412-note-taking--synthesis-surfaces)
   - 4.13 [Integration with ch.04 Memory & ch.05 Code Index](#413-integration-with-ch04-memory--ch05-code-index)
   - 4.14 [The Research Tab (post-shell UI)](#414-the-research-tab-post-shell-ui)
5. [How We Exceed ("cloud literally cannot do this")](#5-how-we-exceed-cloud-literally-cannot-do-this)
6. [Failure Modes + Mitigations](#6-failure-modes--mitigations)
7. [Extensibility (new sources / ingestors)](#7-extensibility-new-sources--ingestors)
8. [Bleeding-Edge / Moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open Questions / Dials](#9-open-questions--dials)
10. [Cross-References](#10-cross-references)

---

## 1. Purpose & Scope

Every cloud coding agent (Claude Code, Cursor, Copilot Workspace) treats the web as a *transient* lookup: search, read a snippet, discard. None of them remember what you read last week, none of them will spend eight unattended hours reading 200 papers for you, and none of them keep a private graph of your own field that gets denser every night. The economic reason is structural: in a per-token, per-seat cloud business, *unattended depth is a cost center*. HIDE inverts this. On the user's own Apple Silicon machine, compute the user already paid for is **free at the margin**, so a research run that would cost $40 of API tokens in the cloud costs $0 of electricity overnight on HIDE.

This chapter specifies the **Research & Knowledge Lab** subsystem of HIDE: the second pillar (after coding) that makes HIDE a category of one — a *coding IDE that is also a research laboratory*.

**In scope:**

- The **multi-source research pipeline**: fan-out web search → fetch → read → adversarial verification → cited synthesis, with dedup and source-quality scoring (builds on the `deep-research` harness pattern HIDE already ships).
- **Ingestion**: PDF, arXiv, HTML, repo, dataset → parsed structured documents; equation, table, and figure extraction; OCR for scanned material.
- The persistent **Knowledge Graph** (KG): entity & edge schema, storage, provenance, incremental construction, and GraphRAG-style query.
- **Citation / literature mapping**: "compare N papers", "build a literature map", citation-graph traversal.
- The **Research → Issues → Code** pipeline: findings become repo issues/tasks/experiments; equations become typed code stubs.
- **Experiment planning & tracking**: hypotheses, runs, results, reproducibility ledger.
- **Note-taking & synthesis** surfaces (zettelkasten / canvas).
- Concrete **integration contracts** with ch.04 (memory substrate) and ch.05 (code index).

**Out of scope (owned elsewhere):**

- Raw web *fetch/search/browser* primitives and their sandboxing → **ch.03**. This chapter *consumes* them; it does not re-specify HTTP, headless-browser, or robots policy.
- The long-term **memory** store, embedding index internals, and recall ranking → **ch.04**. We define how research *writes into* memory and *reads from* it.
- The **code index** (symbol graph, repo embeddings) → **ch.05**. We define how research links *to* code symbols and how equations land in the repo.
- The app **shell**, window management, command palette, settings → **ch.01/02**. The Research Tab UI here is specified fully but flagged **post-shell** (a later panel; ships after the core editor + agent shell are stable).
- The local **runtime** (chat + embeddings server) → existing Hawking runtime. We treat its HTTP surface as a fixed contract (see §4.13 and §10).

**Scoping flags (honoring the bible's discipline):**

- 🏗️ **POST-SHELL**: the Research Tab UI (§4.14) and any panel chrome ship *after* the app shell. The *engine* (pipeline, KG, ingestion) can be built and exercised headless (CLI + tests) before any UI exists.
- ⏸️ **HF-DEFERRED**: anything requiring Hugging Face model downloads (e.g., a local Nougat/GROBID-replacement vision model for PDF parsing) is deferred; v1 uses CPU-only deterministic parsers (PDFium text layer, rule-based table/equation extraction) and the *already-running local runtime* for embeddings/synthesis. Vision-model ingestion is a §8 moonshot.
- 🧪 **RUNTIME-TESTING**: `.tq` quantization and 32B-class models are a *runtime* concern (Hawking Condense). The Research Lab is **model-agnostic**: it calls the local runtime's OpenAI-compatible endpoints and does not care whether a 0.5B or a condensed 32B is behind them. Larger local models simply make synthesis better.

---

## 2. Tenets

1. **Local-first, private by construction.** Local PDFs, datasets, and notes are *never* uploaded. The KG lives on disk under the user's control. Network egress happens only through ch.03's audited fetch/search path, only for *public* sources, and every egress is logged. A research run can be executed fully offline against already-ingested material.
2. **Provenance is not optional — it is the data model.** Every claim, every node, every synthesized sentence carries a back-pointer to its source span (document + char offset, or URL + retrieval timestamp + content hash). A finding you cannot trace to a source does not exist in the graph. This is the antidote to hallucinated citations (§6).
3. **Adversarial by default.** Synthesis never trusts a single source. Claims are cross-checked across independent sources; contradictions are *first-class graph edges* (`refutes`, `contradicts`), not errors to suppress. The system is designed to *surface disagreement*, not paper over it.
4. **The graph is the moat; text is the cache.** Reports, summaries, and notes are *derived artifacts* — regenerable from the graph. The durable, compounding asset is the structured, provenance-rich, deduplicated knowledge graph that grows every night. Lose a report → regenerate it. Lose the graph → lose years.
5. **Research and code share one substrate.** A paper's claim can link to the function that implements it; an equation extracted from a PDF can be turned into a typed code stub in the repo; a research finding can become a tracked issue. The KG and the code index (ch.05) are *joined*, not siloed.
6. **Free overnight depth.** The pipeline is built to run unattended for hours: checkpointed, resumable, budget-bounded by *wall-clock and disk*, not by token cost. The default posture is "go deep" because depth is free.
7. **Determinism where it matters.** Ingestion (parse → structured doc) is deterministic and content-addressed: re-ingesting the same bytes yields the same nodes with the same IDs. Synthesis (LLM) is non-deterministic but *seeded and logged* so a report can be re-derived and diffed.
8. **Incremental, never rebuild-the-world.** New sources extend the graph; they do not trigger a full re-embed or re-extract. Entity resolution, edge insertion, and community detection are *streaming/online* operations (§4.8).
9. **Decisive defaults, exposed dials.** Every fan-out width, quality threshold, and dedup radius has a chosen default (§9). Power users can turn dials; nobody *has* to.
10. **Honest epistemics.** The system distinguishes *measured* (extracted from a source) from *inferred* (LLM-synthesized) from *speculative* (the user's hypothesis). These are different node `confidence` tiers and are rendered differently.

---

## 3. State of the Art + Limits (cited)

> Tagging convention: **[PROVEN]** = shipped & widely reproduced; **[EMERGING]** = published & demonstrated but not yet commodity; **[SPECULATIVE]** = our extrapolation / not yet demonstrated at this integration. Citations are to public systems/papers as of 2025–2026.

### 3.1 Deep-research agents

- **OpenAI Deep Research** (2025), **Google Gemini Deep Research** (2024–2025), **Perplexity Deep Research** (2025): agentic loops that plan → fan-out search → read → synthesize a cited report over many minutes. **[PROVEN]** that the *pattern* (iterative search-read-synthesize with citations) produces strong long-form research.
  - **Limits we exploit:** (a) **per-token cost caps depth** — these run for minutes, not a supervised overnight; (b) **no persistent memory** — each report starts cold, nothing accretes; (c) **public-web only / no local corpus** — your private PDFs and datasets are invisible; (d) **citations are surface-level** — a flat reference list, not a queryable graph with `refutes`/`implements` edges; (e) **no code integration** — findings cannot become issues or functions in *your* repo.
- HIDE already ships a `deep-research` skill (fan-out search → fetch → adversarial verify → cited report). This chapter generalizes it from "produce one report" to "feed a persistent graph + drive code."

### 3.2 GraphRAG & knowledge-graph construction

- **Microsoft GraphRAG** (2024): LLM extracts entities + relationships from a corpus, builds a graph, runs hierarchical community detection (Leiden), generates community summaries, and answers *global* questions by map-reduce over communities. **[PROVEN]** that graph + community summaries beat vanilla vector RAG on global/sensemaking queries.
  - **Limits:** expensive to build (full-corpus LLM extraction), batch-oriented (re-index to update), and *stateless across sessions* in typical deployments. HIDE makes it **incremental and persistent** (§4.8).
- **LightRAG** (2024), **nano-graphrag**, **HippoRAG / HippoRAG 2** (2024–2025, memory-indexing via personalized PageRank over a KG): **[EMERGING]** lighter-weight, dual-level (local+global) retrieval; HippoRAG frames the KG as long-term memory with associative recall — directly relevant to our ch.04 integration.
- **Property-graph stores**: Neo4j, KùzuDB (embedded, Cypher, columnar, single-file — *ideal for local-first*), Oxigraph (embedded RDF/SPARQL). **[PROVEN]** embeddable graph engines exist that need no server.

### 3.3 Scientific-paper ingestion

- **GROBID** (machine-learning extraction of bibliographic + structured full-text from scholarly PDFs → TEI XML): **[PROVEN]** the de-facto standard for header/reference/section parsing. Java service; deterministic enough; strong on references, weaker on complex tables/equations.
- **Nougat** (Meta, 2023 — visual transformer PDF → Markdown with LaTeX math): **[PROVEN]** for born-digital scientific PDFs, including inline/display math as LaTeX. **Limits:** GPU-hungry, hallucinates/repeats on out-of-distribution pages, slow per page. (🧪 HF-deferred in HIDE v1.)
- **Marker** (2024, PDF → Markdown, layout + tables + math, faster than Nougat), **PyMuPDF/PDFium** text-layer extraction, **pdfplumber/Camelot/Tabula** for tables: **[PROVEN]** deterministic CPU paths suitable for v1.
- **arXiv API** (bulk metadata + PDF + often LaTeX source), **Semantic Scholar API / S2ORC** (200M+ papers, abstracts, citation edges, embeddings via SPECTER2), **OpenAlex** (open catalog of works/authors/venues/concepts, full citation graph, generous API), **Crossref** (DOI metadata), **Unpaywall** (legal OA full-text links): **[PROVEN]** rich, free/low-cost metadata + citation-graph sources.
- **PDF understanding / tables+equations:** layout models (**LayoutLMv3**, **DocLayout-YOLO**), table-structure recognition (**TableFormer**, **PubTables-1M**), math OCR (**pix2tex/LaTeX-OCR**, **Texify**): **[EMERGING→PROVEN]** depending on component. (🧪 vision components HF-deferred.)

### 3.4 Citation graphs & literature mapping

- **Connected Papers** (graph by co-citation/bibliographic coupling similarity, not direct citation), **Litmaps**, **Research Rabbit**, **Inciteful**, **Open Knowledge Maps**: **[PROVEN]** interactive citation/similarity maps.
  - **Limits:** cloud-hosted, your private/unpublished work isn't in them, no link to your code, ephemeral (you can't *own* the map or grow it with notes). HIDE builds an equivalent **locally**, seeded by your own reading, joined to your code.

### 3.5 Literature-review automation & research assistants

- **Elicit** (find papers, extract data into tables, summarize, screen for systematic review), **SciSpace** (chat-with-PDF, extract, explain math), **Consensus** (claim-level evidence aggregation), **scite.ai** (Smart Citations: classifies citations as *supporting / mentioning / contrasting*): **[PROVEN]** that *claim-level* and *citation-stance* extraction is feasible and useful — directly motivates our `supports/refutes/mentions` edges (§4.2).
  - **Limits:** SaaS, per-seat, your corpus + extractions live on their servers, no code/experiment integration.

### 3.6 Experiment planning & tracking

- **Weights & Biases**, **MLflow**, **DVC**, **Aim**, **Sacred/Omniboard**: **[PROVEN]** experiment tracking (params, metrics, artifacts, lineage). Mostly ML-run-centric and cloud-leaning.
- **AI-Scientist** (Sakana, 2024 — end-to-end idea→experiment→paper), **DSPy/agent experiment loops**, **automated hypothesis generation**: **[EMERGING]** agentic experiment *planning*; quality of autonomous science is still uneven.
  - **Limits:** none of these are *joined to your literature graph* — the hypothesis that motivated a run isn't linked to the papers it came from or the code that ran it. HIDE closes that loop (§4.11).

### 3.7 Note-taking / zettelkasten / PKM

- **Obsidian** (Markdown + bidirectional `[[links]]` + local graph view + community plugins), **Logseq** (outliner, block-references), **Roam**, **Dendron**, **Zotero** (reference manager + PDF annotation + Better BibTeX): **[PROVEN]** local-first PKM with linked notes is a mature, beloved pattern.
  - **Limits:** the graph is *manual* (you make the links) and *disconnected from automated research and from code*. HIDE's KG is *auto-populated* by the research pipeline and *bridged* to a Markdown zettelkasten the user can also edit by hand — best of both.

### 3.8 The gap HIDE fills

No single system is simultaneously: (1) **local-first & private**, (2) **persistent & compounding** (a graph that grows nightly), (3) **multi-source + adversarial** (not a single-DB chat), (4) **deep & free** (overnight, no per-token tax), and (5) **joined to code** (research ⇄ issues ⇄ functions ⇄ experiments). Each of the above does *one or two* of these. HIDE does all five, and the local plane is precisely where the cloud is *structurally* unable to follow (§5).

---

## 4. The Hawking Design

### 4.1 Module Layout

The Research Lab is a set of cooperating crates under `crates/` (headless engine) plus a later UI panel. Names are chosen to match the existing `hawking-*` workspace convention.

```
crates/
  hawking-research/            # orchestrator: the research pipeline state machine (§4.6)
    src/
      pipeline.rs              # PlanScope → FanOut → Fetch → Read → Verify → Synthesize FSM
      planner.rs               # query decomposition → sub-questions, search-term expansion
      fanout.rs                # parallel search across providers (ch.03), budget governor
      verify.rs                # adversarial cross-checking, claim triangulation (§4.7)
      synthesize.rs            # cited-report generation; writes claims+edges to KG
      budget.rs                # wall-clock / disk / fetch-count governor; checkpointing
      run_ledger.rs            # append-only run journal (resumable, auditable)

  hawking-ingest/              # ingestion adapters + parsing (§4.4, §4.5)
    src/
      adapter.rs               # SourceAdapter trait (the ingestion interface)
      adapters/
        arxiv.rs   semantic_scholar.rs   openalex.rs   crossref.rs   unpaywall.rs
        pdf_local.rs   html.rs   repo.rs   dataset.rs   zotero.rs   bibtex.rs
      parse/
        pdf.rs                 # PDFium text-layer + structure (deterministic, CPU)
        grobid.rs              # optional GROBID client (TEI XML) when service present
        tables.rs              # rule + heuristic table extraction (pdfplumber-style)
        equations.rs           # display/inline math detection → LaTeX (regex+layout; pix2tex later)
        ocr.rs                 # Tesseract/Vision fallback for scanned PDFs (§4.5)
        normalize.rs           # → StructuredDoc canonical form (content-addressed)

  hawking-kg/                  # the knowledge graph: schema, store, query (§4.2–4.3, §4.8)
    src/
      schema.rs                # node/edge types, ID scheme, provenance records
      store.rs                 # embedded property-graph (KùzuDB) + blob CAS + vector handoff
      entity_resolution.rs     # incremental dedup / canonicalization (papers, authors, concepts)
      extract.rs               # LLM entity+relation extraction from StructuredDoc (GraphRAG-style)
      community.rs             # online community detection + summaries (Leiden/label-prop)
      query.rs                 # GraphRAG local+global query; Cypher passthrough; path queries
      provenance.rs            # span back-pointers, content hashes, retrieval receipts

  hawking-litmap/              # citation/literature mapping workflows (§4.9)
    src/ compare.rs  map.rs  coupling.rs  timeline.rs  gaps.rs

  hawking-experiments/         # hypotheses, runs, reproducibility ledger (§4.11)
    src/ hypothesis.rs  run.rs  ledger.rs  repro.rs  equation_to_code.rs

  hawking-bridge/              # research ⇄ code/issues/memory bridges (§4.10, §4.13)
    src/
      issues.rs                # Finding → Issue/Task mapping (§4.10)
      memory.rs                # KG → ch.04 memory writes; ch.04 → KG reads
      code_index.rs            # KG ↔ ch.05 code-index symbol links
      equations.rs             # equation node → typed code stub (§4.10)

  hawking-research-ui/         # 🏗️ POST-SHELL: the Research Tab panel (§4.14)
```

**Dependency direction (strict, acyclic):**

```
hawking-research ─┬─▶ hawking-ingest ─▶ (ch.03 fetch/search)
                  ├─▶ hawking-kg ─────▶ (embedded graph store + ch.04 vector index)
                  ├─▶ hawking-litmap ─▶ hawking-kg
                  └─▶ hawking-bridge ─▶ {hawking-kg, ch.04 memory, ch.05 code index}
hawking-experiments ─▶ {hawking-kg, hawking-bridge}
hawking-research-ui ─▶ everything above (read-mostly; commands go through the engine)
```

The engine layers (`research`, `ingest`, `kg`, `litmap`, `experiments`, `bridge`) are **headless and fully testable via CLI + integration tests** before any UI exists. This is what lets us build the lab now and skin it later.

**LLM/embedding access** is *always* indirected through the local runtime's HTTP surface (an injected `RuntimeClient` trait), never a hard dependency on a model:

```rust
/// The only way the Research Lab talks to a model. Backed by the local Hawking
/// runtime (OpenAI-compatible): POST /v1/chat/completions (SSE), POST /v1/embeddings,
/// and the native POST /v1/hawking/generate. Model-agnostic: 0.5B or condensed 32B.
pub trait RuntimeClient: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;          // /v1/embeddings
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse>;            // /v1/chat/completions
    async fn chat_stream(&self, req: ChatRequest) -> Result<TokenStream>;      // SSE
    fn model_id(&self) -> &str;                                                // for run provenance
    fn context_window(&self) -> usize;                                         // chunk sizing
}
```

> **Why a trait, not a direct call:** it (a) lets tests run with a deterministic mock model, (b) lets the lab run fully offline against cached embeddings, and (c) means a runtime upgrade (bigger condensed model) is a config change, not a code change.

---

### 4.2 The Knowledge Graph: Schema

The KG is a **property graph** (typed nodes + typed edges, both with properties), chosen over pure-RDF because property graphs are friendlier for the heterogeneous, provenance-heavy, weighted edges we need, and embeddable engines (KùzuDB) speak Cypher. (We can still *export* to RDF/SPARQL for interop — §7.)

#### 4.2.1 Identity & content-addressing

Every node has a stable **`id`**:

- **Canonical entities** (Paper, Author, Venue, Dataset, Concept, Method) get IDs from authoritative external IDs when available, normalized: `paper:doi:10.1145/3534678`, `paper:arxiv:2404.16130`, `author:orcid:0000-0002-...`, `concept:wikidata:Q11660`, else a deterministic hash of canonical fields (`author:sha256(name|affiliation_norm)`).
- **Content-derived nodes** (StructuredDoc, Chunk, Claim, Equation, Table, Figure) get **content-addressed** IDs: `chunk:sha256(normalized_text)`, `claim:sha256(canonical_claim_text|paper_id)`. Re-ingesting identical bytes → identical IDs → idempotent ingestion.
- **User nodes** (Note, Hypothesis, ExperimentRun, Finding, Report) get ULIDs (sortable, time-ordered): `note:01J...`.

#### 4.2.2 Node types

| Node | Key properties | Notes |
|---|---|---|
| **Source** | `kind` (arxiv/web/pdf/repo/dataset), `uri`, `retrieved_at`, `content_hash`, `mime`, `license`, `paywalled:bool` | The *raw* origin. One Source → one or more StructuredDocs. |
| **StructuredDoc** | `title`, `lang`, `format`, `n_pages`, `parser`, `parser_version`, `parse_confidence` | Canonical parsed form of a Source (§4.5). |
| **Chunk** | `text`, `section_path` (e.g. `["3","3.2"]`), `char_span`, `page`, `embedding_ref` | Retrieval unit. `embedding_ref` points into ch.04's vector index. |
| **Paper** | `title`, `abstract`, `year`, `venue`, `doi`, `arxiv_id`, `s2_id`, `oa_status`, `n_citations` | Bibliographic entity; may exist *before* full text is ingested (metadata-only). |
| **Author** | `name`, `orcid`, `affiliations[]`, `h_index?` | Resolved across papers (§4.8 entity resolution). |
| **Venue** | `name`, `type` (journal/conf/preprint), `issn`, `rank?` | |
| **Concept** | `label`, `aliases[]`, `wikidata?`, `definition`, `definition_provenance` | A topic/term. Hierarchical via `broader/narrower` edges. |
| **Method** | `label`, `aliases[]`, `description` | A technique/algorithm/architecture (e.g., "Leiden community detection"). |
| **Dataset** | `label`, `uri`, `modality`, `size`, `license`, `splits?` | |
| **Claim** | `text` (canonical statement), `polarity` (assert/negate), `confidence_tier` (measured/inferred/speculative), `quantitative?` (value+unit+CI) | The atom of knowledge. Always provenance-linked. |
| **Equation** | `latex`, `symbols[]` (with descriptions), `context_chunk`, `numbered_as?` | Extractable to code (§4.10). |
| **Table** | `caption`, `grid` (rows×cols), `units`, `source_chunk` | |
| **Figure** | `caption`, `image_ref`, `kind` (plot/diagram/photo) | |
| **CodeSymbol** | `repo`, `path`, `symbol`, `kind`, `index_ref` | **Mirror/handle** into ch.05's code index (not a copy — a join key). |
| **Note** | `title`, `markdown`, `tags[]`, `author=user`, `created/updated` | User zettel; bridged to Markdown files (§4.12). |
| **Hypothesis** | `statement`, `rationale`, `status` (open/supported/refuted/inconclusive), `confidence` | Drives experiments (§4.11). |
| **ExperimentRun** | `cmd`, `params`, `env_hash`, `code_rev`, `metrics`, `artifacts[]`, `started/ended`, `seed`, `status` | Reproducibility ledger row. |
| **Finding** | `summary`, `importance`, `actionable:bool` | Output of a research run; can mint Issues (§4.10). |
| **Report** | `title`, `markdown_ref`, `run_id`, `query`, `generated_at`, `model_id` | Derived artifact; regenerable. |
| **ResearchRun** | `query`, `scope`, `budget`, `status`, `started/ended`, `sources_seen`, `seed` | The pipeline invocation (§4.6). |

> **Confidence tiers** (`Claim.confidence_tier`, `Hypothesis.confidence`) are load-bearing per Tenet 10:
> `measured` = a number/fact extracted verbatim from a source; `inferred` = LLM synthesis combining sources; `speculative` = user hypothesis or model conjecture not yet sourced. Rendered with distinct affordances (§4.14).

#### 4.2.3 Edge types

Edges are **typed, directed, and carry properties** (at minimum `provenance_ref` and `weight`/`confidence`).

| Edge | From → To | Key properties | Meaning |
|---|---|---|---|
| `CITES` | Paper → Paper | `stance` (support/mention/contrast), `context_chunk` | Direct citation; `stance` à la scite Smart Citations. |
| `CO_CITED_WITH` | Paper ↔ Paper | `strength` | Derived (two papers cited together) — for Connected-Papers-style maps. |
| `BIB_COUPLED_WITH` | Paper ↔ Paper | `strength` | Derived (share references) — similarity without direct citation. |
| `AUTHORED_BY` | Paper → Author | `position`, `corresponding:bool` | |
| `PUBLISHED_IN` | Paper → Venue | `year` | |
| `MENTIONS` | StructuredDoc/Chunk → {Concept,Method,Dataset} | `salience`, `char_span` | Entity linking. |
| `INTRODUCES` | Paper → {Concept,Method,Dataset} | | This paper is the *origin* of X. |
| `USES` | Paper → {Method,Dataset} | | |
| `EVALUATES_ON` | Paper → Dataset | `metric`, `score` | |
| `MAKES_CLAIM` | Paper/Chunk → Claim | `char_span` | Provenance of a claim. |
| `SUPPORTS` | {Paper,Claim,ExperimentRun} → Claim | `weight`, `provenance_ref` | Evidence for. |
| `REFUTES` | {Paper,Claim,ExperimentRun} → Claim | `weight`, `provenance_ref` | Evidence against (first-class!). |
| `CONTRADICTS` | Claim ↔ Claim | `detected_by` (verifier/user) | Two claims in tension. |
| `DERIVES_FROM` | {Equation,Method,Concept} → {Equation,Method,Concept} | | Derivation/specialization lineage. |
| `IMPLEMENTS` | CodeSymbol → {Method,Equation,Paper} | `fidelity` (exact/approx/partial), `provenance_ref` | **The research⇄code join.** |
| `DEFINED_BY` | Equation → Chunk | | |
| `BROADER`/`NARROWER` | Concept ↔ Concept | | Concept hierarchy. |
| `SIMILAR_TO` | any ↔ any (typed) | `cosine`, `index_ref` | Embedding-space neighbor (from ch.04). |
| `TESTS` | ExperimentRun → Hypothesis | | |
| `MOTIVATED_BY` | Hypothesis → {Paper,Claim,Finding} | | Where the idea came from. |
| `PRODUCED` | ResearchRun → {Finding,Report,Claim,…} | | Run lineage. |
| `MINTED_ISSUE` | Finding → Issue(external/ch.05) | `issue_ref` | Research→issues link (§4.10). |
| `ANNOTATES` | Note → any | | User note attached to any node. |
| `IN_COMMUNITY` | any → Community | `level` | Hierarchical community membership (§4.8). |

#### 4.2.4 Provenance record (attached to nodes *and* edges)

```rust
/// Every fact carries one of these. No provenance → not in the graph (Tenet 2).
pub struct Provenance {
    pub source_id: NodeId,            // the Source/StructuredDoc this came from
    pub locator: Locator,            // exact span / page / URL fragment
    pub content_hash: [u8; 32],      // hash of the *evidence bytes* (immutable receipt)
    pub retrieved_at: DateTime<Utc>, // when the source was fetched/parsed
    pub extractor: ExtractorId,      // parser/LLM that produced this (name + version)
    pub method: ExtractMethod,       // Deterministic | Llm { model_id, prompt_hash, seed } | UserEntered
    pub confidence: f32,             // extractor-reported confidence
}
pub enum Locator {
    CharSpan { start: usize, end: usize, page: Option<u32> },
    Url { url: String, fragment: Option<String> },
    TableCell { table_id: NodeId, row: u32, col: u32 },
    CodeRange { index_ref: String, start_line: u32, end_line: u32 },
}
```

> The `content_hash` of evidence bytes is what makes a citation *verifiable*: if a synthesized sentence claims "[12] reports 73% accuracy", the system can re-open span `[12]` and check that the bytes still hash-match and still contain "73%". A claim that fails this check is flagged (§6, hallucinated-citation guard).

---

### 4.3 Storage Substrate & Provenance

Three co-located stores under a single `~/.hawking/research/` root (path under ch.04's data-dir umbrella):

```
~/.hawking/research/
  graph.kuzu/            # KùzuDB embedded property graph (nodes+edges+properties+indexes)
  cas/                   # content-addressed blob store: raw PDFs, parsed JSON, figure images
    ab/cd/abcd…ef.pdf    #   sharded by hash prefix; immutable; deduplicated by content
  vectors/               # handoff to ch.04 vector index (embeddings live there; we store refs)
  runs/                  # append-only run ledgers (JSONL) — resumable, auditable (§4.6)
  notes/                 # Markdown zettelkasten mirror (bidirectional with Note nodes, §4.12)
  index.sqlite           # lightweight metadata/FTS sidecar (fast filters, full-text fallback)
```

**Why this split:**

- **KùzuDB** — embedded, single-process, Cypher, columnar, fast multi-hop traversal, no server to run. **[PROVEN]** as an embeddable graph engine; the right local-first choice. (Decisive default; Neo4j-embedded or Oxigraph are §7 swap-ins behind the `store.rs` trait.)
- **CAS (content-addressed store)** — raw bytes (PDF, HTML snapshot, parsed JSON) keyed by SHA-256. Guarantees: (a) **dedup** (same paper fetched twice = one blob), (b) **immutability** (a citation's evidence cannot silently change), (c) **offline replay** (everything needed to re-derive a report is on disk).
- **Vectors live in ch.04's index, not here.** We store only `embedding_ref` keys (Tenet 5 + clean ownership: ch.04 owns embedding lifecycle, ANN params, recall). This avoids two competing vector stores. See §4.13 for the contract.
- **Run ledgers** are append-only JSONL (one event per line) so a crashed overnight run resumes from the last checkpoint without re-fetching (§4.6).
- **Notes as Markdown files** so the user keeps an Obsidian-compatible vault they own, while the graph stays in sync (§4.12).

**Provenance storage:** `Provenance` records are stored as edge/node properties in Kùzu *and* the evidence bytes are pinned in CAS by `content_hash`. The pair (graph property + immutable blob) is the durable provenance.

**Backups & portability:** the entire `research/` dir is a self-contained, copyable folder. `kuzu export` produces portable CSV/Parquet for the graph; CAS is just files. There is **no cloud dependency** for any of it (Tenet 1).

**Encryption (dial):** at-rest encryption of `cas/` and `graph.kuzu/` via OS keychain-held key is an opt-in dial (§9) for users with sensitive private corpora.

---

### 4.4 Ingestion: The Adapter Interface

All sources enter through one trait. Adding a new source = implementing `SourceAdapter` + registering it. This is the single most important extensibility seam (§7).

```rust
/// Implement once per source kind (arXiv, local PDF, HTML page, repo, dataset, …).
/// The pipeline never special-cases a source — it only knows this trait.
#[async_trait]
pub trait SourceAdapter: Send + Sync {
    /// Stable id, e.g. "arxiv", "pdf_local", "openalex". Used in provenance + config.
    fn kind(&self) -> &'static str;

    /// Can this adapter handle the given reference? (URI scheme, file ext, host, DOI prefix…)
    fn can_handle(&self, reference: &SourceRef) -> bool;

    /// Resolve a reference to one or more concrete, fetchable sources WITHOUT downloading
    /// bodies yet (cheap: metadata + canonical URI + dedup key). Enables fan-out planning
    /// and dedup before paying fetch cost.
    async fn resolve(&self, reference: &SourceRef, ctx: &IngestCtx)
        -> Result<Vec<ResolvedSource>>;

    /// Fetch raw bytes (through ch.03's audited fetch — adapters MUST NOT open sockets
    /// directly). Returns bytes + headers + the content hash. Respects robots/ToS/paywall
    /// policy enforced by ch.03. Idempotent: if CAS already has the hash, returns cached.
    async fn fetch(&self, src: &ResolvedSource, ctx: &IngestCtx) -> Result<RawArtifact>;

    /// Parse raw bytes into the canonical StructuredDoc (deterministic where possible).
    /// May delegate to parse/ modules (PDF, GROBID, tables, equations, OCR — §4.5).
    async fn parse(&self, raw: &RawArtifact, ctx: &IngestCtx) -> Result<StructuredDoc>;

    /// Optional: emit bibliographic/citation metadata edges cheaply from an API
    /// (e.g., OpenAlex citation list) WITHOUT full-text parse — enables metadata-only
    /// Paper nodes + citation graph before any PDF is read.
    async fn metadata(&self, src: &ResolvedSource, ctx: &IngestCtx)
        -> Result<Option<BiblioMetadata>> { Ok(None) }
}

pub struct IngestCtx<'a> {
    pub fetch: &'a dyn FetchService,   // ch.03 audited fetch/search/browser
    pub cas:   &'a dyn BlobStore,      // content-addressed store (dedup, cache)
    pub runtime: &'a dyn RuntimeClient,// local model (for parse-time embedding/cleanup)
    pub budget: &'a BudgetGuard,       // wall-clock/disk/fetch governor (§4.6)
    pub policy: &'a EgressPolicy,      // robots, rate-limit, paywall, allowlist (ch.03)
}
```

`StructuredDoc` is the **canonical contract** every adapter must produce — the rest of the system (KG extraction, chunking, search) only ever sees this:

```rust
pub struct StructuredDoc {
    pub id: NodeId,                    // content-addressed (sha256 of normalized form)
    pub source_id: NodeId,
    pub title: Option<String>,
    pub lang: Lang,
    pub sections: Vec<Section>,        // tree: section_path + heading + paragraphs
    pub references: Vec<RawReference>, // parsed bibliography (→ Paper nodes + CITES edges)
    pub equations: Vec<RawEquation>,   // latex + symbol table + context (§4.5)
    pub tables: Vec<RawTable>,         // grid + caption + units
    pub figures: Vec<RawFigure>,       // caption + image blob ref
    pub metadata: BiblioMetadata,      // doi/arxiv/authors/venue/year (best-effort)
    pub parser: ExtractorId,           // name + version (provenance)
    pub parse_confidence: f32,         // 0..1; low → flagged for review (§6)
}
```

**Adapter registry & dispatch:**

```rust
pub struct AdapterRegistry { adapters: Vec<Box<dyn SourceAdapter>> }
impl AdapterRegistry {
    pub fn route(&self, r: &SourceRef) -> Option<&dyn SourceAdapter> {
        // first adapter whose can_handle() returns true; user can pin order via config
        self.adapters.iter().map(|a| a.as_ref()).find(|a| a.can_handle(r))
    }
}
```

**v1 adapter set (decisive default):** `arxiv`, `openalex` (citation graph + metadata), `semantic_scholar` (abstracts + SPECTER2 embeddings + citations), `crossref` (DOI metadata), `unpaywall` (legal OA full text), `pdf_local`, `html`, `repo` (clone/read a git repo → docs + README + code handles into ch.05), `dataset` (CSV/Parquet/HF-dataset *card* — 🧪 actual HF download deferred), `zotero` + `bibtex` (import an existing library). Each is independently testable with a recorded fixture.

---

### 4.5 Ingestion: PDF / Equation / Table / OCR Pipeline

PDF is the hard case; we specify it concretely. The pipeline is a **fallback cascade** — cheapest deterministic path first, escalate only when confidence is low.

```
parse_pdf(raw) :=
  1. CLASSIFY: born-digital (has text layer) vs scanned (image-only)?
       — PDFium: does page have extractable text covering > T_text of area?
  2. BORN-DIGITAL PATH (deterministic, CPU, default):
       a. PDFium text-layer extraction with positional boxes (x,y,w,h per glyph/word).
       b. LAYOUT: column detection + reading-order reconstruction (geometry-based:
          x-gaps → columns; y-order within column). Section headers via font-size/bold
          heuristics → section_path tree.
       c. REFERENCES: detect bibliography section; parse entries (regex + heuristics);
          if a GROBID service is configured, prefer its TEI reference parse (better).
       d. EQUATIONS: detect display math (isolated lines, math fonts, numbering "(3)")
          and inline math; reconstruct LaTeX (symbol-by-symbol from font/position;
          v1 = best-effort; 🧪 pix2tex/Texify vision model deferred). Build a symbol
          table (each symbol + a guessed description from surrounding text).
       e. TABLES: detect ruled + unruled tables (line-detection + whitespace columns,
          pdfplumber/Camelot-style); reconstruct grid + caption + units.
       f. FIGURES: extract embedded images + nearest caption.
       g. CONFIDENCE: per-component score; aggregate parse_confidence.
  3. SCANNED PATH (escalation):
       a. RASTERIZE pages → images.
       b. OCR: Tesseract (CPU, default) → text + boxes; (🧪 Apple Vision / TrOCR
          for higher accuracy deferred behind a feature flag).
       c. Re-enter step 2b–f on OCR output (lower confidence; flagged).
  4. ESCALATION TRIGGER (optional, behind 🧪 HF flag): if parse_confidence < T_low
       OR equation/table density is high and reconstruction failed, queue the doc for
       a vision-model re-parse (Nougat/Marker-class) — a §8 moonshot, off by default.
  5. NORMALIZE → StructuredDoc (content-addressed). Re-parsing identical bytes with the
       same parser version yields an identical StructuredDoc (determinism, Tenet 7).
```

**Key decisions:**

- **Deterministic CPU first, vision later.** v1 ships *no* GPU/vision dependency (🧪 HF-deferred). PDFium + heuristics handle the bulk of born-digital arXiv PDFs; GROBID (if the user runs the service) upgrades reference parsing. This keeps v1 buildable, fast, and offline.
- **Prefer LaTeX source when available.** For arXiv papers, the `arxiv` adapter fetches the **LaTeX source tarball** when present — equations and structure come out *perfectly* from source, sidestepping PDF math OCR entirely. (Massive quality win for the most common scientific source.)
- **Equations carry a symbol table**, not just LaTeX — because §4.10 turns equations into code and needs to know what each symbol *is*.
- **Everything is provenance-stamped**: each Section/Equation/Table/Figure records its `char_span`/`page` so downstream Claims trace back exactly (Tenet 2).
- **Low-confidence parses are quarantined**, not silently trusted: `parse_confidence < T_review` (§9) routes the doc to a "needs review" queue in the Research Tab (§6, bad-OCR mitigation).

---

### 4.6 The Research Pipeline: State Machine

The heart of the lab. A `ResearchRun` is an explicit FSM, checkpointed to a run ledger so an overnight run is **resumable** and **auditable**. This generalizes the existing `deep-research` skill into a persistent, graph-writing engine.

```
States:  PlanScope → FanOutSearch → Triage → Fetch → Read → Verify
                 → Synthesize → Persist → (Reflect ↺ | Done)

Transitions are budget-gated (BudgetGuard) and every state appends events to runs/<id>.jsonl.
```

```python
# Pseudocode (engine is Rust; this is the executable spec for hawking-research/pipeline.rs)

def research_run(query, scope, budget, kg, runtime, fetch, cas):
    run = open_run_ledger(query, scope, budget, seed=stable_seed(query))
    state = resume_or_start(run)                  # crash-resume from last checkpoint

    # ── 1. PLAN ───────────────────────────────────────────────────────────────
    if state <= PLAN:
        plan = runtime.chat(planner_prompt(query, scope))   # decompose → sub-questions
        # plan = { sub_questions:[...], search_terms:[...], expected_entities:[...],
        #          stop_conditions:{coverage, novelty_floor}, source_kinds:[...] }
        # Seed from EXISTING KG: what do we already know? (don't re-research)
        prior = kg.local_query(query, k=GRAPH_SEED_K)       # GraphRAG local (§4.8)
        plan = fold_prior_knowledge(plan, prior)            # gaps-only research
        run.checkpoint(PLAN, plan)

    # ── 2. FAN-OUT SEARCH (parallel, budget-bounded) ──────────────────────────
    if state <= FANOUT:
        candidates = []
        for q in plan.search_terms:                          # FAN_OUT_WIDTH parallel
            for provider in providers(scope):                # web (ch.03), arXiv, S2, OpenAlex
                if budget.exhausted(): break
                hits = fetch.search(provider, q, k=PER_QUERY_K)   # ch.03 / adapter.metadata
                candidates += hits
        run.checkpoint(FANOUT, candidates)

    # ── 3. TRIAGE: dedup + source-quality score + select ──────────────────────
    if state <= TRIAGE:
        # Dedup by canonical id (DOI/arXiv/url-normalize) AND by embedding near-dup
        deduped = dedup(candidates, kg, radius=DEDUP_COSINE)         # §4.7
        scored  = [(c, source_quality(c)) for c in deduped]          # §4.7 scoring
        # Skip anything already fully ingested & fresh in KG (don't re-fetch)
        fresh   = [c for c,_ in scored if not kg.has_fresh(c, ttl=SOURCE_TTL)]
        selected = topk_by_score_and_diversity(fresh, k=READ_BUDGET) # diversity = anti-echo
        run.checkpoint(TRIAGE, selected)

    # ── 4. FETCH (through ch.03 audited fetch; CAS-cached, dedup) ──────────────
    if state <= FETCH:
        for src in selected:
            if budget.exhausted(): break
            raw = adapter_for(src).fetch(src, ctx)            # paywall/robots enforced by ch.03
            cas.put(raw)                                      # immutable, dedup by hash
        run.checkpoint(FETCH, fetched_hashes)

    # ── 5. READ: parse → chunk → embed → EXTRACT entities/claims into KG ───────
    if state <= READ:
        for raw in fetched:
            doc   = adapter_for(raw).parse(raw, ctx)          # §4.5 StructuredDoc
            chunks = chunk(doc)                               # section-aware
            embs   = runtime.embed([c.text for c in chunks])  # → ch.04 vector index
            kg.upsert_doc(doc, chunks, embs)                  # nodes + provenance
            facts  = kg.extract(doc, runtime)                 # GraphRAG-style: entities,
                                                              #   relations, CLAIMS (§4.8)
            kg.merge(facts, entity_resolution=True)           # incremental dedup (§4.8)
        run.checkpoint(READ, read_doc_ids)

    # ── 6. VERIFY (adversarial): triangulate claims, surface contradictions ────
    if state <= VERIFY:
        claims = kg.claims_for_run(run)
        for claim in claims:
            support = kg.evidence(claim, polarity=+1)          # SUPPORTS edges
            refute  = kg.evidence(claim, polarity=-1)          # REFUTES edges
            if independent_sources(support) < MIN_CORROBORATION:
                # actively seek a SECOND independent source or a refutation
                extra = targeted_search(claim, fetch, exclude=support.sources)
                ingest_and_link(extra, claim, kg)
            claim.confidence_tier = grade(support, refute)     # measured/inferred/contested
            if refute: kg.add_edge(CONTRADICTS, claim, refute.claim)
        # Hallucinated-citation guard: re-open each cited span, re-hash, re-check the
        # quoted figure/phrase still present (Tenet 2 / §6).
        verify_citations_against_cas(claims, cas)
        run.checkpoint(VERIFY, verification_report)

    # ── 7. SYNTHESIZE: cited report from VERIFIED graph slice ──────────────────
    if state <= SYNTHESIZE:
        subgraph = kg.run_subgraph(run)                        # only this run's verified facts
        report = runtime.chat(synth_prompt(query, subgraph))   # every sentence → claim id(s)
        report = enforce_citations(report, subgraph)           # drop/flag uncited sentences
        run.checkpoint(SYNTHESIZE, report)

    # ── 8. PERSIST: Findings, Report node, edges; feed memory + maybe issues ───
    if state <= PERSIST:
        findings = extract_findings(report, subgraph)          # actionable items
        kg.add_report(report, run); kg.add_findings(findings, run)
        bridge.memory.write(findings, subgraph)                # → ch.04 long-term memory (§4.13)
        if scope.auto_issues: bridge.issues.mint(findings)     # → issues (§4.10), opt-in
        run.checkpoint(PERSIST, findings)

    # ── 9. REFLECT: coverage / novelty gate → loop or finish ───────────────────
    coverage = assess_coverage(plan.sub_questions, subgraph)
    novelty  = last_round_novelty(run)                         # new entities/claims added
    if (coverage < plan.stop_conditions.coverage
        and novelty > plan.stop_conditions.novelty_floor
        and not budget.exhausted()):
        plan = replan_gaps(plan, subgraph)                     # research only what's missing
        run.checkpoint(PLAN, plan); goto FANOUT                # ↺ another round
    else:
        run.finalize(DONE)
        return Report(report, findings, subgraph_ref=run.id)
```

**Budget governor (replaces token-cost limits with local-appropriate limits):**

```rust
pub struct Budget {
    pub max_wall: Duration,        // e.g., overnight = 8h (default for `--deep`)
    pub max_fetches: usize,        // politeness + disk bound, not money
    pub max_disk_bytes: u64,       // CAS growth cap
    pub max_rounds: u32,           // reflect-loop ceiling
    pub politeness: RateLimits,    // per-host rate caps (ch.03 enforces)
}
```

> **Resumability is the overnight unlock.** Each state checkpoints to `runs/<id>.jsonl`. If the machine sleeps or HIDE restarts at 3 a.m., the run resumes from the last completed state without re-fetching (CAS dedup) or re-extracting (content-addressed nodes). This is what makes "kick off a 200-paper sweep before bed" reliable.

**Run modes (presets over the same FSM):**

- `--quick`: 1 round, `FAN_OUT_WIDTH` small, `READ_BUDGET` ~8, minutes. (Interactive.)
- `--standard`: 2–3 rounds, ~25 sources. (Default.)
- `--deep`: rounds until coverage/novelty stall, `max_wall` = overnight, hundreds of sources. (The free-overnight differentiator.)

---

### 4.7 Source-Quality Scoring & Adversarial Verification

**Dedup (two layers):**

1. **Canonical-id dedup**: normalize DOI / arXiv id / URL (strip tracking params, `arxiv.org/abs` vs `/pdf`, version suffixes) → exact-match collapse.
2. **Near-duplicate dedup**: embed candidate titles+abstracts (ch.04), collapse within `DEDUP_COSINE` (default 0.95). Catches mirrors, preprint↔published, blog-reposts. The *highest-quality* representative is kept; others become `SAME_AS` aliases (provenance preserved).

**Source-quality score** (0..1, transparent & tunable — *not* a black box):

```
quality(src) = w1·venue_signal        # peer-reviewed venue / venue rank / preprint penalty
             + w2·citation_signal      # citation count (age-normalized), OpenAlex/S2
             + w3·recency_signal       # newer ↑ for fast-moving fields (dial per field)
             + w4·authority_signal     # author h-index / institutional / domain allowlist
             + w5·primary_vs_secondary # primary research ↑ vs blog/secondary ↓
             + w6·corroboration        # how many *independent* sources agree
             - p1·contradiction_penalty# flagged as refuted by higher-quality sources
             - p2·domain_blocklist     # content-farm / known-unreliable hosts (ch.03 list)
```

Weights default to a research-sensible profile and are a §9 dial (e.g., a *news* sweep weights recency; a *foundations* sweep weights citations+venue). **Every score is explainable**: the Research Tab shows the breakdown per source (Tenet 2/10).

**Adversarial verification (the anti-hallucination core):**

1. **Triangulation**: a `measured` claim needs `MIN_CORROBORATION` (default 2) *independent* sources (independent = different authors *and* not citing each other for that figure). If under-corroborated, the verifier *actively searches for a second source or a refutation* before grading.
2. **Contradiction surfacing**: when sources disagree, the system does **not** average them away — it records `CONTRADICTS` edges and presents both with their quality scores. (E.g., "Paper A reports 73% (peer-reviewed, 2024); Blog B claims 91% (no venue, contradicts A).")
3. **Citation re-verification**: before any sentence ships in a report, its cited span is re-opened from CAS, re-hashed against the stored `content_hash`, and checked that the *quoted figure/phrase is literally present*. Sentences whose citations don't verify are dropped or flagged — **this is the structural guard against the #1 deep-research failure mode (hallucinated/mis-attributed citations).**
4. **Stance classification** on citations (`SUPPORTS`/`MENTIONS`/`CONTRASTS`, à la scite): a paper *citing* another isn't necessarily *agreeing* — the edge `stance` captures this so literature maps show support vs dispute, not just connectivity.

---

### 4.8 GraphRAG-Style Incremental Build & Query

**Construction (incremental, online — not batch rebuild):**

- **Extraction**: for each new `StructuredDoc`/chunk, an LLM extraction prompt (via `RuntimeClient`) emits typed entities (Concept/Method/Dataset), claims, and relations with **span provenance** (GraphRAG/LightRAG pattern). Output is constrained to the schema (§4.2) via a typed function-call/JSON contract and validated before merge.
- **Entity resolution (the hard, crucial part)** — `entity_resolution.rs`, run **online** as each batch arrives:
  - **Blocking** by normalized key (lowercased label, alias set, external id) to limit comparisons.
  - **Candidate matching** by (a) exact external id, (b) string similarity (Jaro-Winkler) on labels/aliases, (c) embedding cosine on label+context.
  - **Merge** above `RESOLVE_THRESHOLD` (default 0.92): unify nodes, union aliases, *preserve all provenance* (a merged Concept node keeps every span that mentioned any alias). Sub-threshold-but-close pairs become `SIMILAR_TO` (not merged) and can be surfaced for user confirmation.
  - **Never destructive**: merges are recorded as `SAME_AS` redirections so they're *reversible* if a later, better signal says two concepts were wrongly unified.
- **Community detection (online)**: incremental label-propagation / streaming-Leiden over the entity graph maintains hierarchical communities as nodes arrive (rather than recomputing from scratch). Each community gets an LLM-generated **summary node** refreshed lazily when its membership churns beyond `COMMUNITY_DIRTY`. (GraphRAG's community-summary idea, made incremental.)

**Query (dual-level, GraphRAG-style):**

```rust
pub enum KgQuery {
    /// LOCAL: entity-centric. Embed query → seed nodes (ch.04 ANN) → ego-graph expand
    /// k hops → rank chunks/claims by relevance+centrality → return with provenance.
    Local { text: String, hops: u8, k: usize },

    /// GLOBAL: corpus-level sensemaking ("what are the main approaches to X?").
    /// Map over relevant COMMUNITY SUMMARIES → reduce into an answer. Beats vector RAG
    /// on global questions (GraphRAG result).
    Global { text: String, level: u8 },

    /// PATH / structural: "how does paper A connect to method M?" — shortest/witness
    /// paths over typed edges (CITES/IMPLEMENTS/DERIVES_FROM…). Cypher under the hood.
    Path { from: NodeId, to: NodeId, edge_filter: Vec<EdgeType> },

    /// HYBRID: vector recall ∪ graph expansion ∪ FTS, fused + re-ranked. Default for
    /// "answer with my whole corpus" questions; also the backend for chat-over-research.
    Hybrid { text: String },

    /// Raw Cypher escape hatch for power users / the litmap module.
    Cypher(String),
}
```

> **This is the GraphRAG advantage made persistent:** the cloud rebuilds a graph per-job and throws it away; HIDE's graph *is the corpus* and answers get better as it grows — every paper you read makes every future "compare/summarize/global" query sharper, for free, forever.

---

### 4.9 Citation / Literature Mapping Workflows

Built on the citation edges (`CITES`, `CO_CITED_WITH`, `BIB_COUPLED_WITH`) and entity edges. Concrete, named workflows (each a CLI command + later a Research-Tab view):

- **`litmap build <seed papers|topic>`** — Connected-Papers-equivalent, *local*. From seed paper(s), pull citation neighborhood (OpenAlex/S2 via adapters), compute **co-citation + bibliographic-coupling similarity** (not just direct citation, like Connected Papers), lay out a graph (force-directed; communities colored by §4.8). Unlike Connected Papers: it includes *your unpublished notes/papers*, links nodes to *your code*, and you *own* it and can grow it.
- **`litmap compare <paperA> <paperB> ... <paperN>`** — "compare N papers": auto-builds a **comparison table** (rows = papers; columns = method, dataset, metric+score, key claim, limitations, stance toward each other), each cell **provenance-linked** to the source span. Uses `EVALUATES_ON`, `USES`, `MAKES_CLAIM`, `CONTRADICTS` edges. (Elicit-style extraction, but joined to the full graph + offline + free.)
- **`litmap timeline <topic>`** — chronological evolution of a `Method`/`Concept`: who `INTRODUCES` it, the chain of `DERIVES_FROM` improvements, citation-weighted milestones. Renders a timeline with branch points.
- **`litmap gaps <topic>`** — **research-gap finder**: surfaces under-explored regions — concept pairs with no connecting paper, claims with weak corroboration, methods never `EVALUATES_ON` a given dataset, contradictions left unresolved. *This is a generative differentiator: it proposes what to research/experiment next (feeds §4.11).*
- **`litmap influence <paper>`** — forward/backward citation tree with stance coloring (who *built on* vs *disputed* this), age-normalized influence.
- **`litmap consensus <claim>`** — Consensus.app-style: aggregate all `SUPPORTS`/`REFUTES` evidence for a claim across the corpus with quality-weighted tally and the contradiction set.

All maps are **live views over the KG** — they update as ingestion proceeds and are queryable/exportable (GraphML, CSV, BibTeX, or a static HTML map you own).

---

### 4.10 Research → Issues → Code Pipeline

The unification play: research outputs become *work in your repo*. Three concrete bridges (in `hawking-bridge`).

#### 4.10.1 Finding → Issue / Task

A `Finding` (actionable output of a research run) maps to a repo issue/task. **Decisive mapping contract:**

```rust
pub struct IssueDraft {
    pub title: String,                 // imperative, from finding.summary
    pub body_markdown: String,         // context + WHY (links to claims/papers) + acceptance
    pub labels: Vec<String>,           // ["research", topic, kind:{bug|feature|experiment|chore}]
    pub provenance: Vec<NodeId>,       // Claim/Paper/Equation nodes that justify this issue
    pub suggested_effort: Effort,      // S/M/L (heuristic from finding scope)
    pub linked_symbols: Vec<NodeId>,   // CodeSymbol nodes (ch.05) this likely touches
    pub experiment: Option<ExperimentSpec>, // if finding implies an experiment (§4.11)
}

impl IssueBridge {
    /// Findings → issue drafts. NEVER auto-files by default — produces drafts the user
    /// reviews in the Research Tab (or `--auto-issues` opts into filing). Each filed issue
    /// gets a MINTED_ISSUE edge back to its Findings → bidirectional traceability.
    fn mint(&self, findings: &[Finding]) -> Vec<IssueDraft> { /* … */ }

    /// Sinks: GitHub/GitLab (gh/glab CLI), a local Markdown issues/ board, or ch.05's
    /// task system — chosen by config. Sink is a trait so new trackers plug in (§7).
    fn file(&self, draft: &IssueDraft, sink: &dyn IssueSink) -> Result<IssueRef>;
}
```

The mapping logic (`findings → drafts`):

```
for finding in findings:
    kind   = classify(finding)            # bug-implication | feature-idea | experiment | doc | chore
    why    = render_provenance(finding)   # "Motivated by [Paper X §3.2 claim …], contradicts current
                                          #  impl in src/foo.rs:42 (see CodeSymbol …)"
    accept = acceptance_criteria(finding) # what 'done' looks like, drawn from the claim/metric
    symbols= code_index.search(finding.summary)   # ch.05: where in MY code this lands
    exp    = experiment_spec(finding) if kind == experiment else None
    emit IssueDraft{ title: imperative(finding.summary), body: why+accept, kind, symbols, exp }
```

> Example: a sweep on "KV-cache quantization" finds a paper claiming int4 KV with per-channel scales loses <0.5 ppl. The Finding → IssueDraft: *"Evaluate int4 per-channel KV scales in serve path"*, body cites the paper's claim node + the contradicting current `MAKES_CLAIM` from your own notes + links `CodeSymbol(src/.../kv_cache.rs)` from ch.05 + attaches an `ExperimentSpec` (baseline ppl vs int4-KV ppl). This is research **turning directly into a tracked, sourced, code-anchored task** — no cloud agent does this because their research and your repo never share a substrate.

#### 4.10.2 Equation → Code

```rust
impl EquationBridge {
    /// Equation node (latex + symbol table) → typed code stub in the target language,
    /// with: a docstring citing the source paper+equation number, named params from the
    /// symbol table, a unit/shape contract, and a property-test skeleton. Produces a diff
    /// for the user to apply (never auto-commits). IMPLEMENTS edge links the new CodeSymbol
    /// back to the Equation + Paper (research⇄code join, Tenet 5).
    fn to_code(&self, eq: &Equation, lang: Lang, ctx: &CodeCtx) -> CodeStubDiff;
}
```

Pipeline: `latex + symbols → LLM (RuntimeClient) → function signature (params = symbols, types from context) → body (translated math) → docstring with citation → property test (e.g., dimensional sanity, known limit cases) → diff`. The generated symbol becomes a `CodeSymbol` node `IMPLEMENTS` the `Equation` with `fidelity` (exact/approx). ch.05 indexes the new code; the link is bidirectional.

#### 4.10.3 Paper → Reproduction scaffold

`reproduce <paper>` assembles an experiment scaffold from a paper's `USES`/`EVALUATES_ON`/`Equation` nodes: a checklist of methods+datasets to obtain, equation stubs (§4.10.2), a `Hypothesis` ("we can reproduce result R"), and an `ExperimentRun` template (§4.11) — turning "read a paper" into "run the paper" inside one environment.

---

### 4.11 Experiment Planning & Tracking

Closes the loop literature → hypothesis → experiment → result → back-to-graph. `hawking-experiments`.

```rust
pub struct Hypothesis {
    pub id: NodeId, pub statement: String, pub rationale: String,
    pub motivated_by: Vec<NodeId>,     // MOTIVATED_BY: Papers/Claims/Findings/gaps (§4.9)
    pub predictions: Vec<Prediction>,  // falsifiable: metric, direction, threshold
    pub status: HypothesisStatus,      // Open | Supported | Refuted | Inconclusive
}

pub struct ExperimentSpec {            // a *plan* (pre-registration)
    pub hypothesis: NodeId,
    pub method: String,                // what to run
    pub variables: Variables,          // independent/controlled/dependent + ranges
    pub datasets: Vec<NodeId>,
    pub metrics: Vec<String>,
    pub power_note: Option<String>,    // expected effect / how many runs
    pub command_template: String,      // the actual command to execute the run
}

pub struct ExperimentRun {             // an *execution* (immutable record)
    pub id: NodeId, pub spec: NodeId,
    pub cmd: String, pub params: serde_json::Value,
    pub code_rev: String,              // git SHA (joins ch.05 / repo)
    pub env_hash: [u8; 32],            // toolchain + deps fingerprint (reproducibility)
    pub seed: u64,
    pub started: DateTime<Utc>, pub ended: Option<DateTime<Utc>>,
    pub metrics: serde_json::Value,    // captured results
    pub artifacts: Vec<NodeId>,        // logs/plots/checkpoints in CAS
    pub status: RunStatus,             // Planned|Running|Done|Failed
}
```

**Tracking & reproducibility:**

- An `ExperimentRun` records `code_rev` + `env_hash` + `seed` + exact `cmd` → **reproducible by construction** (the local-first equivalent of W&B/MLflow lineage, joined to the literature graph). Artifacts (logs, plots) go to CAS; metrics to the graph.
- `TESTS` and `MOTIVATED_BY` edges mean every run is traceable to the hypothesis and the *papers that inspired it* — a link no standalone tracker has.
- **Experiment-planning agent** (EMERGING, §3.6): from a `litmap gaps` result or a `Hypothesis`, propose an `ExperimentSpec` (variables, datasets, metrics, command). The user approves before anything runs. Off by default for autonomous execution; planning is assistive.
- **Result feedback**: when a run completes, its metrics become `Claim` nodes (`confidence_tier = measured`, provenance = this run) that `SUPPORT`/`REFUTE` the `Hypothesis` and can `CONTRADICT` literature claims — *your own experiments become first-class evidence in the same graph as the papers.* (This is the research-laboratory payoff: your findings and the world's findings live together, comparably, locally.)

**This subsystem strongly overlaps the existing Hawking quant-sweep work** (the repo already runs `tools/condense/sweep.py`, ppl ladders, JSONL run logs, watchdogs). `hawking-experiments` is the *general* version of that ad-hoc pattern: the ppl-ladder runs become `ExperimentRun` nodes, the "★residual quant ≈ 1:1" findings become `Claim`s linked to hypotheses. (Concrete near-term dogfood — see §9.)

---

### 4.12 Note-Taking & Synthesis Surfaces

- **Zettelkasten bridge (bidirectional)**: `Note` nodes ⇄ Markdown files in `notes/` with `[[wikilinks]]` and YAML frontmatter (`id`, `tags`, `links`). Edit a note in HIDE's editor *or* in Obsidian/any editor — a file watcher syncs both ways. `[[links]]` become `ANNOTATES`/typed edges; the KG's auto-discovered edges surface as backlinks in the note. **You get an auto-populated Obsidian vault.** (Tenet 5; bridges ch.04.)
- **Research canvas**: a spatial surface to drag papers/claims/notes and draw relations → those drawings *write edges to the KG* (manual edges carry `provenance = UserEntered`). Mixed human+machine graph.
- **Synthesis surfaces**:
  - *Annotated bibliography* (auto, from a topic query): each paper's summary + key claims + your notes + stance, exportable.
  - *Literature-review draft*: `Global` KG query → structured review (intro / approaches-by-community / contradictions / gaps / refs), every sentence claim-linked, BibTeX emitted.
  - *Daily research digest* (🏗️ scheduling, post-shell): an overnight `--deep` run on your saved interests → morning digest of new papers + what changed in the graph + suggested issues. (The "wake up to free research" experience.)
- **Chat-over-research**: a chat panel whose retrieval backend is the `Hybrid` KG query (§4.8) — ask your *own* corpus questions, get answers with inline provenance chips. (SciSpace "chat with papers", but over your *entire private library at once*, offline.)

---

### 4.13 Integration with ch.04 Memory & ch.05 Code Index

This section is the **cross-cutting contract** (also returned to the caller). Clean ownership boundaries; the Research Lab *links into* these subsystems rather than duplicating them.

#### 4.13.1 ↔ ch.04 Memory Substrate

ch.04 owns long-term semantic memory + the embedding/vector index. The Research Lab is a **producer and consumer**:

- **Embeddings ownership**: the Research Lab never runs its own ANN index. It calls ch.04 to embed (`RuntimeClient.embed`, ultimately `/v1/embeddings`) and to store/recall vectors. KG `Chunk.embedding_ref`, `Claim`, `Concept`, etc. hold **opaque `embedding_ref` keys** into ch.04's index. (One vector store, no drift.)
- **Memory write (research → memory)** — `bridge::memory.write(findings, subgraph)`:
  ```rust
  /// Research promotes durable knowledge into ch.04 long-term memory. Contract:
  ///  - Each MemoryItem carries: text, embedding_ref, source_node_ids (KG provenance),
  ///    confidence_tier, salience, created_at, ttl?(none=permanent), kind=Research.
  ///  - ch.04 dedups against existing memory; on conflict, higher-quality/with-provenance wins.
  ///  - The memory item retains a back-link (kg_ref) so recall can hop INTO the graph.
  fn write(&self, items: &[MemoryItem]) -> Result<()>;
  ```
  → Findings, well-corroborated `Claim`s, and key `Concept` definitions become long-term memory. *What you researched last month informs the coding agent today* — across sessions, locally, forever.
- **Memory read (memory → research)**: at `PlanScope`, the pipeline queries ch.04 for what the user *already knows/researched* to **avoid re-researching** and to seed the plan (`fold_prior_knowledge`). Memory recall can return `kg_ref`s, letting the agent expand from a remembered fact into the full graph neighborhood.
- **Shared provenance vocabulary**: both subsystems use the same `Provenance` shape (§4.2.4). A memory item and a KG claim that came from the same source share `content_hash` → unified "where did I learn this?" across memory and research.

#### 4.13.2 ↔ ch.05 Code Index

ch.05 owns the symbol graph + repo embeddings. The Research Lab **joins** to it via `CodeSymbol` handle-nodes (never copies code):

- `CodeSymbol` nodes store only `{repo, path, symbol, kind, index_ref}` — a **join key** into ch.05, kept fresh by ch.05's indexer (the Research Lab subscribes to ch.05 index updates to repair stale handles).
- **Edges that span the boundary**: `IMPLEMENTS` (CodeSymbol → Method/Equation/Paper), `MINTED_ISSUE` (Finding → Issue touching symbols), `ExperimentRun.code_rev` (run → git SHA).
- **Queries that span the boundary**:
  - *"What paper does this function implement?"* — from a symbol (ch.05), traverse `IMPLEMENTS` into the KG → the Paper/Equation + its provenance.
  - *"Where in my code is method M used?"* — from a `Method` node, follow `IMPLEMENTS` to `CodeSymbol`s → ch.05 locations.
  - *"This finding affects which code?"* — `issues.rs` calls `code_index.search(finding.summary)` → candidate symbols for the issue draft (§4.10.1).
- **Direction of truth**: ch.05 is authoritative for code structure; the KG is authoritative for research structure; `IMPLEMENTS`/`MINTED_ISSUE` are the only edges allowed to cross, and they live in the KG with provenance.

#### 4.13.3 ↔ Local Runtime (fixed contract)

The Research Lab depends only on the runtime's HTTP surface (already shipping): `POST /v1/embeddings` (embeddings), `POST /v1/chat/completions` (SSE synthesis/extraction/planning), `POST /v1/hawking/generate` (native). All via the `RuntimeClient` trait (§4.1). **Model-agnostic** (🧪 a condensed 32B simply yields better synthesis; the lab's code is unchanged).

---

### 4.14 The Research Tab (post-shell UI)

> 🏗️ **POST-SHELL.** Specified fully here; ships after the editor + agent shell. The engine (§4.1–4.13) is usable headless via CLI before this exists.

A first-class tab beside Editor/Agent. Panels:

- **Library** — ingested sources (papers/PDFs/datasets/repos), filter/search (FTS + semantic), ingestion status, parse-confidence badges (low-confidence → "needs review" — §6).
- **Graph** — interactive KG view (filter by node/edge type, community coloring, expand neighborhoods). Click a node → provenance + connected code/notes. This is the "Connected Papers you own" surface.
- **Research Runs** — launch/monitor runs (mode `quick/standard/deep`, scope, budget); live progress through the FSM states; the **run ledger** as an auditable timeline; resume a paused overnight run.
- **Reports** — generated cited reports; every sentence has a **provenance chip** (hover → source span; click → open the PDF at that page). Regenerate/diff a report from the graph.
- **Lit Maps** — the §4.9 workflows (compare table, timeline, gaps, consensus) as views.
- **Experiments** — hypotheses board, run tracker, results, links to motivating papers + code rev.
- **Notes / Canvas** — zettelkasten editor + spatial canvas (§4.12).
- **Review queue** — low-confidence parses, unresolved contradictions, under-corroborated claims, near-duplicate-merge confirmations: the human-in-the-loop surface for trust.

UX laws: **provenance is always one click away**; **measured vs inferred vs speculative are visually distinct** (Tenet 10); contradictions are *shown*, never hidden (Tenet 3).

---

## 5. How We Exceed ("cloud literally cannot do this")

Each item is a *structural* advantage of the local plane — not a feature the cloud forgot, but one its business/architecture forbids.

1. **Free overnight deep research.** No per-token meter → the default is "go deep." An 8-hour, 300-source, multi-round adversarial sweep costs $0 of marginal compute on hardware the user owns. Cloud deep-research is metered and *time-boxed to minutes* because unattended depth is a direct cost. **Cloud cannot match the unit economics of free.** (And it runs while you sleep, resumable across a laptop closing its lid.)
2. **A private, persistent, compounding knowledge graph.** Your entire reading/research history becomes a structured graph that gets *denser every night*, lives on *your disk*, and is *never uploaded*. Cloud assistants start every research job cold and remember nothing across sessions; their privacy model can't promise your corpus never leaves. **The compounding asset is impossible to rent.**
3. **Local PDFs & datasets that never leave the machine.** Confidential papers, unpublished drafts, proprietary datasets, NDA material — ingested, parsed, graphed, queried, *fully offline*. No cloud tool can promise "your unpublished manuscript and your company's data were never transmitted" because the work *is* the transmission.
4. **Research and code in one substrate.** Findings become *sourced issues in your repo*; equations become *typed functions linked back to the paper*; your *experiments become evidence in the same graph as the literature*; a function can answer *"which paper do I implement?"*. Cloud research tools and your repo never share a data model — they *cannot* form these join edges. **The graph spans papers↔claims↔code↔experiments because it's all local.**
5. **GraphRAG that improves with use, for free.** The cloud rebuilds a throwaway graph per job; HIDE's graph *is* the corpus, so every "compare these / summarize the field / what are the gaps" query is sharper than the last — and re-indexing is free overnight, not a billed batch job.
6. **Adversarial verification you can audit to the byte.** Every cited figure is re-checked against an *immutable local copy* of the source (content-hash receipt). You can prove a citation is real because the evidence bytes are on your disk. Cloud reports cite the live web, which mutates and 404s.
7. **Your own experiments in the loop.** Local compute means you can *run* the experiment a paper inspired and fold the result back into the graph as first-class evidence — a research↔experiment↔literature loop that no read-only cloud research tool closes.

> **The one-sentence moat:** *HIDE is the only environment where free overnight research builds a private, permanent knowledge graph that is joined to your code and your experiments — the cloud cannot offer free, cannot offer permanent-and-private, and cannot offer joined-to-your-repo, all three at once.*

---

## 6. Failure Modes + Mitigations

| Failure | Why it happens | Mitigation (designed-in) |
|---|---|---|
| **Hallucinated / mis-attributed citations** | LLM synthesis invents a source or attributes a real claim to the wrong paper | **Provenance-or-it-doesn't-exist (Tenet 2).** Synthesis can only cite `Claim` nodes that exist in the run subgraph; `enforce_citations()` drops uncited sentences; **citation re-verification** re-opens each cited span from CAS, re-hashes, and confirms the quoted figure/phrase is literally present (§4.7.3). A claim that fails is flagged, not shipped. |
| **Paywalls / inaccessible sources** | Many papers are gated | Adapters prefer **legal OA** (Unpaywall, arXiv, preprints, author PDFs, OpenAlex/S2 abstracts). Paywalled items are recorded as **metadata-only Paper nodes** (`paywalled:true`) — still in the citation graph, just without full text. ch.03 enforces ToS; we never bypass paywalls. The Review queue surfaces "wanted but gated" so the user can supply a legitimately-obtained copy via the `pdf_local` adapter. |
| **Bad OCR / parse errors** | Scanned PDFs, complex layouts, math/tables | **Confidence-gated quarantine.** `parse_confidence < T_review` → Review queue, not silent trust. Fallback cascade (born-digital → OCR → optional vision re-parse §8). **Prefer LaTeX source** for arXiv (perfect math). Equations/tables that fail reconstruction are stored as **image + uncertain text**, flagged, never asserted as `measured`. |
| **Stale facts / source drift** | Web pages change; "current best" ages | CAS pins the *exact bytes seen* (immutable receipt) + `retrieved_at`; `SOURCE_TTL` triggers re-fetch for time-sensitive queries; a re-fetch that hash-differs raises a "source changed" event and re-verifies dependent claims. |
| **Echo chamber / false corroboration** | N sources all copy one origin → fake consensus | **Independence test** in triangulation (different authors *and* not citing each other for that figure); diversity term in `topk_by_score_and_diversity`; near-dup dedup collapses mirrors *before* they inflate corroboration counts (§4.7). |
| **Entity-resolution errors** (merge two different concepts, or split one) | Name collisions, aliases | Conservative `RESOLVE_THRESHOLD`; merges are **reversible** `SAME_AS` redirections (non-destructive); sub-threshold pairs go to Review for user confirmation rather than auto-merge (§4.8). |
| **Graph bloat / drift over years** | Endless ingestion | Community summarization compresses; TTL/archival policy; `kuzu` scales to large graphs; CAS dedup bounds blob growth; periodic *consolidation* pass (mirrors the existing memory-consolidate discipline) merges redundant claims. |
| **LLM extraction schema violations** | Free-form generation off-spec | Extraction is a **typed/constrained** call validated against the §4.2 schema before merge; invalid extractions are rejected + logged, never written. |
| **Runaway overnight run** (cost-free ≠ harm-free: disk, rate-limit bans) | "Go deep" default | `Budget` caps wall-clock/disk/fetch-count; ch.03 per-host **politeness** rate limits prevent IP bans; the run is checkpointed so it can always be stopped and resumed. |
| **Over-trust of `inferred` content** | Users read synthesis as fact | Hard visual + data distinction `measured` vs `inferred` vs `speculative` (Tenet 10); reports lead with corroboration level; contradictions shown inline. |

---

## 7. Extensibility (new sources / ingestors)

The system is built to grow along three seams; each is a single trait.

1. **New source → implement `SourceAdapter`** (§4.4) + register. Examples to add later: PubMed/Europe PMC (biomed), bioRxiv/medRxiv/SSRN, USPTO/Google Patents, GitHub/GitLab issues+wikis, internal company wikis (Confluence/Notion export), RSS/news feeds, podcast transcripts, YouTube transcripts, Slack/email export (private corpora!), HF dataset/model cards (⏸️ when HF un-deferred). The pipeline is unchanged — it only knows the trait.
2. **New parser → implement a `parse/` module** behind `StructuredDoc`. Swap PDFium→Marker, add a vision model (§8), add a chemistry/formula parser (e.g., RDKit for SMILES), a music/score parser, etc. As long as it yields `StructuredDoc`, everything downstream works.
3. **New sink → implement `IssueSink`** (§4.10) for trackers (Jira, Linear, Trello, local board) and `RuntimeClient` for model backends. **New graph store** → implement the `store.rs` trait (Neo4j-embedded, Oxigraph/RDF, DuckDB-PGQ) if KùzuDB ever doesn't fit.
4. **Interop exports**: KG → RDF/SPARQL, GraphML, Cypher dump, BibTeX, CSL-JSON, Obsidian vault, RIS — so HIDE is never a roach-motel; your graph is portable.
5. **Plugin extraction prompts**: domain packs (a "bio pack", "ML pack", "law pack") supply specialized entity/relation extraction prompts + schema extensions (subtypes of `Method`/`Dataset`) without touching the core.

---

## 8. Bleeding-Edge / Moonshots (ranked)

Ranked by `(impact × feasibility) / cost`, most actionable first. Tags as §3.

1. **Vision-model PDF ingestion (Nougat/Marker-class), local.** 🧪 EMERGING. Escalation path for low-confidence parses → near-perfect math/table extraction on hard PDFs. *Why ranked #1:* directly attacks the biggest quality risk (bad parse), reuses the §4.5 escalation hook, and Apple-Silicon-runnable models exist. Gated on HF un-defer + a quality gate vs the deterministic path.
2. **Overnight autonomous research scheduler.** 🏗️ POST-SHELL + EMERGING. Cron-like: nightly `--deep` runs over saved interests → morning digest + auto-drafted issues + graph delta. *The headline "wake up to a literature review" experience.* Feasible once shell + scheduling exist; engine already supports it.
3. **`litmap gaps` → auto experiment proposals → (approved) auto-runs.** EMERGING (AI-Scientist-adjacent). Gap-finder proposes hypotheses + `ExperimentSpec`; user approves; HIDE *runs them locally* and folds results back as evidence. Closes the full research↔experiment loop. Ranked high because the repo already runs sweeps (dogfood = the quant ladder, §4.11).
4. **Cross-modal claim grounding.** SPECULATIVE→EMERGING. Link a textual claim to the *figure/table* that backs it (not just the paragraph) — "this 73% comes from Table 3, cell (2,4)" with a `TableCell` locator. Higher-fidelity provenance.
5. **Reproducibility scorer.** SPECULATIVE. For each paper, auto-assess reproducibility (code available? data available? equations extractable? compute feasible locally?) → a score + a one-click `reproduce` scaffold (§4.10.3).
6. **Federated graph sync (privacy-preserving).** SPECULATIVE. Two researchers merge *public* slices of their graphs (or a lab shares a curated subgraph) without exposing private nodes — selective, encrypted, opt-in graph federation. Keeps local-first while enabling collaboration.
7. **Continuous personalized embedding adaptation.** SPECULATIVE + 🧪. Fine-tune/adapt the *local* embedding model on the user's corpus so retrieval matches *their* field's vocabulary — a personalization the cloud can't do per-user economically. (Synergy with Hawking Condense's on-device adaptation work.)
8. **Argument/debate graphs.** SPECULATIVE. Beyond `supports/refutes`: model full argument structure (premises→conclusions, assumptions, counterarguments) for contested topics — a reasoning map, not just a citation map.
9. **Auto-generated survey papers.** SPECULATIVE. From a mature topic subgraph, draft a citation-complete survey (with figures = generated lit maps) — export to LaTeX. The natural endpoint of §4.12 synthesis.

---

## 9. Open Questions / Dials

**Open questions (need a decision or measurement):**

- **Graph store**: commit to KùzuDB now, or abstract behind `store.rs` and benchmark Kùzu vs Neo4j-embedded vs Oxigraph on a 100k-node corpus first? (Leaning: ship Kùzu behind the trait — reversible.)
- **Extraction model size**: is the local runtime's default model strong enough for reliable schema-constrained entity/relation extraction, or does extraction *require* a condensed larger model? (🧪 measure once 7B/32B condensed runtime is the daily driver.)
- **Entity-resolution precision/recall tradeoff**: where exactly to set `RESOLVE_THRESHOLD` — wrong merges are reversible but corrosive; missed merges fragment the graph. Needs a labeled eval set.
- **Community-detection cost online**: does streaming-Leiden stay cheap as the graph grows, or do we need periodic batch recompute windows (overnight)?
- **Dogfood first**: should `hawking-experiments` *first* absorb the existing quant-sweep tooling (`tools/condense/sweep.py`, ppl ladders) as the proving ground before generalizing? (Strong yes — real runs, real findings, immediate value.)
- **Note sync conflicts**: bidirectional Markdown↔Note sync needs a conflict policy (last-write-wins vs 3-way merge vs CRDT) — which?

**Dials (defaults chosen; user/profile-overridable):**

| Dial | Default | Range / note |
|---|---|---|
| `FAN_OUT_WIDTH` (parallel search terms) | 6 | 2–20 |
| `PER_QUERY_K` (hits per search) | 10 | |
| `READ_BUDGET` (sources read/round) | 25 (standard) | 8 (quick) … unbounded (deep) |
| `DEDUP_COSINE` (near-dup radius) | 0.95 | |
| `RESOLVE_THRESHOLD` (entity merge) | 0.92 | conservative |
| `MIN_CORROBORATION` (independent sources/claim) | 2 | 1 (fast) … 3 (rigorous) |
| `SOURCE_TTL` (re-fetch staleness) | field-dependent (news: hours; foundations: ∞) | |
| `GRAPH_SEED_K` (prior-knowledge seed) | 20 | |
| `COMMUNITY_DIRTY` (re-summarize threshold) | 20% membership churn | |
| `T_review` (parse-confidence quarantine) | 0.6 | low → more human review |
| quality weights `w1..w6` | research profile | per-mode profiles (news/foundations/survey) |
| `Budget.max_wall` | 30 min (standard) / overnight (deep) | |
| at-rest encryption | off | opt-in (OS keychain key) |
| auto-file issues | off (draft-only) | `--auto-issues` to file |

---

## 10. Cross-References

- **ch.01 / ch.02 — App Shell & Panels**: the Research Tab (§4.14) is a 🏗️ post-shell panel hung on the shell's tab/command-palette infrastructure; the engine ships headless first.
- **ch.03 — Web Fetch / Search / Browser**: *consumed by* `hawking-ingest` and the pipeline's fan-out. ch.03 owns HTTP, headless browser, robots/ToS, rate-limiting, paywall policy, and the egress audit log. Adapters MUST route all network access through ch.03's `FetchService`/`EgressPolicy` (§4.4) — they never open sockets directly.
- **ch.04 — Memory / Knowledge Substrate**: owns embeddings + the vector index + long-term semantic memory. Research *writes* findings/claims into memory (`bridge::memory.write`, §4.13.1) and *reads* prior knowledge at plan-time; KG holds opaque `embedding_ref`s into ch.04's index (one vector store). Shared `Provenance` vocabulary.
- **ch.05 — Code Index**: owns the symbol graph + repo embeddings. The KG joins via `CodeSymbol` handle-nodes and the cross-boundary edges `IMPLEMENTS` / `MINTED_ISSUE` and `ExperimentRun.code_rev` (§4.13.2). Research→issues (§4.10) calls ch.05 to anchor findings to symbols.
- **Local runtime (existing Hawking serve)**: fixed HTTP contract — `POST /v1/embeddings`, `POST /v1/chat/completions` (SSE), `POST /v1/hawking/generate` — abstracted by `RuntimeClient` (§4.1, §4.13.3). Model-agnostic (🧪 `.tq`/32B condensed models improve synthesis quality only).
- **`deep-research` skill (existing)**: the v0 of §4.6 — this chapter generalizes its fan-out→fetch→verify→cited-report loop into a persistent, graph-writing, resumable, code-bridging pipeline.
- **Hawking Condense (sibling product)**: supplies the condensed local models (`.tq`) that power better local synthesis/extraction; §8.7 (personalized embedding adaptation) is a direct synergy.
- **`tools/condense/` sweep tooling (existing)**: the immediate dogfood for `hawking-experiments` (§4.11) — ppl ladders → `ExperimentRun` nodes, sweep findings → `Claim`s.

---

*End Chapter 08 · Research & Knowledge Lab.*
