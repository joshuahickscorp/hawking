# HCLI Perception Service Plan

**Status:** PLAN + SCAFFOLD ONLY — gated on Proto-Frankenstein offload.  
**Authority:** `HAWKING_ASCENSION_BIBLE.md` §19 (Perception service).  
**Scaffold crate:** `crates/hawking-perception`  
**Schema marker:** `hcli.perception.v0`

---

## 1. Purpose

Give HCLI agents a **cheap-first document perception path** so papers, profiling
screenshots, kernel timelines, PDF specs, benchmark charts, architecture
diagrams, and UI state can become **structured, coordinate-bound evidence**
without OCR-ing every page at maximum cost.

Default path (bible, non-negotiable order):

```text
cheap metadata/text pass
→ retrieve relevant pages
→ detect relevant regions
→ selective OCR or vision
→ parse tables/charts
→ structured evidence
→ coordinate-bound citation
```

---

## 2. Preconditions / gates

| Gate | Requirement |
|------|-------------|
| Proto-Frankenstein | Offloaded, hash-verified, out of active local storage envelope |
| Live model work | **Out of scope** for this plan stage — no Qwen/Gravity downloads, no vision model wiring |
| Frankenstein evidence | Do **not** touch `lab/operators/frankenstein_*` or frankenstein campaign evidence |
| Existing text research | Prefer reuse of `hawking-research` ingest/CAS seams for text evidence pins |

This stage ships **interfaces, stubs, budgets, and tool names only**. Real PDF
rasterization, OCR, and multimodal vision are later engineering after the
ascension bootstrap models are resident.

---

## 3. Existing patterns reused

| Existing | Reuse |
|----------|--------|
| `hawking-research::ingest::{StructuredDoc, SectionEvidence, DocSpan}` | Text-section evidence already CAS-pinned; PDF body parse is an **documented open seam** there |
| `hawking-research::cas::{pin_evidence, verify_evidence, blake3_hex}` | Citation re-verification pattern for quote bytes |
| `hide_core::types::{BlobRef, Provenance, TrustLevel}` | Content-addressed blob + trust labels (mirrored lightly in perception types) |
| `hide_backend::lenses_evidence::EvidenceTier` | Claim strength ordering; scaffold uses `EvidenceStrength` without backend dep |
| `hide-kernel` tool registry / `hide_protocol::Tool` | Future registration of `document.*` tools with declared effects (`ReadFs`) |
| `hide-backend::hcli_bridge` | Future capability area / method surface; not extended in this scaffold |

**Audit result:** No production OCR/vision pipeline exists under `crates/`,
`lab/`, or `tools/`. Graph/condense/eval tooling is unrelated. Scaffold is clean
interfaces + deterministic text fixtures.

---

## 4. Tool surface (stable names)

Bible-mandated tools — encoded as `DocumentToolName` in the scaffold:

| Tool | Stage | Cost tier |
|------|-------|-----------|
| `document.inspect` | Cheap metadata / free text | Metadata |
| `document.retrieve_pages` | Subset of pages | PageText |
| `document.detect_regions` | Regions on retrieved pages | PageText |
| `document.parse_region` | Selective text/OCR/vision | PageText → RegionOcr → Vision |
| `document.parse_table` | Table structure | RegionOcr (or free text) |
| `document.parse_chart` | Chart series extraction | Vision preferred |
| `document.verify_structure` | Outline / table shape checks | Metadata/PageText |
| `document.cite_coordinates` | Coordinate-bound citation | Metadata |

**Rule:** Never escalate every page to Vision. `PipelineBudget` caps selective
regions and can disable OCR/vision entirely.

---

## 5. Design

### 5.1 Types (`hawking-perception`)

- `DocumentHandle` — id + optional content hash / URI / media type  
- `DocumentMeta` / `PageSummary` — inspect output  
- `PageContent` — retrieved page text (layer vs OCR flag)  
- `Region` / `RegionKind` / `CoordBox` — normalized page geometry (0‥1)  
- `ParsedRegion`, `TableParse`, `ChartParse`  
- `Citation` — quote + `CoordBox` + content hash + `EvidenceStrength`  
- `StructuredEvidence` — final bundle for LEVEL 2 bus / research KG  

### 5.2 Service trait

`DocumentService` methods map 1:1 to the eight tools. Implementors:

| Backend | Role | Status |
|---------|------|--------|
| `StubDocumentService` | In-memory multi-page text fixtures | **Shipped** |
| PDF text-layer extractor | Free text without OCR | Future |
| Region OCR (Apple Vision / Tesseract / …) | Selective only | Future, model-infra |
| Multimodal vision | Charts/diagrams/UI | Future, model-infra |

### 5.3 Pipeline

`PerceptionPipeline::run(handle, PerceptionQuery)`:

1. Charge metadata budget → `inspect`  
2. Select pages (explicit or free-text rank) → `retrieve_pages`  
3. `detect_regions` on retrieved pages only  
4. Selective parse (table/chart/text) with region cap  
5. `verify_structure`  
6. Emit `StructuredEvidence` with coordinate-bound `Citation`s  

Budget defaults: OCR **off**, vision **off**, max 16 selective regions, 500 cost
units. Production agents must opt into OCR/vision deliberately.

### 5.4 Coordinate-bound citation

Citations pin:

- `document_id`  
- normalized `CoordBox` (page + x0,y0,x1,y1)  
- optional `region_id`  
- quote + `blake3` content hash  
- strength ≥ `coordinate_bound` when geometry is present  

This is what later verification authority and LEVEL 2 evidence graphs consume.

---

## 6. Phased delivery

### Phase P0 — Scaffold (this work)

- [x] Crate `hawking-perception` with types, trait, tools, pipeline, stub  
- [x] Unit tests: tool names, cheap pipeline, budget refusal, citation geometry  
- [x] This plan document  

### Phase P1 — Wire into HCLI (after Frankenstein offload + agent OS activation)

- [ ] Register `document.*` in hide-kernel tool registry with `Effect::ReadFs`  
- [ ] Optional `hcli_bridge` capability area `perception` (read-only description)  
- [ ] Project `StructuredEvidence` → `hawking-research` claims / CAS pins  
- [ ] Project citations into LEVEL 2 `EvidenceGraph` (`hawking-comms`)  

### Phase P2 — Cheap PDF/text backends (still no max-cost OCR)

- [ ] Local PDF text-layer + outline via a pinned pure decoder  
- [ ] Image/screenshot dimensions + EXIF-style metadata inspect  
- [ ] Page ranking with real embeddings only if already resident (no new model pulls in this phase unless owner-approved)  

### Phase P3 — Selective OCR / vision (real model work — separate programme)

- [ ] Region-bounded OCR backend behind feature flag  
- [ ] Chart/diagram vision adapter (same-model Qwen multimodal or dedicated)  
- [ ] Cost meters + receipts for every Vision escalation  
- [ ] Never default to full-document OCR  

---

## 7. Non-goals

- Implementing OCR, Vision, PDFium, or screenshot capture in this stage  
- Pulling Qwen / Gravity / any model weights  
- Claiming production perception capability from the stub fixture  
- Mutating frankenstein operators or live GLM recapture evidence  
- Push / PR / remote publish  

---

## 8. Acceptance for P0

| Check | Evidence |
|-------|----------|
| Tool names match bible | `DocumentToolName::ALL` → eight `document.*` strings |
| Cheap-first enforced | OCR disabled → `max_cost_tier <= page_text` |
| Citations coordinate-bound | `Citation.box_` normalized; content hash non-empty |
| No vision infra dep | Crate deps: serde, thiserror, blake3 only |
| Tests pass | `cargo test -p hawking-perception` |

---

## 9. Downstream consumers

- Agent OS retrieval gateway (§15) — page/region retrieval  
- Communication bus LEVEL 2 — evidence graphs / tool results  
- Verification authority (§22) — re-open cited coordinates  
- Research lab — papers + profiling screenshots as structured docs  

---

## 10. Open questions (do not block P0)

1. Which PDF text backend is preferred on Apple-first (pdfium vs quartz vs pure Rust)?  
2. Should screenshot perception share the same tool namespace or `screen.*`?  
3. Exact receipt schema for Vision cost ledgers (align with FLOPS ledger §11)?  
