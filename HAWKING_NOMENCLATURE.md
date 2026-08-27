# Hawking Nomenclature — Canonical Paradigm

**Status:** current forward vocabulary
**Version:** `HAWKING_NOMENCLATURE_V1`
**Scope:** active HCLI, Flash-Next, Gravity, Noetic, accelerator, and resident
work. Historical receipts and sealed identifiers remain unchanged.

This is a glossary and semantic contract, not a second roadmap. The existing
campaign canons and handoffs remain historical sources; new work uses this
vocabulary and carries `nomenclature_version`.

## The canonical pipeline

```text
SOURCE SPECIMEN
      ↓
DOCTOR
      ↓
GRAVITY
      ↓
NOETIC IR / NOETIC PROGRAM
      ↓
NOETIC COMPILER
      ↓
PHYSICAL GRAPH COMPILER
      ↓
HAWKING ACCELERATOR
      ↓
NOETIC EXECUTABLE CANDIDATES
      ↓
PARETO FRONTIER
      ↓
SINGULARITY
      ↓
RESIDENT
```

## Canonical meanings

| Term | Meaning | Hawking boundary |
|---|---|---|
| **Source Specimen** | The cold, pinned source checkpoint from which a candidate is derived. | The exact Flash-Next ModelLake specimen and manifest are a Source Specimen. |
| **Doctor** | Measures, prescribes, verifies, and rejects. | Doctor is a mechanism on the Gravity/Noetic line, not a model or campaign brand. |
| **Gravity** | The process/search engine for capability-preserving reduction of physical cost. Quantization is one Gravity operator. | Gravity is not a file format. Existing `.gravity` formats and `hawking.gravity.*` schemas are compatibility-preserved historical/runtime identifiers. |
| **Noetic IR / Noetic Program** | The representation and executable-intelligence ontology of a candidate. It can contain tensors, codebooks, routing, state, generators, graphs, and policies; it need not be a tensor. | `hcli.noetic.representation_descriptor.v1` is the active descriptor boundary for Flash experiments. |
| **Noetic Executable** | A complete runnable bundle independent of the cold Source Specimen. | A complete Flash bundle is not yet built; `FLASH_NEXT_NOETIC_EXECUTABLE.json` is an explicitly refused scaffold. |
| **Noetic Executable Candidate** | A qualified, runnable candidate under consideration for a product/machine profile. | Bounded Flash Q4 transform, loader, and kernel receipts are evidence for a candidate component, not a complete candidate. |
| **Physical Graph Compiler** | Lowers Noetic structure into device dataflow, placement, synchronization, and kernel calls. | `hcli.physical_graph.v1` remains a planning boundary until execution evidence exists. |
| **Hawking Accelerator** | The physical execution substrate: Metal/GPU and any future governed accelerator backend. | A Metal dispatch receipt is physical component evidence; it is not a full model runtime receipt. |
| **Pareto Frontier** | The set of non-dominated qualified Noetic Executables across the measured objective dimensions. | EBPW, accepted TPS, capability, memory, and product constraints must be considered together. |
| **Pareto Archive** | The durable record of frontier candidates and their rejection/qualification reasons. | Older Pareto tables remain readable; they are not silently renamed or rewritten. |
| **Singularity** | One Noetic Executable promoted from the Pareto Frontier for an explicit product contract, capability contract, machine genome, and execution profile. | No Singularity is selected by the current Flash scaffold. |
| **Singularity Profile** | The explicit identity of the product/machine/execution profile that justifies a Singularity choice. | A future selection must include profile identity; “best model” is not an identity. |
| **Resident** | The currently instantiated/running Noetic Executable. | The sealed Qwen resident is the current local default profile, not an HCLI type assumption. |

## Selection law

Never choose a Singularity by minimum EBPW alone. EBPW is an accounting
dimension, not a universal ordering. A higher-EBPW candidate may be the
Singularity when it is capability-equivalent, fits the product/memory contract,
and wins the selected execution profile's accepted useful work or latency
objective. The lower-EBPW candidate remains in the Pareto Archive and may be
selected for another profile.

For Flash-Next, the current evidence therefore means:

- independent Q4/G64 is the selected **bounded tensor representation** because
  its observed tensor EBPW is lower (`4.25` bits/value);
- shared BF16 basis + NF4 residual is retained as a lower-error quality
  alternate (`4.28125` bits/value);
- neither is a Singularity or complete Noetic Executable until full closure,
  capability, and protected complete-token execution are proven.

## Compatibility and migration

Historical names are evidence, not cleanup debt. Do not rename or move sealed
receipts, hashed paths, schema strings, crate names, or the large ModelLake
tree merely for aesthetics. New code may translate old language into the
canonical semantic class:

| Legacy/ambiguous phrase | Canonical interpretation |
|---|---|
| `source model`, `checkpoint`, downloaded model | **Source Specimen** |
| `quantizer`, `quantization` | **Gravity operator** when it is a cost-reduction search/transform; keep the narrower term when it names an algorithm. |
| `compressed model`, `compact model` | **Noetic representation** or **Noetic Executable Candidate**, depending on whether it is storage-only or runnable. |
| `artifact` | **Noetic Executable**, **candidate**, or **receipt** only after semantic inspection; retain `artifact` in compatibility schemas when it is part of a sealed ABI. |
| `winner`, `best model`, `final model`, `production model` | **Pareto Candidate** or **Singularity** only when qualification and profile identity justify it; otherwise use the precise observed status. |
| `resident model` | **Resident** when the object is actually instantiated; otherwise **Source Specimen** or candidate. |
| `model lake` | **ModelLake source-specimen store**; storage location, not a representation or runtime. |

New active receipts should contain:

```json
{"nomenclature_version": "HAWKING_NOMENCLATURE_V1"}
```

Parsers may expose a canonical semantic view while retaining the original
field and schema. Historical receipts must remain byte-for-byte untouched.

## Active implementation map

- `hcli.nomenclature` owns the forward vocabulary, version, pipeline, and
  compatibility aliases.
- `hcli.physical_graph` serializes the canonical version and compiler-stage
  metadata while retaining its existing schema and behavior.
- `hcli.agentos` Flash receipts add the canonical version and distinguish
  Source Specimen evidence, Noetic representation, bounded accelerator
  evidence, executable scaffold, and Resident/TPS claims.
- `tools/headless/nomenclature_census.py` and
  `receipts/headless/NOMENCLATURE_CENSUS.json` remain the search/index boundary.
  The census classifies names rather than performing mechanical renames.
- `tools/headless/nomenclature_census.py`, old canons, and sealed receipts are
  not permission to treat every occurrence of “artifact,” “gravity,” or
  “resident” as the same semantic object. Inspect behavior first.

## Final law

> **GRAVITY DISCOVERS.**
> **NOETIC REPRESENTS.**
> **THE COMPILER LOWERS.**
> **THE ACCELERATOR EXECUTES.**
> **PARETO REMEMBERS.**
> **SINGULARITY CHOOSES.**
> **THE RESIDENT RUNS.**
