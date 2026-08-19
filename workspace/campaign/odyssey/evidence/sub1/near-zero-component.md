I'll start by reading the frontier brief and amendment in full, then enumerate each near-zero mechanism with honest byte accounting.The amendment is the target: per-component collapse to 0.01–0.1 BPW, not a uniform codec. Next I’ll pull the existing science and how complete BPW is actually counted so the accounting stays honest.I have the kill ledger. Next I’ll pull complete-BPW accounting, sensitivity maps, and the GLM 0.167 precedent so each mechanism is grounded in this campaign, not generic compression lore.O005 already shows a single expert can be zeroed on the fast battery. I’ll pull complete-BPW formulas and the GLM 0.167 precedent so the per-component byte math is exact.I’ll locate the complete-BPW formula and the GLM 0.167 / Q80 numbers so the per-component accounting is sourced, not invented.I'll use the connected search MCP to locate the tensor names and `complete_bpw` formula.I'll query GitHub/sg MCP schemas, then search the repo for those tensor and bpw strings.**`/Users/scammermike/Downloads/hawking` content search**

0 hits (no content-search tool in this session; `list_dir` only). Searched tree by name, not line text.

**PATTERNS TRIED (not executed as grep):** `moe_intermediate_size 768`, `switch_mlp`, `experts.0`, `gate_proj` / `up_proj` / `down_proj` shapes, `complete_bpw`

**Name-only locations (not `file:line` matches):**

**`/Users/scammermike/Downloads/hawking/tools`**
- `/Users/scammermike/Downloads/hawking/tools/odyssey_candgen.py`
- `/Users/scammermike/Downloads/hawking/tools/gravity_bpw.py`
- `/Users/scammermike/Downloads/hawking/tools/gravity_bpw_family.py`
- `/Users/scammermike/Downloads/hawking/tools/__pycache__/odyssey_candgen.cpython-312.pyc`
- `/Users/scammermike/Downloads/hawking/tools/__pycache__/gravity_bpw.cpython-314.pyc`

**`/Users/scammermike/Downloads/hawking/crates/hawking-core/src`**
- `/Users/scammermike/Downloads/hawking/crates/hawking-core/src/model/qwen_moe.rs`
- `/Users/scammermike/Downloads/hawking/crates/hawking-core/src/model/qwen_dense.rs`
- `/Users/scammermike/Downloads/hawking/crates/hawking-core/src/model/arch_config.rs`
- `/Users/scammermike/Downloads/hawking/crates/hawking-core/src/moe.rs`

**`/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey`**
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/patients/O001/census.json`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/patients/O004/census.json`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/patients/O005/census.json`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/downloads/O001_falcon-h1-7b.log`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/downloads/O004_mistralai_Mistral-Small-3.1-24B-Instruct-2503.log`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/downloads/O005_qwen3-30b-a3b.log`

**complete_bpw formula:** not retrieved (no file open / no grep).

**Novelty lane reports**
- `/Users/scammermike/Downloads/hawking/receipts/odyssey-i/`: 27 json files; **no** `*NOVELTY*` names.
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/evidence/sub1/`: **empty**.
- Novelty *contracts* (not receipts):  
  `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/contracts/auto/o005_novelty-adversarial-falsifier.md`  
  `o005_novelty-arch.md`  
  `o005_novelty-compression.md`  
  `o005_novelty-kernel.md`  
  `o005_novelty-numerical.md`  
  `o005_novelty-representation.md`

0 hits in 0 files — PATTERNS TRIED: moe_intermediate_size 768, switch_mlp, experts.0, gate_proj/up_proj/down_proj, complete_bpw (content grep unavailable).I'll read the listed paths and search tools for complete_bpw / packed_bpw accounting.The locators couldn’t grep. I’ll read the BPW accounting sources and use the sourcegraph MCP for the 0.167 / Q80 numbers.I'll locate the requested strings via the search MCP after loading its schema.Searching remaining files for complete_bpw / packed_bpw definitions.0 hits — no Grep/content-search tool is available in this session (only `list_dir`; Read is forbidden). Line-level matches cannot be produced.

**PATTERNS TRIED:** none executed (no search tool). Intended queries: `0.167`, `0.755`, `complete_bpw`, `1.43`, `nominal_bits`, `glm52.pq` / `0.75 BPW` / `0.116`, `indexshare` / `IndexShare`, `complete_bpw` / `packed_bpw` / `bits_per_weight`, `false-win` / `no_fake_density`, `student dense map` / `0.0104 BPW`, `Q80` / `qwen80` mixed 1.43, prune / drop-to-constant / codebook amortization.

**Paths listed only (not grepped):** `/Users/scammermike/Downloads/hawking`, `/Users/scammermike/Downloads/hawking/tools`, `/Users/scammermike/Downloads/hawking/workspace` (not opened). `/Users/scammermike/Downloads/h_odyssey.md` not searchable without Read/Grep.

0 hits in 0 files=== /Users/scammermike/Downloads/hawking/tools/odyssey_candgen.py [L348-L464] ===
```
# ---------------------------------------------------------------------------
# Native / complete-bpw estimates
# ---------------------------------------------------------------------------


def _nav_flag(table: dict, key: Any, default: bool = False) -> bool:
    if not table:
        return default
    if key in table:
        return bool(table[key])
    s = str(key)
    if s in table:
        return bool(table[s])
    if isinstance(key, float):
        s2 = _fmt_budget(key)
        if s2 in table:
            return bool(table[s2])
    return default


def is_native(
    families: dict,
    *,
    form: str,
    bits: int | None,
    group: int | None,
    metadata_codec: str = "raw",
    correction_budget: float = 0.0,
    router_precision: int | None = None,
    correction_token: <REDACTED> = False,
) -> bool:
    nav = families.get("native_availability") or {}
    forms = nav.get("forms") or {}
    if form == "tiers" and not forms.get("tiers", False):
        return False
    if form == "scale_joint" and not forms.get("scale_joint", False):
        return False
    if correction_token and not forms.get("correction_token", False):
        return False
    if bits is not None and not _nav_flag(nav.get("bit_classes") or {}, int(bits), False):
        return False
    if group is not None and not _nav_flag(
        nav.get("group_sizes") or {}, int(group), True
    ):
        return False
    if not _nav_flag(nav.get("metadata_codec") or {}, metadata_codec or "raw", True):
        return False
    if float(correction_budget or 0) > 0 and not _nav_flag(
        nav.get("correction_budget") or {}, float(correction_budget), False
    ):
        return False
    if router_precision is not None and not _nav_flag(
        nav.get("router_precision") or {}, int(router_precision), True
    ):
        return False
    return True


def affine_complete_bpw(bits: float, group: int | None, metadata_codec: str = "raw") -> float:
    """Payload bits + f16 scale + f16 zp per group, adjusted for metadata codec.

    mlx affine grouped q3-g32 lands near 4.0 complete_bpw (O005 MEASURED 4.0253).
    """
    bits = float(bits)
    g = int(group or 64)
    sidecar = (16.0 / g) * 2.0  # scale + zero-point
    if metadata_codec == "shared":
        sidecar *= 0.35
    elif metadata_codec == "entropy":
        sidecar *= 0.70
    return bits + sidecar


def estimate_complete_bpw(parsed: dict) -> float:
    form = parsed["form"]
    meta = parsed.get("metadata_codec") or "raw"
    corr = float(parsed.get("correction_budget") or 0)
    if form == "mixed":
        lo = int(parsed["mixed_lo"])
        hi = int(parsed["mixed_hi"])
        # Most mass at lo; a small sensitive subset promoted to hi.
        payload = 0.85 * lo + 0.15 * hi
        group = parsed.get("group") or 64
        bpw = affine_complete_bpw(payload, group, meta)
    elif form == "tiers":
        tiers = parsed.get("tiers") or ["t0"]
        # T0 stored at 1 bit, each extra tier adds a 0.5-bit residual plane (counted).
        bpw = affine_complete_bpw(1.0, 32, meta) + 0.5 * max(0, len(tiers) - 1)
    elif form == "scale_joint":
        # Joint r-search can shave sidecar entropy; still count the affine container.
        bpw = affine_complete_bpw(int(parsed["bits"]), parsed.get("group"), meta)
        bpw *= 0.97
    else:
        bpw = affine_complete_bpw(
            int(parsed["bits"] or 4), parsed.get("group"), meta
        )
    bpw += corr * 16.0  # correction_budget is a fraction of weights at f16
    return bpw


def estimate_stored_bytes(census: dict, complete_bpw: float) -> int:
    params = int(census.get("total_params") or 0)
    if params <= 0:
        return 0
    return int(params * float(complete_bpw) / 8.0)


def estimate_active_bytes(
    census: dict, patient_class: str, complete_bpw: float
) -> int:
    src_bpw = float(census.get("stored_bpw") or 16.0) or 16.0
    scale = float(complete_bpw) / src_bpw
    if patient_class == "moe" and census.get("active_bytes_per_token"):
        return int(float(census["active_bytes_per_token"]) * scale)
    stored = estimate_stored_bytes(census, complete_bpw)
    return stored
```

=== /Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/compile_economics_schema.json [L1-L40] ===
```
{
  "$schema": "hawking.odyssey.compile_economics/v1",
  "_doc": "Compile-economics + detachment-metrics schema (S003 §7-11, S004 §27/§71). Event log: COMPILE_ECONOMICS.jsonl. Derived packet: derive(patient). Campaign fit: ODYSSEY_COST_MODEL.json (UNCERTAINTY required; never a point estimate as fact). Detachment: detachment_metrics(). Evidence class on every number (§18).",
  "event_log": {
    "path": "workspace/campaign/odyssey/COMPILE_ECONOMICS.jsonl",
    "schema": "hawking.odyssey.compile_economics.event.v1",
    "required": ["schema", "patient", "event", "wall_s", "bytes_scanned", "bytes_transformed", "ts", "_evidence"],
    "fields": ["patient", "event", "wall_s", "bytes_scanned", "bytes_transformed", "grok_lane", "opus", "ts", "source_bytes", "source_params", "active_params", "complete_bpw", "bpw", "candidate_class", "valid_nx", "cheap_kill", "doctor_kind", "rules_reused", "rules_retuned", "rules_new", "opus_tokens", "cycle", "gpu_s", "cpu_s", "full_source_passes", "sampled_passes", "transform_passes", "verify_passes"]
  },
  "compile_economics_packet": {
    "source": "S003 §7",
    "fields": ["acquisition_wall", "census_wall", "doctor_baseline_wall", "gravity_analysis_wall", "gravity_search_wall", "gravity_pack_wall", "gravity_verify_wall", "nx_lower_wall", "kernel_probe_wall", "total_gpu_wall", "total_cpu_wall", "grok_lane_count", "grok_wall", "opus_escalations", "opus_wall", "total_patient_wall", "first_valid_nx_wall", "retirement_wall", "source_bytes", "bytes_scanned", "bytes_transformed"]
  },
  "normalized": {
    "source": "S003 §8",
    "fields": ["gravity_gb_s_scanned", "gravity_gb_s_transformed", "s_per_B_source_param", "s_per_B_active_param", "first_nx_s_per_B_param", "doctor_s_per_B_active", "experiments_per_valid_nx", "grok_lanes_per_patient", "opus_per_patient", "rules_reused", "rules_retuned", "rules_new"],
    "rule": "ratios that cannot be formed are null — never inf/nan. Do not Goodhart; instrumentation only."
  },
  "frontier_depth": {
    "source": "S004 §27",
    "fields": ["time_to_first_valid_nx", "time_to_conventional_anchor", "time_to_first_sub3", "time_to_first_sub2_5", "time_to_first_sub2", "time_to_best_frontier", "candidates_to_frontier", "cheap_kills", "doctor_fast_count", "full_doctor_count", "grok_calls_to_frontier"]
  },
  "transfer_acceleration": {
    "source": "S003 §9",
    "rule": "do not invent expected-from-scratch until enough evidence; until then report rules_reused + new_rule_count + actual_wall"
  },
  "cost_model": {
    "path": "workspace/campaign/odyssey/ODYSSEY_COST_MODEL.json",
    "schema": "hawking.odyssey.cost_model.v1",
    "inputs": ["source_bytes", "parameter_count", "active_parameter_count", "tensor_count", "expert_count", "architecture_family", "modality", "representation_passes", "rule_reuse_pct", "doctor_workload", "native_primitives", "source_acquisition_size"],
    "outputs": ["acquisition_wall", "gravity_wall", "first_nx_wall", "doctor_wall", "patient_wall"],
    "rule": "status=insufficient data when n_patients<2. Every estimate carries uncertainty + _not_fact + _label INFERRED|UNKNOWN. Never present a point estimate as fact."
  },
  "detachment_metrics": {
    "source": "S004 §71",
    "fields": ["deterministic_decisions", "total_decisions", "deterministic_fraction", "grok_escalations_per_patient", "opus_escalations_per_patient", "opus_escalations_per_cycle", "patient_wall_per_opus", "patient_wall_per_opus_token", "rules_per# Near-zero_opus_escalation", "nr component lane

_nx_candidates_per_opus**Target_escalation", "desired_trend (amend", "desired_trend_flags"],ment):**
    "desired_trend": " a *singledeterministic_fraction UP, candidate_throughput* organ UP, grok_novelty / expert / channel / layer_quality UP, opus_dependency DOWN at",
    "rule": "deterministic **0.1_decisions/total from RUN_,LOG.jsonl if present. Desired-trend flags are policy direction, not an observed slope (need 0.01, or ~0 ≥2 snapshots)."
  }
}
 BP```

=== /Users/scammW**, with completeermike/Downloads/hawking/ accounting, so theworkspace/docs/implementation/gravity/ *total* can fall belowGRAVITY_CONTAINER_SPEC.md [L70- 1 /L160] === 0.2 because
```
16 BPW and most components collapse physically written nowhere. The artifact then read and a few stay as proof that the BF16
source protected.

 could be evicted.

## The two**This rates

Both live in `compression` and is not both a must reconcile against physical bytes.

```text
packed bit_bpw     compressed payload bits /-width retreat compressed elements
               what the codec achieved.** Affine grouped quant on the tensors it compressed

complete_ floors near **bpw   all payload bits / all~1 elements
               what the shard actually costs–4 complete, native organs included BPW** (
```

`complete_bpw` is the only rate a candidate mayO005 `q3- be judged on. A campaign target suchg32-experts` ** as
"<= 0.75MEASURED 4.025" always refers to `complete_bp3** stored /w`. `packed_bpw` **4. exists to attribute a result to
the2305** active). Near codec and is never a headline.

`-zero is **verify()` recomputes both from the body and rejects thestructure, file if either claim is off by
 drop, sharemore than 1e-6, or if any descriptor, or generate**.

 carries no payload.

## Native tensors

EvidenceA tensor whose `codec` starts with class is `native.` carries its exact source bytes marked on. It is not an
approximation and must round-trip every number. **UNKNOWN bit-identically. Three reasons a** means tensor takes this path:

```text
 not measuredPROTECTED_BUDGET_CLASS      the contract classified it as control-sensitive
 here —NON_BF16_CONTROL_TENSOR not a     not a ladder candidate, still declared source weight
NO guess.

---

## _ADMISSI0. Accounting law (nonBLE_LADDER_RUNG   no rung-negotiable)

Normalize met the ceiling, so protect rather than every component exceed it
```

Native tensors are against ** billed at their real width, in `sourcecomplete_bpw`, from the same element count of that component bytes
that are stored. Billing and storing are the same**, never the reduced act.

## Integrity

Two levels, degrees of freedom (` on purpose:

```text
body_sha256      coverstools/gravity_bpw every payload byte; catches truncation and corruption.py`).

```
 on open
tensor.sha256   component_complete_bytes per tensor; identifies which tensor is bad =
    payload instead of condemning the shard
```
  + scales

A hash check alone was never sufficient, and the Generation + zp/ A artifacts prove it: they
passedbiases
  both levels while missing eleven organs per organ-bearing shard. + amortized_ Coverage is a
separate property from integrity and is checked separatelyshared_.

## Representation IDs

Thetables     # codebook container does not imply that any codec is / U, permanent. Current IDs:

```V / templatestext
glm52.pq.r / generator
  + offsets0.v1                product quantization, the Generation A control
  + correction / family
glm52.functional.block residual /.v1     repair
  + tier native functional block student
glm52.indexshare.student. +v1   IndexShare-aware attention router-remap student
glm52.hybrid.doctor + drop.v1        native base plus a serialized Doctor correction
llama.residual-pq.v1-flags
  + alignment padding          additive residual PQ: fp16 [stage][code
  + attributable container][D] descriptor tables and
                              MSB-first [
  + mandatory persistentrow][chunk][stage] reconstruction state

component_complete indices; direct compact execution
native._bpw = 8bf16                   exact source bytes, b * component_complete_bytesfloat16
native.f32                    / N_source exact source bytes, float32
```
component

New IDs may be added_local_bpw    at any time without touching the container version = 8 * (payload + local_. A reader that
does not know an ID can still enumerate, locate,meta + hash-verify and skip the tensor correction) / N_source.

## What v1 promises

```
                         #text
the 20-byte prefix index and its field order
the header is UTF-8 JSON-only; NOT at a declared length
every tensor is locatable by name without reading the body a headline

model_
every tensor carries a non-emptycomplete_bpw     payload and its own digest
both rates = 8 * (Σ reconcile against physical bytes
native.<dtype unique bytes> payloads are exact source bytes
```, shared

## What v1 does not promise tables ON

```text
any particular codec,CE) / N_model rung, or geometry
that a given
```

**Two representation id will still be produced
tensor rates, always.** ordering beyond the declared offsets
that the `local header carries any field not listed above
_bpw` can```
```

=== /Users/ be ~scammermike/Downloads/haw0 while `completeking/tools/gravity_ir.py_bpw` is  [L1-L2+ if the33] ===
``` codebook is private or
#!/usr/bin/ thinlyenv python3
"""Gravity IR — shared.

**Gravity a physical model program, not a tensor ABI-codec table.

The existing recipe vocabulary:** tensor is `tensor: codec`. It cannot `bytes` is never  express anything
this campaign now wants:0 (`GRAVITY_ structure shared across sites,CONTAINER blocks_ generatedSPEC rather.md`). A
than stored, additive correction stages, collapsed exact islands. Promising mechanisms had
 component still occupiesnowhere to land even when they measured well a descriptor +.

Design rules, each one a consequence ≥1 payload byte of a failure already paid for:

1. Budget. EVERY node reports **64 B its own stored bytes. Complete BPW is computed FROM the pad** unless
   program, the writer so a mechanism cannot claim a density it packs flags does not have. A pack once
.

   reported 3.6138 while**Eight-axis carrying an uncounted 1.814 family** (`gravity GB leftover.

2. Shared objects are_bpw_family.py CONTENT-ADDRESSED and counted`) ONCE no matter how — do not collapse many
   sites reference them. This them is the only way sharing can pay::

| Axis the whole point of
   a dictionary | This across 64 layers lane |
|---|---|
 is that it is stored once. Double| STORED | every-counting it
   would make sharing artifact look worthless; forgetting it entirely would make byte, sharing
   look inode-deduped free. Neither is honest.

3. |
| ACTIVE / DRAM The BPW denominator is the ORIGINAL source | bytes touched parameter count, never the
   candidate / token's own degrees of freedom. A structural after representation with fewer
   variables must still gather- normalise against theskip |
| GENER source or the number is not
  ATED | compute comparable.

4. Every node names the instead kernel that consumes it. A representation with of store ( no
   execution path is not a representation, it isHadamard already a compression demo.

 **REFVerification is the round-trip: BPUTED**)W computed from the program must equal BP |
| CORREW
measured fromCTION | repair bytes on disk. If it does not, the IR is not describing the artifact plane |
|.
"""
``` SHARED | table stored once (

=== /Users/scammermike/G035 **Downloads/hawking/tools/gravityREFUTED_ir.py [L96-L** cross-layer share at176] ===
``` matched bits on Q
def quant_tensor(elements,wen3.8) |
 bits, group, kernel, scale_| STATE | KV / SSMbytes_per_group=2, — not this lane header=40):
    # CE |

**Fake winsIL, not floor. A trailing partial group still needs its own scale, and floor
    # (instant reject)** silently drops it:

| 65 elements at g=64 is Fake | Why |
|---| 2 groups,---|
| Index and floor reported 1. On a-only BP
    # non-divisible shapeW, codebook off that understates every candidate that uses it-books.
    groups = max(1, | `no -(-elements // group))
   _fake_density` |
 code_bytes = (elements * bits| Expand-to-dense + 7) // 8
, then GEM    return Node("QuantTensor", kernelM, bill,
                stored_bytes=code_ only the seedbytes + groups * scale_bytes_ | notper_group + header,
                elements a native=elements, meta={"bits": bits sub-0.1, "group": group})


def object |
| dense_tensor(elements, dtype_ `topkbytes, kernel,/E header=40):
    return Node` as cost("DenseTensor", kernel, stored_bytes=elements * ( dtype_bytes +O005  header,
                elements=elements, meta0.0625)={"dtype_bytes": dtype_bytes | selection, not})


def shared_basis(elements bytes, coeff_bits, basis_cid (`O005, kernel, header=40):
_NX_gather`) |
    """Per-site coefficients against a| mlx basis stored once in the pool."""
 still    return Node("SharedBasis", kernel residents,
                stored_ the full expert body |bytes=(elements * coeff_bits + 7) // stored 8 + header,
                shared_ 0 + activerefs=[basis_cid], elements= full |
elements,
                meta| Proxy-activation={"coeff_bits": coeff_bits cosine | Q})


def sparse_correction(n_exceptions, valuewen3.8 sub_bytes, index-bit all_bits, kernel, header=40 collapsed;):
    """Exact values on a output small set. Index cost is counted ---div ~0.69 it usually dominates."""
    return Node |
| Fast ("SparseCorrection", kernel,
                stored12-item Δ0 as_bytes=n_exceptions * value Doctor_bytes + ( | G046/G048n_exceptions * index_bits +: battery 7) // 8 + header too narrow |
| 0,
                elements=0, meta={"-byte Gravity tensorn": n_exceptions, "index | malformed_bits": index_bits})


 |
def exact_island(n_elements| Alignment, value_bytes, kernel, index ignored_bits=0 on , header=40):
    """2–64A compile-time-known region kept exact. index_ B payloads | pad canbits=0 when be the set is static."""
    return 10–30 Node("ExactIsland", kernel,
               × the payload |

 stored_bytes=n_elements ***Affine value_bytes + sidecar (n_elements * index_bits (for contrast + 7) // 8 + header,
                elements, not this=0, meta={"n": n lane):**  
_elements, "`affineindex_bits": index_bits})_complete_bpw =


def generated_block(elements, bits + 32 code_bytes,/ generator_g`cid, (`od kernel, decode_yssey_candgen.pyflops_per_elem=0.`). q3-0,
                    headerg32 → =40):
    """A block computed from a tiny4.0 nominal; code plus a shared O005 landed generator."""
    **4.0253**. return Node("GeneratedBlock", kernel, Cannot reach 0.01 stored_bytes=code_bytes +.

---

## 1 header,
                shared._ Componentrefs sizes=[ (generatordenominator_cid], elements)

**=elements,
                active_bytes=O005** Qcode_bytes,
wen3-30B-                meta={"decode_flops_per_elem": decodeA3B —_flops_per **MEASURED_elem})


# ---------------------------------------------------------------- program

** census, **DER@dataclass
class SiteIVED** splits:
    name: str
    elements: int
   :

| Component terms: List[ | N paramsNode]


class Program:
    def __ | init__(self, name, source_0.10pin=None):
        self.name BPW | 0.01 BPW |
 = name
        self.source_pin = source_pin|---|---:|
        self.pool = SharedPool---:|---:|
()
        self.sites: List| [Site] = []

    def1 organ add(self, name, elements, (gate ** terms):
       or** up **or** self.sites.append(Site(name down), , elements, terms))
        return1 expert self

    #, 1 layer | ---- cost 

    def site_bytes1,572,864 |(self) -> int:
        return 19 sum(t.stored_bytes for s in self.,661sites for t in B | 1, s.terms)

966 B |
| 1    def total_bytes(self) expert (3 -> int:
        """Site-exclusive organs bytes plus every referenced shared object, counted), 1 layer |  once."""
        used4, =718 {cid,592 | for s in self 58.sites for t in s.terms,982 for cid in t.shared_refs B | 5}
        shared = sum(self.,898 B |
| pool.objects[1 expert-**c]["nbytes"] for c inindex** ×  used)
        return self.site_48 layers | 226,bytes() + shared

    def complete492,416 | 2_bpw(.83 MB | 283self) -> float:
        return  KB |
| Expert8 * self.total_bytes() body ( / SOURCE_PARAMall)_COUNT
```

=== /Users | 28,991,/scammermike/Downloads/029,248 | 95hawking/tools% of model/gravity_container.py [L62 | |
|-L100] ===
```
        Router (all "cost": {
            "total_bytes": program.) | 12,582total_bytes(),
            "site,912 | 0.03 GB bf16 | **do_bytes": program.site_bytes(),
            " not collapse**shared_bytes": |
| Attn / program.total_bytes() - program embed / lm_head |.site_bytes(),
            " 906complete_effective_bpw": program M.complete_bpw(),
            / 311 M /  "active_bytes311 M | protect_per_token": program.active_ | |

**O003**bytes_per_token(),
        Kimi-VL — },
    }
    # the seal 26 MoE layers ( covers everything above it78, so any later edit is detectable
 expert    body["content_id"] = modules / 3 _sha_json(body)
   ), **DER return body


REQUIRED = ("schema",IVED** = "source", "program", "shared **_pool", "kernels", "machineMEASURED**/",
            "64:

|doctor", "tabula", "cost Component | N |
", "content_id")


def|---|---:|
| verify(body, expect_bpw 1 expert-index_denominator=SOURCE_PARAM_COUNT):
    """Structural + integrity check. Returns a list of failures; empty means good."""
    fails = [] × 26 layers | 224,919,552 |
| Shared experts | 449
    for k,839,104 — ** in REQUIRED:
        if k not inalways-on; body:
            fails.append(f" domissing section: {k}")
    not drop** |

 if fails:
       **O006** language return fails

    stated = body[" Mocontent_id"]
    recomputedE matches O = _sha_json({k:005 expert body (** v for k, v in body.items() if k != "content_idMEASURED** 28,"})
    if stated != recomputed991,029,248).:
        fails.append(f"content

**O001** Falcon_id mismatch: stated-H {1stated —[: one16]} recomputed layer MLP ≈ {recomputed[:16]}")

    113 if body["source"]["bpw.2 M (**_denominator"] != expect_bpwDERIVED**)._denominator:
        fails.append(f SSM"bpw_denominator {body[' organsource']['bpw_denominator']} exists is not the "
; zeroing it kills                     f"original source parameter count { the batteryexpect_bpw_denominator}")

.

**O004    cost = body["cost"]
** dense    bpw = 8 * cost MLP body["total_bytes"] / body[" **MEASUREDsource"]["bp** 20.43 Bw_denominator"]
    if abs. Per-layer /(bpw - cost["complete_ per-channeleffective_bpw"]) > 1 sensitivity **UNKNOWN** (e-9:
no sensitivity        fails.append(f"declared BP receipt).W {cost['complete_effective_

---

## 2bpw']} !=. What recomputed {bpw}")
``` actually toler

=== /Users/scammermates near-zero

Fastike/Downloads/hawking/tools/odyssey_patient_runner battery.py [L202, in-place mlx0-L ablation, 4-bit2140] ===
```
def measure_ specimen. **Notcomplete_accounting(model, dest:** held-out Doctor Path, params: int, *, moe. Round-: bool) -> dict:
    """8 on anComplete bpw: already-4-bit organ payload+scales+biases+metadata+ is near-identity —headers. No fake density."""
    ignore it as payload = scales = biases = metadata = a descent 0
    signal.

| Component for path, val in tree_flatten | Zero Δ(model.parameters()):
        if hits not isinstance(val | Patients, mx.array):
            continue | Near
        b =-zero? |
 int(val.nbytes)
        leaf|---|---| =---| (---|
path.rsplit(".", 1)[-1| **1] if path else expert-index, "").lower()
        if leaf in all layers** | ** {"scales", "0** | Oscale"}:
            scales += b
005 hot         elif leaf in {"biases", "49 + random 77bias"}:
            biases += b
; O006 hot        elif leaf in 40 + random 78 {"table", " | **Yestables", "offsets", "offset", "lut", " — first targetg_idx"}** |
| :
            metadata += b
        else1 expert-index, all:
            payload += b
    live layers | **_total = payload−1** | O + scales + biases + metadata
   003 **hot**  disk = measure_dir_tensor_41 | Borderbytes(dest)
    tensor_bytesline; random = int(disk.get("stored_bytes") or  42 is Δ0)
    repr_bytes = 0 |
| All experts0
    for | p in −10 dest /.glob("*.saf −12 | O005 /etensors"):
        try:
            O003+ repr_bytes += p.stat().O006 | Nost_size
        except OSError |
| Shared:
            continue
    for extra in expert ( ("config.json", "model.safall) |etensors.index.json"):
        ep −12 | O003 | = dest / extra
        if ep No |
| Router /.is_file():
            try:
 embed / lm_head /                repr_bytes += ep.stat norm | −10..().st_size−12 | all measured
            except OSError:
                pass | No |
| All attn
    header_bytes = max( | −12 Mo0, repr_bytes - tensor_E; **−5bytes) if repr_bytes else  (0
    complete_bytes = (stilltensor_bytes or live_total) 5/12)** O + header_bytes
    complete_001 | hybridbpw = (complete_bytes * SSM 8 / params) if params else carries some language None
    live | O_bpw = (live_total001 **per- * 8 / params) if paramslayer attn** is else None
    policy_acc = opportunistic (load_od |
| All SSMyssey_policy().get("accounting | −10 | O_gates") or {}).get("no001 | No |
| All_fake_density dense") MLP
 |    − return9 {
        "payload_bytes": int / −11 | O001(payload),
 / O003 | No as        "scales_bytes": int( a blockscales),
        "biases_bytes |
| Gate vs": int(biases up vs down,),
        "metadata_bytes": per-layer int(metadata),
        "header, per-_bytes": intchannel | —(header_bytes),
        " | — | **UNKNOWN**live_bytes": int(live_ (organtotal),
        "disk_tensor inversion is found_bytes": int(tensor_bytesry-IN),
        "FERRED, not Odysseycomplete_bytes": int(complete_-measured) |
| Layerbytes),
        "complete_bp 0 vs midw": round(/late | —complete_bpw, 4) | — | L if complete_bp0 is a **different sourcew is not None else None,
       ** (NS- "live_bpw": round(layer-zero LIVElive_bpw, 4)). Do not assume L if live_bp0 isw drop ispable not. None |

 else None,
       **Cold "disk_tensors": disk,
       -expert drop "no_fake_density": policy is closed_acc,
        on this "_label": "MEASURED (live cohort:** 0 never nbytes + safetensors headers)",
-routed on O005        "_evidence": "MEASURED ( / O003 / O006complete_bpw = payload+scales. Any+biases+metadata drop is “+headers)",
    }
the```

=== /Users/scammermike/Downloads/hawking/tools/odyssey_patient_runner.py [L other top-k already implement the function,” not “this expert is unused.”

**2570-L2713Prior] ===
```
    live_ arttotal, organs_b = measure_ that boundslive_organ_ this lanebytes(model, moe=moe)
**

- GLM    disk = measure activation_dir_tensor_bytes(dest-aware **0.755)
    stored_ cos @ bytes0 =. int(167disk["stored_ BPW on realbytes"])
    params = int(census.get("total activations** — only_params") or real sub-1/5 0)
    precedent (SUB if params <= 0:
        raise1 packet SystemExit("census). Weight total_params missing; cannot compute stored-space GLM_bpw")
    stored_ sub-bit:bpw = stored **0.116–0_bytes * 8 / params
.157 cos @ 0    active_bytes.75 BPW**,, active_params = active_bytes below_from_organs(
        organs_b, live null **0.898_total, census, live
   ** (Type )
    # Prefer- on1).- Sedisk storedaled GLM artifacts_bytes for the at ** artifact; organ split0.88–0. from live nbytes.
    # Re-98 complete_scale organ bytes tobpw** are disk total if they differ (lazy vs **REFUSED** (`combust packed).
   ` / `rus`).
 if live_total > 0 and- Student live_total != **dense map @ stored_bytes:
        scale = stored 0.0104 BP_bytes / liveW**_total
        organs_b = works — {k: int **not weight-space**(round(v (`dead_levers.md * scale)) for k, v in`). Existence organs_b.items proof for()}
        active_bytes, active_params = active_bytes_from generate-the_organs(
-function.
- NS            organs_b, stored_bytes-raw, census, live-weight-PQ/
        )
   VQ @ active_bpw = (active_ ~1 bit: **DEADbytes * 8 / active_params**.
) if active_- NS-inter-expertparams else None
...-redundancy,
    accounting = measure_ NS-cross-expert/complete_accounting(model, dest,cross-layer tying params, moe=, NS-expert-mermoe)
    complete_bpw =ging: **DEAD** accounting.get("complete_bpw") on named
    if complete_bpw is parents (cosine ~ None:
        complete1e-4 / template_bpw = round(stored_ energy =bpw, 4)
...
 null /        "labels": {
            "stored merge error ~1_bytes": "MEASURED",
.0). Reopen only            "stored_bpw": " if **meanDERIVED (MEASURED bytes *  pairwise expert8 / census params)",
            " cosine ≥ 0.10complete_bpw** or merge": "MEASURED (payload+scales rel-+biases+metadata+headers) *error ≤ 0.5 8 / params",
            ".
- NS-nominal_bits": "DERIVED (kronecker: **ODYSSEY_POLICY.gravity_DEAD** depthspecs)",
            "candidate_class;": "DERIVED (ODYSSEY_POLICY.gravity_specs; deterministic)",
            "active_bytes_per_token": **LIVE on layer 0**.
- NS-global-dense-lowrank ( (
                "DERIVED (census activeQwen3.8):-param split × MEASURED organ bytes **DEAD**)"
            ), for global
            "active_bpw": "DERIVED",
            "battery": "MEASURED",
            " dense; does not auto-kill MoE crossdelta_hits":-expert LR "DERIVED (.
- Fthis MEASURED minus EXTERNALFN MEASURE block-D)",256 sparsity:
            "ver Typedict": "DERIVED",
           -1 dead "failure_localization": "DERIVED (scattered (rank organs by MEASURED sensitivity delta)",
        },
```

===; /Users/scammermike/ 0.2% skipDownloads/hawking/receipts/ @ odyssey-i/O00599% recall).
-_GRAVITY_q3- Low-rank residual codecg32-experts.json [L16: Type-1 dead (-LSVD energy low).
-40] ===
``` Learned LUT
  "stored codebook_bytes": 15362682880 on Apple GPU:,
  "stored_bpw": dead; QTIP Metal 4.0253,
  " decode:active_bytes_per_token": Type-1 dead.
 177313792- Hadamard generated0,
  "active_bpw transform: **REFUTED": 4.2305,
 **. G "params": 30532122624035: cross-layer,
  "active share at_params_per_token":  matched bits **REFUTE3353032704,
  "orgD**.
- NSans_bytes_quantized": {
-entropy-coded Lloyd    "embed": 175030272 PQ indices: ,
    "attn0.0–0": 509632512,
    ".7%,norm": 397312,
    " noiserouter": 7077888,
   .
- NS-posthoc "expert": -scalar-gain on14495514624,
    "lm k_head": 175030272
-means: pinned  },
  "disk_tensors at 1.0.
": {
    "- NS-ternstored_bytes": 153626828ary-factorization: loses80,
    " to VQ at matched ratetensor_count": 1351,
    "dtypes":.
- NS {
      "BF-uniform-subbit-16": 965,
      "Uallocation: dead32": 386; organ inversion
    },
    "_label": (gate "MEASURED (safetensors headers/up sensitive,, no weight load down tolerant) is the)"
  },
  "live replacement **_nbytes": 15362682880on,
```

=== foundry parents /Users/scammermike/, not yetDownloads/hawking Odyssey-measured**./receipts/odyssey-

---

## 3i/O005. Mechanisms_GRAVITY_q3-g32-experts

### M1 — Pr.json [L218-Lune-to-zero +226] ===
 sparse repair

**One line```
  "labels": {
   .** Delete "stored_bytes the component.": "MEASURED",
    " Whenstored_bpw the Doctor": "DERIVED (MEASURED bytes (or routed * 8 /-token KL) breaks, census params)",
    "active_ add thebytes_per_ smallest repair that restorestoken": "DERIVED (census active it (hi-param split ×-prec rows, MEASURED organ bytes)",
    " residual onactive_bpw": "DERIVED routed",
    " tokens, orbattery": "MEASURED",
    a skip in "delta_hits the gather).

**Byte": "DERIVED (this MEASURED accounting**

| minus EXTERNAL MEASURED Bin)",
    " | Formulaverdict": "DERIVED"
  | O },
```

005 1 expertMISSING /Users/scamm-index (Nermike/Downloads/hawking/=226,492,416receipts/od) |
|---|---|---|yssey-i
| Payload | /O005_GRAVITY.json0 | 0 |
|
MISSING /Users Drop flag +/scammermike/Downloads/ codec id | ≥hawking/receipt1s/odys B,sey-i/ billO005_GRAVITY_q 8–2-g3216 B | -experts.json
16 B |
| Alignment |MISSING /Users/scammermike ≤/Downloads/haw64 B | 64king/receipts/odyssey B |
| Router remap (-i/Ooptional) | L005_GRAVITY_q4 × E-g64.json bits, packed; or keep 128-way and emit 0 | 48×128 bits = 768 B if bitmask; 0 if no remap |
| Repair, k f16 rows of width m | `k · m · 2` | k=0 until break |
| Container descriptor | JSON name/shape/codec/hash | **UNKNOWN** (~0.2–1 KB/tensor typical; measure) |
| **complete_bpw, no repair** | `8 · (flag+pad+desc) / N` | **~3e-6** if desc ~64 B; **UNKNOWN** with real header |

Repair budget to stay in band (one 2048×768 organ, m=768):

| k rows kept f16 | Bytes | complete_bpw |
|---:|---:|---:|
| 1 | 1,536 | **0.0078** |
| 2 | 3,072 | **0.0156** |
| 8 | 12,288 | **0.0625** |
| 16 | 24,576 | **0.125** (exits 0.1) |

Unstructured residual at keep-fraction p, q8: `p · 1.0` BPW. p ≤ 0.09 for 0.1; p ≤ 0.002 for 0.01. FFN sparsity kill says p~0.002 **structured** is unlikely.

**stored_bpw** ≈ 0 + flag. **active_bpw** = 0 **only if** the gather **skips** the expert. mlx today: `full_expert_body_resident=true` (`O005_NX_gather`) — drop that does not change the kernel is a **stored-only** win.

**Expected reachable (honest)**

| Granularity | stored complete | Quality |
|---|---|---|
| 1 expert-index, all layers | ~0 (flags) | Fast battery **MEASURED Δ0** on O005/O006; **UNKNOWN** held-out |
| ~8–16 random (layer,expert) cells | ~0 | **UNKNOWN** |
| Keep-count binary search per layer | 0 on dropped cells | First break **UNKNOWN** (search it) |
| All-but-topk inventory | 0 on ~120/128 | **HIGH** Doctor risk; do not claim |

**Doctor risk.** Low on *one* expert-index vs the 12-item battery (**MEASURED**). High on mass drop. Battery-blind: a token still has 7 other experts (O005) / 5 (O003). Limiters: gate/up (foundry), router if remap is sloppy, coding/long-ctx (untested). Shared experts (O003): **do not prune**.

**Cheapest falsifier.** Not another 12-prompt run.  
Take ≥256 held-out tokens that **selected** the dropped expert (force-route if needed). Teacher vs dropped **KL / output-div on those tokens**. If KL is large, the battery was blind — kill “drop-without-repair” for that cell, try k-row repair, remeasure. Cost: one load, one organ, no pack.

**Native path.** Real win = **skip GEMM** (router mask / gather hole). Cost/token: <REDACTED> FLOPs, 0 DRAM for that cell. Fake: gather a zero tensor and multiply. No decode kernel. Repair of k rows = tiny GEMM, ~`2·k·m` FLOPs, native if the existing quantized GEMM already handles skinny N.

**Applicability.** MoE expert cells first (O005/O006/O003). Dense: single **layer × organ** (down first, if organ inversion replicates). Hybrid: single attn layer on O001 (all-attn still 5/12). Not: embed, lm_head, router, norms, SSM A_log/D/dt, shared experts.

**Confidence.** **MEASURED** that 1 expert-index can vanish on the fast battery. **HYPOTHESIS** that repair stays inside 0.01–0.1 when it breaks. **UNKNOWN** keep-fraction under real Doctor.

**Transfer.** Same Δ0 pattern on O005 and O006 (siblings). O003 hot-index already **−1** — transfer is not free. Dense/hybrid untested at this granularity.

**Prior art.** Wanda / SparseGPT / 2:4 (trained or activation-aware; unstructured is kernel-dead here). Not BitNet.

---

### M2 — Drop-to-constant / mean + correction

**One line.** Replace W with a constant or per-row mean; store only the residuals that matter.

**Byte accounting** (one 2048×768 organ, N=1,572,864)

| Variant | Bytes before correction | complete_bpw |
|---|---:|---:|
| 1 global f16 scalar | 2 + pad 64 | **3.4e-4** |
| Per-row f16 mean (2048 rows, gate/up) | 4,096 | **0.0208** |
| Per-row f16 mean (768 rows, down) | 1,536 | **0.0078** |
| Per-row f16 mean + f16 scale | ×2 | 0.0156–0.0416 |
| + correction keep p at q8 | + p·N·1 | add p |

To stay ≤0.10 with per-row mean on down: **p ≤ 0.092**. To stay ≤0.01: **p ≤ 0.002** — essentially no correction.

This is **not** NS-posthoc-scalar-gain (that is a gain on a k-means reconstruction, algebraically pinned at 1.0). This **changes the source** to a rank-0 / row-mean matrix.

**stored_bpw** = table above. **active_bpw** = same if the kernel is a broadcast-add (constant) or row-scale (mean); **higher** if you materialize dense W.

**Expected reachable.** Algebraic floor **0.008–0.021** on a single organ with **zero** correction. Whether Doctor holds at p=0 is **UNKNOWN**. If it needs p≳0.1, the mechanism leaves the band.

**Doctor risk.** High if applied to gate/up/attn/embed. Opportunistic on **down**, late layers, single expert cells. Mean-only is a large functional change; the 0.755 @ 0.167 GLM win was activation-aware, not “replace with mean.”

**Cheapest falsifier.** One layer, one down (or one expert down): `W ← row_mean(W)`. Hidden-state cosine / rel-L2 vs teacher on **real** routed activations (not Gaussian). If cosine ≲ 0.95 or output-div ≳ 0.1, p=0 is dead — do not pack. Then add the top-p residual by activation-weighted |r| and remeasure. Kill if p needed > 0.09.

**Native path.** Constant: `y = c · 1ᵀ x` = `c · sum(x)` — a reduction, not a GEMM. Per-row mean: `y_i = μ_i · sum(x)` — one mul per row. Cheapest decode on the machine. Correction: sparse or skinny GEMM. **Do not** expand to dense.

**Applicability.** Same as M1, plus **dense down** (O004) and **O001 MLP down** as opportunistic cells. Not router/norms (zeroing already kills).

**Confidence.** Algebra **DERIVED**. Quality **HYPOTHESIS**. FFN-sparsity Type-1 kill makes a tiny unstructured p unlikely.

**Transfer.** Mean-replacement is architecture-agnostic. Tolerance will not be — measure per organ.

**Prior art.** Act-aware pruning residuals; AQLM’s first stage is a coarse codebook (mean is the k=1 codebook). QuIP# incoherence does **not** make a mean a good code.

---

### M3 — Shared codebook index (amortized)

**One line.** The component stores an **index** (or a short index list) into a table used by many components. Near-zero is **amortization**, not a free lunch.

**Byte accounting — two geometries**

**A. Whole-component template** (one index for the entire expert / organ)

```
table = K templates × N_comp × bytes_w
share_c = table / U          # U = users of this table
local  = ceil(log2(K)/8)     # 1 B for K≤256
complete_bpw = 8 · (share_c + local + pad) / N_comp
```

O005, one-layer expert (N=4,718,592), **K=16 f16 templates**:

| Users U | Share / component | complete_bpw |
|---:|---:|---:|
| 128 (same layer, all experts) | 1.18 MB | **2.00** — index-only is a lie |
| 6,144 (128×48) | 24.6 KB | **0.0417** |
| 6,144, templates q3 (0.375 B/w) | 4.61 KB | **0.0078** |

**B. Product-quant subvectors** (AQLM / GLM `glm52.pq`)

```
nominal = (N / dsub) · log2(K) / N = log2(K) / dsub
+ codebook: U_groups · K · dsub · bytes_w / (N · U_share)
```

| dsub | K | nominal index BPW | Notes |
|---:|---:|---:|---|
| 8 | 256 | **1.00** | not this lane |
| 64 | 256 | **0.125** | edge of 0.1 |
| 256 | 256 | **0.031** | + codebook share |
| 256 | 16 | **0.0156** | only if clusters exist |

Lloyd-optimal indices are near-uniform (NS-entropy-coded-PQ: **0.0–0.7%** extra). Do not bill an entropy miracle.

**G035** already killed **cross-layer sharing at matched bits** on Qwen3.8. NS-cross-expert tying: best shared template explained **0.2513 vs null 0.2500** on Qwen3-235B. NS-inter-expert cosine **1e-4** on gpt-oss-120b.

**stored_bpw** = complete (table once + all indices). **active_bpw** = indices of **touched** components + **whole table** if the LUT is resident (table can dominate active).

**Expected reachable.** **0.01–0.04 complete** only if (i) U is thousands and (ii) K-means energy actually concentrates. On current MoE parents the concentration premise is **presumptively dead** until cosine ≥ 0.10. Subvector PQ at 0.03 is a **raw-weight** codec — NS-raw-weight-PQ @ 1 bit says quality dies before you get there **unless the source changes**.

**Doctor risk.** Extreme on raw-weight PQ (GLM 0.75 BPW collapse; qwen3-235b A1_1p0 6/6 collapse). Template-index: extreme unless cosine gate reopens. LUT gather **punished on Apple GPU** (`dead_levers`).

**Cheapest falsifier.** **Do not build a codebook.** Measure mean pairwise cosine and energy explained by K∈{2,4,16} templates on **this** patient’s experts (one mid layer, gate/up/down separately), row-normalized and raw. If cosine < 0.10 and explained energy ≈ 1/K, **NS applies — stop**. Cost: load one layer’s expert stack, no Doctor.

**Native path.** Hostile. Learned LUT gather: dead. QTIP-class lookup-free: Metal trellis decode Type-1 dead. A “0.03 BPW” index that **expands to f16 then GEMMs** is a fake sub-0.1 win (reconstruction FLOPs + workspace). Only native if there is a **direct compact kernel** that never materializes W. None exists here for this geometry (**UNKNOWN** whether one can be written; current Metal evidence is negative).

**Applicability.** Only after the cosine/energy gate. If it reopens: MoE expert organs with large U. Dense: only if many layers share a basis (G035 says no on Qwen3.8). Hybrid SSM tensors: **UNKNOWN**, do not assume.

**Confidence.** Amortization algebra **DERIVED**. Quality on Odyssey MoE **HYPOTHESIS, prior = dead**. Do not spend a pack until the cosine gate.

**Transfer.** A kill transfers with the premise (orthogonal experts). A win would be parent-specific (`reopen_if` cosine ≥ 0.10).

**Prior art.** PQ / AQLM / QTIP / QuIP#. GLM `glm52.pq.r0`, `glm52.indexshare.student`. IndexShare is a **student**, not a raw-weight index.

---

### M4 — Cross-component low-rank factor

**One line.** `W ≈ UVᵀ` (and optionally a shared V across experts/organs) + sparse residual. Near-zero is **rank**, not bits.

**Byte accounting** — 2048×768 organ, factors f16:

| r | Bytes `r(n+m)·2` | complete_bpw |
|---:|---:|---:|
| 1 | 5,632 | **0.0286** |
| 2 | 11,264 | **0.0573** |
| 4 | 22,528 | **0.1146** (exits 0.1) |
| 1, factors q3 | ~1,056 | **0.0054** |

Shared V across 128 experts, one layer, r=4, one organ:

```
U: 128 · 2048 · 4 · 2 = 2,097,152
V: 768 · 4 · 2 = 6,144
per expert: 16,432 B → complete_bpw = 0.0836
```

Cross-layer shared V (U per layer): amortize V further — **G035 / tying kills** unless spectrum is peaked.

Kronecker `A ⊗ B`: NS-kronecker — top component **0.27%** of gate energy at depth; **LIVE on layer 0 only**.

**stored_bpw** as above + residual. **active_bpw** = same if executed as two skinny GEMMs (`xV` then `U(·)`). If you form W, you pay dense and lose the win.

**Expected reachable.** **r=1 f16 = 0.029** on this organ shape is the honest **algebraic** 0.01–0.1 candidate. Quality at r=1 is the question. Foundry low-rank residual = Type-1 dead (energy not peaked). Global dense LR dead on Qwen3.8.

**Doctor risk.** High on gate/attn. Possible on a **single late down** or **L0** (different source; Kronecker beat the incumbent there). Shared-V across experts inherits NS-inter-expert-redundancy.

**Cheapest falsifier.** SVD of **one** down, **one** gate, **one** late expert, **one** L0 expert. Report energy in r=1,2,4,8. If r=1 < ~50% energy, r=1 is dead as a standalone (still usable as a base under M1/M2 repair). Then one real-activation forward: `ŷ=U(Vᵀx)` vs teacher. No pack.

**Native path.** Two GEMMs, shapes `(1×m)×(m×r)` and `(1×r)×(r×n)` — trivial Metal/AMX, no LUT. Decode cost ≈ `2r(n+m)` FLOPs vs `2nm`. At r=1, ~2800 vs 1.57e6 FLOPs (~500× less). This is one of the few near-zero forms that is **natively cheaper**, not just smaller. Runtime is often kernel-bound — skinny GEMM must not fall off the fast path (**UNKNOWN** on this box for r=1).

**Applicability.** Per-tensor LR: any organ. Cross-expert shared V: MoE only, after cosine gate. L0 Kronecker: named exception, **do not transfer to depth**. Dense O004 MLP: same SVD test. O001 SSM: **UNKNOWN** (A_log/D are already structured; do not LR them blindly).

**Confidence.** Floors **DERIVED**. Quality **HYPOTHESIS, prior = dead except L0**. Worth the SVD because the falsifier is minutes.

**Transfer.** Spectrum shape is parent- and layer-specific. L0 win does not transfer to L≥1 (already measured on Qwen3-235B).

**Prior art.** LoRA / ASVD / SVD-LLM / QuIP# (incoherence + lattice, not this). BitNet is not low-rank.

---

### M5 — Structural collapse of redundant experts

**One line.** Cut inventory E → K (or drop cells). Remaining experts **are** the function. This is **not** “reconstruct the omitted expert from survivors.”

**Two different bets**

| Bet | Status |
|---|---|
| Omitted expert ∈ span(survivors) — merge / tie / shared template | **DEAD** (NS-expert-merging error ~1.0; tying = null) |
| Tokens that would have used e are already served by the other top-k | **ALIVE as a hypothesis**; 1-index drop **MEASURED Δ0** on the fast battery |

Do not resurrect merge. Descend **keep-count**.

**Byte accounting**

```
dropped cell: M1 flag bytes (~0 BPW)
kept cell:   whatever codec it already has (q3-g32 ≈ 4.0 complete)
router:      if remapped to K, save L·(E−K)·hidden·bytes_w
             O005 router is 12.6 M params — not the prize
```

Illustrative **total** (not per-component), O005, protected organs stay ~4 BPW, dropped experts ~0:

| Keep / layer | Expert contribution | Rough total complete | Class |
|---|---:|---:|---|
| 128 @ q3 (today) | ~3.8 | **4.03 MEASURED** | anchor |
| 16 @ q3, 112 dropped | ~0.50 | **~0.7–0.9** | **DERIVED sketch** |
| 8 @ q3 | ~0.25 | **~0.45–0.7** | sketch |
| 8 @ 0.1 (M4/M2) | ~0.006 | **~0.20–0.25** | sketch; protected attn+embed+head dominate |

These totals are **sketches** (protected-organ complete BPW after mix is **UNKNOWN** at the 4-bit mlx split). Use them as a search map, not a claim.

**stored_bpw** of a dropped cell: ~0. **active_bpw**: 0 if skipped; if the router can still pick a hole, you either no-op (good) or gather zeros (fake).

**Expected reachable.** Per dropped cell: **~0**. How many cells drop before held-out Doctor dies: **UNKNOWN**. That number *is* the experiment.

**Doctor risk.** One index: low vs fast battery, **UNKNOWN** vs held-out. Mass collapse: high. O003 hot-index already **−1**. Limiters: tasks that specialize through a dropped expert; load-balance shift after remap; layer 0.

**Cheapest falsifier.** Geometric keep-count, **one mid layer first** (not all 48): keep {127, 96, 64, 32, 16, 8} experts, drop the rest (zero + skip). Fast battery + **routed-token KL on dropped IDs**. Stop at first break, then step back one and run real Doctor. Do **not** start with 48-layer × 112-drop.

**Native path.** Same as M1: gather skip + optional router reshape. `selected/full` shrinks (NX), stored shrinks (density). mlx must stop residencing the holes.

**Applicability.** MoE only (O005/O006 primary; O003 has shared experts — never collapse those). Dense: the analogue is **layer deletion / FFN drop**, which is a different and harsher bet (all-MLP zero already −9/−11). Hybrid: dropping attn layers is the O001 analogue (all-attn still 5/12).

**Confidence.** Distinction merge-vs-drop is **MEASURED** (merge dead, 1-drop battery-alive). Mass keep-count **HYPOTHESIS**.

**Transfer.** O005↔O006 likely. O003 hotter tail (top-16 mass **31%** vs O005 **18%**) — drop the tail first, protect the hot set. DSV4F / GLM later: **measure cold fraction**; 0-cold is not universal (rule `R-uniform-routing-no-cold-compress`).

**Prior art.** Expert pruning / MergeMoE / MoEfication. MergeMoE is the killed bet. Router-aware pruning is this bet.

---

### M6 — Procedural / generated component

**One line.** Do not store W. Store a seed + a small generator (rules, hypernet, or activation-space student) that **produces the function**.

**Byte accounting**

```
complete_bytes = |θ_generator| + seed + program/bytecode
               + any table G needs
               + residual/correction
               + persistent workspace if it cannot be scratch
amortize |θ| across every component G serves
complete_bpw_c = 8 · (share_c + seed_c + corr_c) / N_c
GENERATED_BPW_EQUIVALENT = 8 · |θ_amortized| / N   # already an axis
```

**Illustrative only (NOT a measurement)** — one hypernet, 10 M f16 params = 20 MB, serves 6,144 O005 (layer,expert) cells of 4.72 M weights:

| | |
|---|---|
| Share / cell | 3.4 KB |
| complete_bpw | **0.0058** + residual |
| If it only serves 128 expert-ids (layer-tied) | 156 KB / cell → **0.265 BPW** |

Hadamard (one tested generated transform) is **REFUTED**. Weight-space generate is unconstrained and easy to fake.

The **real** prior is the GLM **student dense map @ 0.0104 BPW, not weight-space**: match **activations**, not W. That is already inside the 0.01 band **if** complete accounting of the student + any teacher-side leftover is honest. Leftover teacher bytes were the Qwen3.8 3.61→4.15 trap — charge them.

**stored_bpw** = generator + seeds + residual. **active_bpw** = generator working set **per token** if G runs every token, or 0 extra if G is baked at load into a compact executable. Baking into dense W = **fake**.

**Expected reachable.** **0.01-class is the point of this mechanism** (GLM student). Whether an Odyssey expert cell has a similarly cheap activation student is **UNKNOWN**. Weight-space hypernet: **UNKNOWN**, prior generated-transform = dead.

**Doctor risk.** Highest of the six if G is wrong (silent semantic collapse — GLM `combust`/`rus` at 0.88–0.98 **complete**, integrity sealed). A gibberish 0.01 artifact is still valuable **if it names the organ that broke**.

**Cheapest falsifier.** Do **not** train a hypernet first.  
Fit a **tiny linear student** `y ≈ A x` (or `A diag(s) x`) on **real routed activations** of **one** expert-down, held-out tokens. Compare to (i) teacher, (ii) **mean baseline** (M2), (iii) **r=1** (M4). If the student does not beat M2/M4 on output-div, procedural is not cheaper than those and is dead **for that cell**. This is the 0.0104 result’s cheapest relative.

**Native path.** Two honest stories only:

1. **Load-time generate** into a compact native form (binary / LR / skip) — bill G once, execute the compact form. Generation FLOPs are compile cost, not token cost.
2. **Fused generate+GEMM** that never writes dense W — bill G’s active working set every token.

Expand-then-GEMM at decode: **reject**. Decode cost **UNKNOWN** (no kernel). Runtime is often kernel-bound; extra generate ALU can lose even if bytes win (`R-sparse-active-expert-gather` reopen_if).

**Applicability.** Expert **function** replacement (MoE) is the only place with a Hawking existence proof. Dense FFN: same student test. Embed/lm_head: do not generate. SSM recurrences: **UNKNOWN**, high risk. Layer 0: test separately.

**Confidence.** 0.0104 student is **MEASURED** on GLM (weight-space no). Odyssey transfer **UNKNOWN**. Weight-space G **HYPOTHESIS, prior generated = dead**.

**Transfer.** Activation-space students can transfer as a *method*; the student itself is parent-specific. Do not ship a GLM student against Qwen.

**Prior art.** Hypernetworks (Ha 2016), SMASH, implicit neural reps, BitNet/1-bit **QAT** (source-changing; not PTQ). QuIP# / QTIP change coding, not generation. The Hawking object is `glm52.functional.block.v1` / `glm52.indexshare.student.v1`.

---

## 4. Sensitivity-guided descent (how to attack)

Policy already says: geometric / binary descent per component, repair at the boundary, Doctor + false-win gates stop you.

**Order (info / cost)**

1. **Cosine / SVD / energy gates** on one layer — kills M3/M4/M5-merge in minutes if NS applies.
2. **M1 drop** of 1 expert-index (already done) → routed-token KL (real kill test).
3. **M5 keep-count** on one mid layer: 127→8.
4. **M2 mean** and **M4 r=1** on the first cell that M1 cannot drop — activation cosine, not Doctor.
5. **M6 student** only on cells where M2/M4 lose (need a cheaper function approx).
6. **M3 codebook** only if cosine ≥ 0.10.
7. Full Doctor only at the first boundary that cheap probes pass.

**Do not near-zero:** embed, lm_head, router, norms, shared experts, SSM state tensors, *all* experts, *all* attn on MoE.

**Push hard:** single (layer, expert, organ), especially **down**; non-hot expert-ids; O001 **per-layer attn**; any cell whose routed-token KL stays small when zeroed.

**L0:** opposite instinct. Different source; Kronecker live; do not drop first.

---

## 5. Native / NX (physical object)

| Mechanism | Native object | Decode / token | Kernel risk |
|---|---|---|---|
| M1 / M5 drop | hole in gather + flag | 0 if skipped | mlx still residents full body **MEASURED** — must change residency |
| M2 constant/mean | broadcast / row-scale | O(n) or O(1) | easy; do not materialize |
| M3 codebook | LUT or lookup-free | LUT gather **dead** on Apple GPU; QTIP decode **Type-1 dead** | **worst** native story |
| M4 LR | two skinny GEMMs | `2r(n+m)` FLOPs | good if r=1 stays on the fast path **UNKNOWN** |
| M6 generate | compact baked form **or** fused G | load-time = 0; fused = **UNKNOWN** | expand-to-dense = reject |

**Stored vs active.** Dropping 112/128 experts is a **stored** win. Active/token only falls if the **selected** set also got cheaper, or if holes are not resident. O005 selected/full = **0.0625** is already the gather lever; it is **not** 16× cheaper compute (`ratio_selected_over_dense_mlp = 1.0` — selected bytes equal a dense SwiGLU of `intermediate=6144`).

---

## 6. Transfer

| | O005 | O006 | O003 | O001 | O004 |
|---|---|---|---|---|---|
| 1 expert-index drop, fast Δ | 0 / 0 | 0 / 0 | −1 hot / 0 random | n/a | n/a |
| Shared experts | none | none | **protect** | n/a | n/a |
| Cold-expert lever | none | none | none | n/a | n/a |
| Parallel attn/SSM | no | no | no | **attn more droppable** | no |
| Dense FFN cell drop | — | vision MLP not in language specimen | 1 dense layer, Δ −11 if all zero | MLP Δ −9 all-zero | **UNKNOWN** |

A rule is not universal after two Qwens (`TRANSFER_MATRIX` doctrine). Re-run the cosine/SVD/1-drop KL on each new parent.

---

## 7. What this lane will not claim

- A **uniform** 0.01–0.1 codec on raw weights. That family is dead near 1 bit.
- That fast-battery Δ0 is capability. It is a **localization** that a cell is a candidate.
- That a 2-byte index is 0.01 BPW. Charge the table.
- That mlx 4-bit round-8 says an organ is 8-bit-safe. It is a no-op on already-4-bit weights.
- Exact Doctor-holding keep-counts, p, r, or K on Odyssey patients — all **UNKNOWN** until the falsifiers run.
- Q80 1.43 mixed and GLM 0.167 @ 0.755 as *this* lane’s result. They are priors: mixed complete **1.43** is still ≫ 0.1 per component; 0.167 is activation-aware, not a drop.

**Bottom line.** The only **MEASURED** near-zero on current patients is **M1/M5 at granularity = one expert-index**, and only against a 12-item battery. The only **MEASURED** 0.01-class *function* replacement in Hawking history is the **GLM activation student @ 0.0104**, which is M6, not weight PQ. Everything else in the 0.01–0.1 band is **algebraic** (M2 row-mean, M4 r=1, M3 with huge U) and must pass the cheap falsifiers before anyone packs a Gravity shard.
