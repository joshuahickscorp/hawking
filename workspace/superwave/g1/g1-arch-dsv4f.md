# G1-ARCH-DSV4F — transferable mechanism, not the vehicle

Lane: `22-arch-dsv4f`. Base: `2eee9a004` on `grok/22-arch-dsv4f-20260817-105236`.
No GPU run. No artifact produced. Numbers below are **MEASURED** (receipt field),
**COMMIT** (commit message), **CITED** (campaign standing law with no receipt
found), or **COMPUTED** from MEASURED fields.

G0 vehicle this transfer is judged against: resident Genesis body on
`workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1` via
`Qwen38HybridWeights::load` (`receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.md`).

---

## 0. The 78 s → 3.375 s arc is real. It is not three identity steps.

| wall | label | source |
|---:|---|---|
| 77.993 s | CPU streamed 43-layer BOS, `metal_dispatches: 0` | MEASURED `receipts/dsv4f_streamed_forward_l0_l42_receipt.json` L95 `wall_ms: 77993`, L10 `execution_path: host_cpu_streamed_bos_oracle`, L15 `metal_dispatches: 0`, L16 `token_id: 5`, L14 `logit: 16.767436981201172` |
| 28.980 s | same token after Metal linears | MEASURED `receipts/dsv4f_streamed_metal_l0_l42_receipt.json` L285 `wall_ms: 28980`, L10 `host_cpu_streamed_bos_oracle_metal_linears`, L27 `metal_dispatches: 3343` |
| 30.258 s | native-graph + zero-copy **cold** (init 20.110 + body 10.147) | MEASURED `receipts/DSV4F_NATIVE_PLUS_ZEROCOPY_COMBINED.json` L106 `wall_ms: 30258`, L34 `init_ms: 20110`, L8 `body_ms: 10147`. COMMIT `77be38ca6` |
| 3.375 s | same native graph after mmap artifact index | MEASURED `receipts/DSV4F_NATIVE_INDEX_ADMISSION_COLD_TOKEN.json` L123 `wall_ms: 3375`, L45 `init_ms: 177`, L8 `body_ms: 3198`, L17 `hash_invocations: 0`, L14 `bytes_hashed: 0`. COMMIT `368907605` |

COMMIT `c881b0227` names the first cut: "78.0 to 28.98 s/token". That step is
**compute moved off the CPU**, not identity bookkeeping. Layer-0 profile in
that message: MLA linears ~36%, routed expert triple 24%, `act_quant` 9.1%,
`streaming_io` 21.8%. Only the last bucket is identity/IO.

**FALSIFIED:** "the three eliminations that took a token from 78 s to 3.375 s
were all identity bookkeeping." The 78 → 28.98 step is a GPU port. Identity
eliminations sit on the later 30.258 → 3.375 (and 25.5 → 5.8 / 11.41 → 3.31)
windows.

The three identity *classes* named in
`workspace/ops/ascent-lanes/dsv-admission-identity.md` (per-token SHA,
clone-tree-on-open, `st_dev`) are real and separately evidenced below.
`st_dev` is a Q30/complete-binary finding, not a DSV4F token-wall step.

---

## 1. Hash per token

### Mechanism

Every DSV4F tensor read SHA-256'd the content-addressed chunk, then copied
into a Metal buffer. COMMIT `9907b0ebe`:

> Every tensor read was SHA-256ing the entire content-addressed chunk to
> extract a slice, then copying it again into a Metal buffer. … roughly
> 1,502 calls and ~10 GB rehashed per token, about half of profiled wall.
> Resident second token 25.523s to 5.763s, 4.43x.

Then COMMIT `fe13db5a6`:

> admission-time chunk trust - token body 11.41s to 3.31s (3.44x), zero
> bytes hashed per token

One-time seal cost (not a token): `receipts/dsv4f_admission_pass.json`
`hash_wall_ms: 47205`, `bytes_hashed: 159609485896` (159.61 GiB, 69837 chunks),
`example_wall_ms: 75053`. Field `claim`: "one-time full SHA-256 admission;
not a token".

Live skip is in `crates/hawking-core/src/gravity_deepseek_v4.rs`:

```168:172:crates/hawking-core/src/gravity_deepseek_v4.rs
/// `hash_invocations` increments only when SHA-256 actually runs.
/// A second read of an already-verified `(chunk, digest)` is a cache hit
/// and does not re-hash.  `admission_trust_hits` counts first-touch chunks
/// whose sealed receipt identity matched so SHA-256 was skipped.
```

```1485:1517:crates/hawking-core/src/gravity_deepseek_v4.rs
        if cached {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(mmap);
        }
        // ...
        if trusted {
            self.admission_trust_hits.fetch_add(1, Ordering::Relaxed);
            self.verified_digests.lock().insert(binding.sha256.clone());
            return Ok(mmap);
        }
        self.hash_invocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_hashed
            .fetch_add(mmap.len() as u64, Ordering::Relaxed);
        let observed = sha256_hex(&mmap);
```

After the mmap-index cold token: `hash_invocations: 0`, `bytes_hashed: 0`,
`admission_trust_hits: 3314`
(`receipts/DSV4F_NATIVE_INDEX_ADMISSION_COLD_TOKEN.json` L12–L17).
HOST_WALL later: `validation_identity.sha_ns: 0`
(`receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json` L272).

Remaining identity tax (path_resolve + stat, **parallel sum**, not wall):
3314 calls, `path_resolve_ns_parallel_sum_median: 1318045536`,
`verify_ns_parallel_sum_median: 2505505105` (HOST_WALL L274–L275).
COMMIT `766670496` **REFUTED** that this was on the critical path:
path+identity parallel-sum 983 → 343 ms, token A1 1231 vs B1 1303 ms,
"ZERO wall win".

### Qwen3.8 — can this class exist?

**Not on the current G0 token path.** `Qwen38HybridWeights::load` reads each
catalog file and uploads; `step` / `step_complete` do not hash.

```508:536:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        pub fn load(root: impl AsRef<Path>) -> Result<Self> {
            // ...
            let (_manifest, rows) = load_qwen38_manifest(root)?;
            // ...
                let payload = fs::read(&path).map_err(|error| {
                    Error::Model(format!("cannot read {}: {error}", path.display()))
                })?;
```

`rg Sha256|sha256_hex|hash_invocations crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
is empty (this worktree). The only Qwen3.8 SHA-256 in `qwen38_pack.rs` L271–274
hashes the **tensor name** to build a filename at pack time.

`step_complete` bookkeeping is `tokens.push(sampled)`
(`qwen38_hybrid_decode.rs` L3517–L3528), not a digest.

**Can return** if G1 adds a content-addressed chunk reader, a per-tensor
integrity hash inside `step`, or complete-binary cold rehash on the serve
path.

### Exact check (no GPU)

```
rg -n 'Sha256|sha256_hex|hash_invocations|bytes_hashed' \
  crates/hawking-core/src/model/qwen38_hybrid_decode.rs \
  crates/hawking-core/src/model/qwen38_pack.rs
```

Rule out on the token path: zero hits inside `step`, `step_complete`,
`encode_layers`. A hit inside `load` / `load_mixed` / `pack_*` is admission
or pack, not per-token.

If a new reader grows `hash_invocations`: compare a two-token process
(`token 0` vs `token 1` in the same pid). Per-token hash is confirmed iff
`hash_invocations` and `bytes_hashed` increment on token 1. Already-killed
shape is `token1.hash_invocations == 0 && token0` paid once.

---

## 2. Clone tree on open

### Mechanism

JSON parse was **not** the bulk. MEASURED
`receipts/dsv4f_artifact_index.json`:

- L14 `json_parse_is_not_the_bulk: true`
- L15 `bulk_phase: prepare_clone_chunks`
- L16: `stream-ranges.jsonl` appended to 139674 = 2×69837 lines. Every
  process start `clonefile`/`cp -cR`'s the 69837-file chunk tree so admit
  can hash a sealed prefix. 10.4–12.2 s = 88% of init. Manifest+ranges+
  admission JSON parse ~1.7 s.
- L87 `prepare_clone_chunks_ms: 11053` (JSON-path mean)
- L83 `json_path_mean_wall_ms: 13413`
- L84 `index_path_mean_wall_ms: 139` (96.27×)
- L106 after index: `prepare_clone_chunks_ms: 0`, `json_parse_ms: 0`

Source that still exists as the JSON fallback:

```637:664:crates/hawking-core/src/gravity_deepseek_v4_streamed_forward.rs
    let result = crate::startup_timing::time_ms_result("prepare_clone_view", || -> Result<()> {
        // clone manifest / restart-receipt / stream-journal / ranges prefix
        crate::startup_timing::time_ms_result("prepare_clone_chunks", || {
            clone_or_link_tree(&source.join("chunks"), &view_root.join("chunks"))
        })?;
```

`macos_clonefile` at L769–794; `cp -cR` fallback at L739.

Cold-token effect of skipping the clone-view: 30.258 s → 3.375 s
(§0). Init 20.110 s → 177 ms. COMMIT `3236cefb6` open 13.4 s → 139 ms
matches the 3-rep means L113–L115 vs index 138/139/141 ms.

### Qwen3.8 — can this class exist?

**Not as clonefile of a 70k-file tree.** `rg clonefile|clone_or_link_tree|cp -c`
over `qwen38_*.rs` is empty.

G0 load is 755 individual `fs::read`s (`QWEN38_EXPECTED_CATALOG_TENSORS =
402+353`) plus Metal upload. MEASURED cache-hot `load_ns` 4.908 s
(`GENESIS_RESIDENT_BODY.md` L16). Same file **CITED** "~50 s box-cold load"
(L11) with no receipt attached — treat 50 s as unverified.

This is the same *class* (immutable tree walk at open) at ~100× fewer files
and without `clonefile`. It is paid once per process; the resident body
already stopped re-paying it per `genesis_proposes()` (L3–L5, L19
`load_count = 1`).

**Can return** if G1 builds a sealed-prefix admit view by cloning the
artifact, or if the loop falls back to oneshot `ascension_qwen38_hybrid_greedy`
(load+4-token+exit 4.649 s cache-hot, same receipt L15).

### Exact check (no GPU)

```
rg -n 'clonefile|clone_or_link_tree|prepare_clone|cp -cR' \
  crates/hawking-core/src/model/qwen38_hybrid_decode.rs \
  crates/hawking-core/src/model/qwen38_pack.rs \
  crates/hawking-core/src/model/qwen38_host_admission.rs
```

Empty ⇒ clone-tree-on-open is absent. Non-empty ⇒ count files cloned and
time `prepare_clone_*` with `HAWKING_STARTUP_TIMING=1` around `load` only
(no generate). Confirm/rule-out: clone phase > 10% of `load_ns` on a
page-warm process that does not decode.

---

## 3. `st_dev` device identity (Q30, same class)

### Mechanism

APFS remount reassigns the volume device number. Matching on `st_dev`
forced a full cold rehash.

COMMIT `091fe51cf`:

> Drop st_dev from admission identity hard-fail set
> Remount reassigns APFS volume device numbers without touching files.
> Matching on st_dev forced a full cold rehash after every remount.
> Identity remains (size, mtime_ns, inode); content SHA still binds.

Observed remount in source comment
`crates/hawking-core/src/model/qwen_complete_binary/mod.rs` L1445–1450:
`16777234 -> 16777233`, inode + bytes byte-identical.

Current match key (device recorded, not compared):

```133:142:crates/hawking-core/src/model/qwen_complete_binary/admission_warm_receipt.rs
fn identity_matches(observed: &FileIdentity, expected: &FileIdentity) -> bool {
    // device (st_dev) is deliberately NOT compared
    observed.size == expected.size
        && observed.mtime_ns == expected.mtime_ns
        && observed.inode == expected.inode
}
```

Unit test `device_shift_alone_still_matches` L668–672 uses those exact
device numbers.

COMMIT `091fe51cf` patch comment (not in HEAD text): "full ~10s cold rehash
on every post-reboot startup."

**CITED only — no receipt found in-tree** for the standing-law numbers in
`workspace/ops/ascent-lanes/dsv-admission-identity.md` (introduced
`ac0f5d716` as new prose, not copied from a JSON field):

> an admission check keyed on `st_dev` — a mount artifact — that alone
> cost 28 s of startup and, once dropped, took startup 13.5 s -> 2.3 s
> and a warm repack 4606 s -> 95 s (48x).

4606/95 = 48.484… (arithmetic identity). Those four numbers are **CITED**,
not independently MEASURED here. `git grep 4606` / `13.5 s` over receipts
did not hit a receipt field. `REOPEN_IF`: a Q30 startup-timing JSON with
`admit_complete_binary_total` before/after `091fe51cf` appears.

What **is** MEASURED: the defect class (false invalidation on remount) and
the code fix.

### Qwen3.8 — can this class exist?

**Not on the G0 uniform-Q4 load path.** `Qwen38HybridWeights::load` does not
call `file_identity` / `identity_matches`. Genesis artifact is
`uniform-q4-v1`, not a complete-binary warm receipt.

**Yes on any path that reuses complete-binary admission** (mixed HQ30 / 
HGRAVS01 catalogs, a future G1 pack that seals via
`qwen_complete_binary`). That path already excludes `st_dev`. A *new*
identity tuple that re-adds `device` reopens the bug.

### Exact check (no GPU)

```
rg -n 'observed.device == expected.device|metadata.dev\(\)|st_dev' \
  crates/hawking-core/src/model/qwen_complete_binary/admission_warm_receipt.rs \
  crates/hawking-core/src/model/qwen38_hybrid_decode.rs \
  crates/hawking-core/src/model/qwen38_pack.rs
```

Rule out: `identity_matches` does not compare `device`; qwen38 load has no
`metadata.dev()`. Confirm regression: add `device` to the `&&` chain and
the existing test `device_shift_alone_still_matches` must fail.

If G1 writes a new identity struct, the check is the same comparison:
does a remount-only `device` flip invalidate the receipt?

---

## 4. Expert slab IO and the "73% GPU idle" finding

### Expert slab — MEASURED

`receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json`
(DIRTY_ENGINEERING, warm R2–R6, 137/137 CBs timestamped):

- L38 `host_exclusive_ms_median: 564.9`
- L41 `metal_gpu_ns_median: 399023000` (399.0 ms)
- L47 `gpu_idle_host_ms_median: 410.3`
- L178: GPU-idle host ≈ `host_exclusive - (metal_gpu - metal_wait)` ≈ 410 ms
  "and matches expert_slab_io"
- L180 `file_source_reads.ns_per_token_median: 415126416` (415.13 ms)
- L181: `host.expert_slab_io` dominates 406–437 ms; 3,449,290,752 bytes
  (top-6 routed FP4+scale)
- L287–L290 `next_bottleneck`: `host.expert_slab_io` 415.13 ms,
  "streamed top-6 expert read + metal fill, GPU-idle"

`receipts/ascent-2026-08-16/TOKEN_NS_DSV4F.json` L22:

> DSV4F critical path is GPU-idle expert_slab_io then metal.wait/GPU,
> plus host MHC between CBs.

COMPUTED from HOST_WALL: `415.13 / 564.9 = 0.7349` — **73.5% of
host_exclusive is GPU-idle expert-slab IO**. That is the only 73% I can
attach to a receipt.

Later: COMMIT `62827f9f8` / `4b5fde4ae` cut GPU-idle expert mmap
52–64 ms → 24 ms / 50 → 22 ms via admitted fast map. COMMIT `9f1b2dff1`
+ `receipts/ascent-2026-08-16/DSV4F_HOST_EXCLUSIVE_HALVED.json`:
host_exclusive median 460.7 → 221.0 ms (−52%), body 849.9 → 700.3 ms.

### GPU idle — MEASURED, not 73% of token

`receipts/ascent-2026-08-16/TOKEN_NS_DSV4F.json` L804–L809:

| field | ns | COMPUTED / token |
|---|---:|---:|
| `TOTAL_TOKEN_NS` | 1,024,054,500 | 1.000 |
| `TOTAL_GPU_BUSY_NS` | 396,549,000 | 0.387 |
| `TOTAL_GPU_IDLE_NS` | 627,505,500 | **0.613** |
| `TOTAL_GPU_GAP_NS` | 997,680,294 | 0.974 (first-to-last-CB gap; not a fraction of body) |
| `TOTAL_CPU_CRITICAL_NS` | 553,797,500 | 0.541 |

No receipt field equals 0.73 GPU-idle-of-token. Standing law
`workspace/ops/ascent-lanes/_COMMON.md` L39–40 cites **Q80**
"~51% GPU idle" (dispatch/host bound, 0.79% of 700–800 GB/s).
Q80 `receipts/ascent-2026-08-16/TOKEN_NS_Q80.json`:
`TOTAL_GPU_IDLE_NS/TOTAL_TOKEN_NS = 327203133/559171655 = 0.585`.

`gpu_idle_fraction_of_span` is defined
(`gravity_deepseek_v4_token_ns_ledger.rs` L546–551) as
`(device_span - busy) / device_span`. No checked-in DSV4F receipt
emits that field.

**Honest 73%:** 73.5% of DSV4F *host exclusive* is GPU-idle expert-slab
IO. Not "GPU idle 73% of the token."

### Qwen3.8 — can this class exist?

**Expert slab: no.** Qwen3.8 language model is dense SwiGLU. SUPERWAVE
axis note (CITED `SUPERWAVE_STATE.md`): "EXCLUDES Qwen3.8 -- it is DENSE,
no num_experts." G0 holds 14,297,675,776 resident weight bytes after
one load (`GENESIS_RESIDENT_BODY.md` L21;
`QWEN38_SHARED_SESSIONS_SUMMARY.json` `resident_weight_bytes`).
`rg expert_slab crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
is empty.

**GPU idle at 73%: no, on the measured genome.**
`receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`:

- L39 `production_command_buffers: 1`
- L38 `total` dispatches 964
- L1031 `median_gpu_ns: 33912333`
- L1032 `median_wait_ns: 34296583`
- L1035 `median_wall_ns: 35227917`

COMPUTED: `(wait-gpu)/wall = 384250/35227917 = 0.0109` (1.1%).
`(wall-gpu)/wall = 1315584/35227917 = 0.0374` (3.7%).

Complete-token authority
`receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL.json` L4–L8:
complete wall 38.217 ms, GPU 36.987 ms → idle 1.229 ms = **3.2%** of
complete wall. Headline TPS 26.17.

Dominant Qwen3.8 component is **weight_addressing** 21.293 ms = 60.44%
of TOKEN_NS wall (`QWEN38_TOKEN_NS_LEDGER.json` L1534–L1536), not host
IO. `host_preparation` is 0.919 ms = 2.61% (L1515–L1517) — encode bind,
not slab fill.

**Can return** if G1 streams unpacked experts, splits the 1-CB graph
into many CBs with host work between them, or stops residencies.

### Exact check (no GPU)

Static:

```
rg -n 'expert_slab|num_experts|top_k|host.expert' \
  crates/hawking-core/src/model/qwen38_hybrid_decode.rs \
  crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
```

Empty slab name + `production_command_buffers == 1` in the current
ledger ⇒ DSV4F slab/idle shape is absent.

When a GPU lane next emits a Qwen3.8 TOKEN_NS JSON (this lane must not):

```
python3 -c '
import json
d=json.load(open("receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"))
w,g,v=d["median_wall_ns"],d["median_gpu_ns"],d["median_wait_ns"]
print("idle_of_wall",(w-g)/w,"wait_minus_gpu_of_wall",(v-g)/w,"cbs",d["dispatches"]["production_command_buffers"])
'
```

Confirm 73%-class idle iff `(wall-gpu)/wall >= 0.50` and a stage named
`host.expert_slab_io` (or equivalent streamed routed fill) is on the
serial identity sum. Current numbers fail both.

---

## 5. Giant JSON index — manifest + range file were the wall

### Mechanism

Two different JSON pathologies got lumped.

**A. DSV4F artifact open (MEASURED).**
`receipts/dsv4f_artifact_index.json`:

| file | bytes |
|---|---:|
| `manifest.json` | 150,990,326 (144.0 MiB) L22 |
| `stream-ranges.jsonl` | 108,205,334 (103.2 MiB) L23 |
| admission receipt | 13,512,403 (12.9 MiB) L24 |
| mmap index | 26,661,704 (25.4 MiB) |

JSON parse of the manifest is 286 ms (L91). Canonical seal 453 ms (L92).
The *iteration wall* is: appended range journal ⇒ clone 69837-file tree
to hash a sealed prefix (L16). Index admits the durable root; no clone,
no JSON (`gravity_deepseek_v4_artifact_index.rs` L1–8).

**B. `capture-result.json` (CITED + weaker MEASURED).**
Standing law `_COMMON.md` L49: "Giant JSON indexes are a real iteration
wall (1.38 GB capture-result.json)." Introduced as prose in `ac0f5d716`.
`git grep 1.38` finds no receipt field.

COMMIT `33f8a18f3`: "A 155 MB capture-result.json reached history and
GitHub rejected the push at its 100 MB limit." That is a different
number. Writer still exists:
`crates/hawking-core/src/model/dsv4f_activation_capture.rs` L1370, L1577.
`.gitignore` L169 ignores
`workspace/campaign/records/ascension-sandbox/physical/**/capture-result.json`.

**Do not treat 1.38 GB as MEASURED.** Treat "do not serialize the
activation/index working set as one JSON document" as the transferable
rule, evidenced by 151+108 MiB DSV4F metadata and a 155 MB capture body
that could not even be committed.

### Qwen3.8 — can this class exist?

**Small JSON at load, yes. Giant per-token / per-iteration JSON, not on
the G0 path.**

Uniform-Q4 G0: `load_qwen38_manifest` reads `manifest.json` once
(`qwen38_pack.rs` L722–748), pretty-printed catalog of 755 rows. Not a
range journal. Not parsed per token.

Mixed path (not G0, exists for G1): binary `catalog.hq38m20` magic
`HQ38M20\0`, 128-byte records (`qwen38_hybrid_decode.rs` L31–33, L96–173).
No JSON.

Pack-time only: `model.safetensors.index.json` + per-shard header JSON
(`qwen38_pack.rs` L90–95, L211).

**Can return** if G1 writes a doctor/capture `capture-result.json` that
embeds activations, or a pretty-printed per-tensor range map that grows
with every pack iteration.

### Exact check (no GPU)

```
# G0 catalog (path from GENESIS_RESIDENT_BODY.md). File may be absent
# in this sparse worktree — absence is not evidence it is small.
stat -f%z workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json
python3 -c '
import json,pathlib,time
p=pathlib.Path("workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json")
b=p.read_bytes(); t=time.perf_counter(); json.loads(b); dt=time.perf_counter()-t
print("bytes",len(b),"parse_s",dt)
'
```

Rule out giant-index class if `bytes < 8_000_000` and `parse_s < 0.05`
and this parse is not inside `step`. Confirm the class if a new file
named `capture-result.json` / `stream-ranges.jsonl` / pretty `manifest.json`
exceeds ~100 MB or is re-parsed per token / per pack iteration.

Also:

```
rg -n 'capture-result.json|stream-ranges.jsonl|to_vec_pretty' \
  crates/hawking-core/src/model/qwen38_hybrid_decode.rs \
  crates/hawking-core/src/model/qwen38_pack.rs \
  crates/hawking-core/src/model/dsv4f_activation_capture.rs
```

`to_vec_pretty` on the 755-row pack manifest is fine. The same call on
an activation cube is the defect.

---

## 6. Transfer table

| DSV4F / Q30 class | On current Qwen3.8 G0 path? | Same class *can* exist? | Check |
|---|---|---|---|
| Hash per token | **No** | Yes, if a new reader hashes on `step` | `rg Sha256` in `qwen38_hybrid_decode.rs` token path; two-token `hash_invocations` |
| Clone tree on open | **No** (755 `fs::read`s at load only) | Yes, if admit clones a tree or oneshot reloads | `rg clonefile`; `load_ns` phase split |
| `st_dev` identity | **No** on uniform-Q4 load; already dropped in complete-binary | Yes, if a new match key includes `device` | `identity_matches` omits `device`; remount test |
| Expert slab IO | **No** (dense, resident 14.30 GiB) | Yes, if G1 streams routed weights | no `expert_slab` stage; `resident_weight_bytes` stays ~14 GiB |
| 73% GPU idle | **No** (3.2% of complete wall; 1 CB) | Yes, if CB count explodes or host fills between CBs | `(wall-gpu)/wall` on next TOKEN_NS |
| Giant JSON / range journal | **No** at DSV4F scale; 755-row `manifest.json` once | Yes, if capture-result or range JSON becomes the pack/iteration format | `stat` + parse time; refuse >100 MB JSON indexes |

Identity bookkeeping and index bloat **have** been the dominant cost in
DSV4F (hash, clone-tree, 151+108 MiB metadata) and Q30 (`st_dev` false
miss). On **this** Qwen3.8 genome they are not the token. The token is
weight_addressing (60.4% of TOKEN_NS wall). Suspect identity first on
any *new* G1 path; do not assume it is still the 26.4 TPS wall.

Negative already in-repo, transferable: a 2.9× cut of a parallel-sum
identity metric with **zero** token movement (`766670496`). Do not promote
a G1 identity "win" that is not `complete_wall_ns`.

---

## 7. What G1 should steal (science only)

1. Cold proof once → session seal → token path trusts the seal. Never
   SHA / stat / canonicalize / parse geometry per token.
2. Do not clone a tree to admit a prefix. Index the durable root.
3. Do not key identity on `st_dev`.
4. Do not encode the working set as one JSON document.
5. Streamed expert fill is GPU-idle host. Qwen3.8 is not streamed today;
   do not introduce streaming to chase BPW unless complete-token
   measurement still wins.
6. A component microbenchmark or a parallel-sum is not a token claim.

KILLS (do not resurrect as vehicles): DSV4F native graph, DSV4F chunk
tree, DSV4F clone-view admit, Q30 static ≤1.5 coherence
(`_COMMON.md` L46).

REOPEN_IF:

- G1 decode path grows `hash_invocations` on token 1.
- G1 admit grows `prepare_clone_*` or a `capture-result.json` >100 MB.
- G1 identity compare includes `device`.
- G1 TOKEN_NS grows `host.expert_slab_io` or `(wall-gpu)/wall > 0.25`.
- A receipt appears for CITED 13.5→2.3 s / 4606→95 s / 1.38 GB.

---

## Completion report

```
STATUS
SUPPORTED

CLAIMS
1. 78s→3.375s arc exists as campaign token walls. FALSIFIED that all three steps were identity bookkeeping. 78.0→28.98 is a GPU port (MEASURED wall_ms 77993 and 28980). Identity sits on 30.258→3.375 and the earlier hash cuts. Evidence: receipts/dsv4f_streamed_forward_l0_l42_receipt.json L95; receipts/dsv4f_streamed_metal_l0_l42_receipt.json L285; receipts/DSV4F_NATIVE_PLUS_ZEROCOPY_COMBINED.json L106; receipts/DSV4F_NATIVE_INDEX_ADMISSION_COLD_TOKEN.json L123; COMMIT c881b0227.
2. Hash-per-token was identity bookkeeping: ~10 GiB rehashed/token, 25.523s→5.763s then 11.41s→3.31s body, then 0 hashes on the 3.375s cold token. Evidence: COMMIT 9907b0ebe; COMMIT fe13db5a6; receipts/DSV4F_NATIVE_INDEX_ADMISSION_COLD_TOKEN.json L14–L17; receipts/dsv4f_admission_pass.json hash_wall_ms 47205.
3. Clone-tree-on-open was identity bookkeeping: prepare_clone_chunks 11.053 s mean, 88% of init; mmap index 13.413s→0.139s open and 30.258s→3.375s cold token. Evidence: receipts/dsv4f_artifact_index.json L14–L16, L83–L87, L106; gravity_deepseek_v4_streamed_forward.rs L637–L664.
4. st_dev is a Q30/complete-binary identity defect, fixed in identity_matches. 13.5→2.3 s and 4606→95 s (48x) are CITED in dsv-admission-identity.md with no receipt found. Observed remount 16777234→16777233 is in source. Evidence: COMMIT 091fe51cf; admission_warm_receipt.rs L133–L142; qwen_complete_binary/mod.rs L1445–L1450.
5. Expert slab IO is MEASURED 415.13 ms/token GPU-idle, 73.5% of host_exclusive 564.9 ms. GPU idle of the DSV4F token is 61.3%, not 73%. Evidence: DSV4F_HOST_WALL_BASELINE.json L38, L47, L180, L287–L290; TOKEN_NS_DSV4F.json L804–L809.
6. Giant JSON: MEASURED wall is 151 MiB manifest + 108 MiB ranges + clone, not parse (286 ms). 1.38 GB capture-result.json is CITED only; 155 MB is COMMIT 33f8a18f3. Evidence: dsv4f_artifact_index.json L22–L23, L91; _COMMON.md L49.
7. Current Qwen3.8 G0 path does not carry DSV4F-scale hash/clone/st_dev/slab/73%-idle/giant-JSON on the token. Token is weight_addressing 60.4%. Evidence: qwen38_hybrid_decode.rs L508–L536; QWEN38_TOKEN_NS_LEDGER.json L39, L1031–L1035, L1515–L1536; QWEN38_COMPLETE_TOKEN_WALL.json L4–L8; GENESIS_RESIDENT_BODY.md L16–L21.
8. Parallel-sum identity cuts are not token wins. Evidence: COMMIT 766670496.

EVIDENCE
See each claim. Commands used: git show HEAD:<receipt>, git log --format=%B, nl -ba on materialized sources, python3 field extracts. No GPU, no inference.

CHANGES
created workspace/superwave/g1/g1-arch-dsv4f.md
no tracked file modified

TESTS
see final message

RISKS
CITED 13.5/2.3/4606/95/1.38/50s-box-cold are campaign prose. A later receipt could revise them. Sparse checkout: absence of workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json here is not a size measurement.

UNRESOLVED
No in-tree receipt for 13.5→2.3 startup or 4606→95 warm repack. No receipt field gpu_idle_fraction_of_span=0.73. G0 manifest.json byte size not readable in this sparse tree.

NEXT
If G1 adds a reader, run the six static checks in §§1–5 before any Metal timing. A GPU lane owns TOKEN_NS re-measure.
```
