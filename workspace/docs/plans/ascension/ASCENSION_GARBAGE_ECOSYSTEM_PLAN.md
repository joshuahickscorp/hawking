# Ascension Garbage Ecosystem Plan

**Bible:** HAWKING_ASCENSION_BIBLE.md §26  
**Status:** plan + scaffold (gated on Proto-Frankenstein offload)  
**Scaffold:** `workspace/ops/ascension/garbage_ecosystem.py`  
**Tests:** `workspace/ops/ascension/tests/test_garbage_ecosystem.py`

---

## What tonight's code already proves

| Live file | Proven capability | How the scaffold generalizes it |
|-----------|-------------------|---------------------------------|
| `workspace/campaign/records/runs/frankenstein/reclaim_storage_keep_proto.py` | Allow-listed `DELETE_TARGETS` only | `evictable_paths` |
| same | Hard `ALLOWED_ROOTS` confinement — refuse escape | `sandbox_roots` → `sandbox_owned` |
| same | Protected `PROTO_DIR` (Desktop proto-frankenstein) never deleted | `NEVER_AUTO_DELETE_MARKERS` + PINNED class |
| same | Refuse delete unless sealed full-latent V0 receipt endpoint present | `receipt_sealed` + `successor_or_rejection_verified` gates |
| same | Dry-run default; `--apply` explicit | `evaluate_auto_delete(apply=...)` + cleanup receipt |
| same | Report free space after reclaim | `build_cleanup_receipt(free_bytes_before/after)` |

**Do not edit the live reclaim script.** GLM recapture still depends on its disk-floor safety. The scaffold is a new module under `workspace/ops/ascension/`.

---

## Four-state model (bible §26)

```text
PINNED      → never auto-delete; requires human/supervisor promotion out
LEASED      → in active use; reclaim only after lease ends (refs → 0)
EVICTABLE   → candidate for automatic deletion when all gates pass
QUARANTINED → fail-closed default; human inspection only
```

### PINNED (examples)

- stable Hawking tree / production artifacts
- protected tests and receipt authority
- sole rollback
- user worktrees / unknown worktrees
- sealed custom models (incl. Frankenstein / proto-frankenstein)

### LEASED (examples)

- active source window, active candidate, active worktree
- required build cache, current benchmark trace, current checkpoint
- any path with `active_references > 0`

### EVICTABLE (examples)

- rejected candidate, superseded checkpoint
- expired source window, duplicate cache
- retired worktree, unreferenced build products
- **only** when listed in the allow-list **and** under sandbox roots  
  (tonight's pattern: `glm-donor`, `workspace/ops/local/hf-cache`)

### QUARANTINED (examples)

- partial atomic output, failed verification
- unknown ownership, corrupt receipt
- unclassified path (default)

---

## Automatic deletion gates (all required)

```text
sandbox-owned
∧ EVICTABLE
∧ no active references
∧ receipt sealed
∧ successor or rejection verified
∧ rollback preserved
∧ remote hash verified when required
∧ not in never-auto-delete set
```

## Never auto-delete

```text
Frankenstein / proto-frankenstein
stable Hawking
protected authorities
user files
unknown worktrees
sole rollback
unclassified directories  (→ QUARANTINED, not EVICTABLE)
```

## After cleanup

1. Purge cache/trash (future supervisor hook — not in scaffold)
2. Prove free-space recovery (`free_bytes_before` → `free_bytes_after`)
3. Emit sealed cleanup receipt (`hawking.ascension.cleanup_receipt.v1`)

---

## Classification algorithm

1. Hard QUARANTINE: partial/corrupt, unknown ownership, explicit quarantine list  
2. PINNED: explicit pin list, never-auto-delete markers, user/unknown worktree markers  
3. LEASED: explicit lease list or `active_references > 0`  
4. EVICTABLE: explicit allow-list **under** `sandbox_roots` (else QUARANTINED)  
5. Default: QUARANTINED (unclassified)

---

## Integration plan (post Proto-Frankenstein offload)

1. **Inventory pass** — walk sandbox roots, emit `ObjectRecord` JSONL  
2. **Supervisor wire** — campaign orchestrator calls `evaluate_auto_delete` before any delete  
3. **Retire live reclaim allow-list into config** — migrate `DELETE_TARGETS` / `ALLOWED_ROOTS` into a sealed policy file consumed by this module  
4. **Never call delete from sandbox models** — only supervisor + receipt gates  
5. **Cleanup receipt** lands under `workspace/campaign/governance/control/receipts/`

---

## Non-goals (this scaffold)

- No filesystem deletion  
- No daemon  
- No edit to `reclaim_storage_keep_proto.py`  
- No touch of `lab/operators/frankenstein_*` or frankenstein evidence  
- No Qwen / Gravity downloads  

---

## Remaining work

- [ ] Policy config schema (JSON) for roots / allow-lists / pins  
- [ ] Inventory walker with size accounting (`du` family)  
- [ ] Wire pressure governor RED/CRITICAL → `evict_leased_cache` using LEASED→EVICTABLE transitions  
- [ ] Remote hash verifier hook (content-addressed offload)  
- [ ] Promotion path: EVICTABLE → deleted only after human or sealed supervisor receipt  
