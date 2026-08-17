# G1 prefill amortization

STATUS: IMPLEMENT_READY

Seal does not require a fresh session. `generate_greedy` resets every propose.
Prefill the 1407-token sealed system prefix once per session; append the turn.

## 1. Facts

### 1.1 Where the session is reset, and why

`Body.propose` encodes the full wire prompt and calls `generate_greedy` (`tools/agentos/genesis_body/src/main.rs` 622-696).
`generate_greedy` unconditionally `session.reset()` then `session.step`s every prompt token (`crates/hawking-core/src/model/qwen38_hybrid_decode.rs` 3644-3679).

```
3644:    pub fn generate_greedy(...) {
3652:        session.reset();
3664:        for (i, &token) in prompt.iter().enumerate() {
3666:            let (sampled, timing) = session.step(token)?;
```

`reset` zeros position, DeltaNet conv/rec, and GQA K/V (1451-1457).

Why: generate is written as a one-shot. Not a seal rule. Client already sends the full sealed prompt every turn (`tools/agentos/genesis_resident.py` `propose` 281-323 via `inject_runtime_contract` 219-263).

`start_new_session=True` hits in this tree are `subprocess.Popen` process groups (`lab/hcli/special_unit.py` 653), not model KV.

### 1.2 Persistent KV per session

Yes, as allocated workspaces. No, as retained prefix.

`Body::load` attaches four isolated `Qwen38HybridDecodeSession`s against one `Arc<Qwen38HybridWeights>` (`genesis_body` 474-479). Roles: `parent`, `child_a`, `child_b`, `protected_test` (line 30). Each holds its own conv/rec/GQA.

Live AgentOS: gravity=`child_a`, kernel=`child_b`. `session_serve_index` increments (1..5 and 2..6) while `ephemeral_model_state.kv_preserved` is hardcoded `False` (`lab/lineage/continuity.py` 114-116). Objects persist. KV is wiped every generate.

`crates/hawking-core/src/stateful/prefix_cache.rs` is a generic `KvCache` RAM tier for Qwen-dense / disk prefill. Not wired to `Qwen38HybridWorkspace`. KILL as the Qwen3.8 vehicle.

### 1.3 Tokens, not bytes

Live binding (Python `contract_provenance()`, parent files on disk):

`3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f`

(Task wrap `...d459 bf7bc...` is a line-break typo. Measured `...d459bd7fc...`.)

| object | bytes | tokens MEASURED | how |
|---|---:|---:|---|
| QWEN38_GENESIS_SYSTEM_DIRECTIVE.md | 16414 | 4187 | HF BPE `tokenizer.json`, `add_special_tokens=False` |
| GENESIS_CONTINUITY_DIRECTIVE.md | 11912 | 3013 | same |
| GENESIS_OUTPUT_LAW.md | 4871 | 1264 | same |
| three files concat | 33197 | 8464 | same; 4187+3013+1264 |
| `runtime_capsule(child_a)` | 5119 | 1402 | same |
| sealed system prefix `<\|im_start\|>system\n{capsule}<\|im_end\|>\n` | 5149 | **1407** | same; this is what is prefills |

Resident encode is `tokenizer.encode(&rendered, false)` (`genesis_body` 653). Same flag as the counts above.

8464 file tokens > live `max_seq_len` 8192 (`tools/genesis_forever.sh` 70). The files cannot be the prefilled object. The runtime object is the hash capsule (`genesis_contract.py` 1-6, 184-216). Test bound: capsule < 7000 bytes (`test_genesis_contract.py` 139-144).

Reconstructed full wires tokenize to the durable `prompt_len` with delta 0 on every accounted turn (CPU `tokenizers` on `/tmp/g1-paired-*.txt`). Contract prefix is the first 1407 tokens of every such wire.

Successive-turn longest common prefix is 1916-1929 tokens (tools header + stable worker boilerplate). Not the seal. Named as optional extra.

### 1.4 Prefill vs decode (durable receipts)

Source: `/Users/scammermike/Downloads/hawking/workspace/ops/genesis-workers.json` (live, `hawking.genesis.worker_registry.v1`). Accounting added mid-run: kernel walls 157.117s / 206.949s have no `prefill_wall_ns`.

| worker | serve | prompt_len | prefill_s | decode_s | wall_s | prefill/wall | json lines |
|---|---:|---:|---:|---:|---:|---:|---|
| gravity/child_a | 8/1 | 3209 | 200.963 | 3.237 | 204.199 | 0.984 | 137,180-188 |
| gravity/child_a | 11/2 | 3413 | 218.616 | 2.800 | 221.417 | 0.987 | 241,278-286 |
| gravity/child_a | 13/3 | 3446 | 219.999 | 2.791 | 222.790 | 0.987 | 328,365-373 |
| gravity/child_a | 15/4 | 3452 | 221.109 | 2.814 | 223.923 | 0.987 | 415,452-460 |
| gravity/child_a | 17/5 | 3461 | 234.963 | 2.814 | 237.777 | 0.988 | 502,539-547 |
| kernel/child_b | 5/2 | 3222 | 205.143 | 3.099 | 208.242 | 0.985 | 866,907-915 |
| kernel/child_b | 7/3 | 3468 | 222.971 | 3.274 | 226.244 | 0.986 | 961,1003-1011 |
| kernel/child_b | 10/4 | 3529 | 228.521 | 3.488 | 232.010 | 0.985 | 1057,1101-1109 |
| kernel/child_b | 12/5 | 3534 | 229.460 | 4.410 | 233.870 | 0.981 | 1157,1211-1219 |
| kernel/child_b | 14/6 | 3527 | 227.924 | 4.390 | 232.314 | 0.981 | 1267,1321-1329 |
| kernel/child_b | 14/6* | 3515 | 227.192 | 4.386 | 231.578 | 0.981 | 1377,1431-1439 |

\* last kernel row `session_serve_index` 6 in the earlier parse; file grew under a live kernel tick.

n=11 accounted:

- wall_s MEASURED min 204.199 / median 226.244 / mean 224.942 / max 237.777
- prefill/wall MEASURED median 0.985
- prefill ms/token MEASURED median 64.17 (range 62.62-67.89)
- decode ms/token after first new MEASURED median 90.86 (n_new-1; first new is the last prefill `step`)
- actions/hour MEASURED 3600/226.244 = **15.91** (one serialized body)

Operator "207s / 17 actions/h" matches the early accounted pair 204.199 and 208.242 (3600/207 = 17.39). Later prompts grew ~200-300 tokens and walls grew with them.

`special_unit.py` comment `~38.5 ms/token` (HEAD 83-86) is a short-prompt figure. `bound_generation.complete_token_ns` 37_879_375 is the protected complete-token, not these HCLI turns at pos 3200+.

Host round-trips: confirmed. Prefill is one `session.step` per prompt token (3664-3677), not a batched prefill. `reset_ns` at seq 11 is 0.8-12.5 ms (`QWEN38_COMPLETE_TOKEN_WALL.json`). Unmeasured at 8192 (1.00 GiB GQA blit). Still << 90 s. REOPEN_IF 8192 reset_ns is large: add `reset_ns` to `generate_greedy` the way `generate_greedy_complete_wall` already does (3728-3730).

### 1.5 Live envelope

- `max_seq_len` 8192: `genesis_forever.sh` 70; health RPC timed out (resident busy on a propose; PID 50196 holds the socket).
- Weights: 14_297_675_776 B (`genesis-resident.log` "weight_bytes=14297675776").
- GQA KV formula `16 * seq * 4 * 256 * 4 * 2`: 1.000 GiB/session at 8192. DeltaNet conv+rec: 149.62 MiB/session (seq-independent). Four sessions ≈ 4.58 GiB KV+DN + activations. Task 19.3 GiB RSS = weights + that + Metal. Not re-ps'd (sandbox blocked `ps`).
- Decode concurrency 1 by design (`genesis_body` header 1-7; health `decode_concurrency: 1`).
- Artifact sha live CURRENT: `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df` (resident log + worker `bound_generation`).

### 1.6 Seal vs reset

Reset is how the code happens to work.

Seal requires:

- Correct role-bound capsule as the sole system turn (`inject_runtime_contract` 227-263; rejects stale/wrong-role/forged/duplicate).
- File hashes + binding echoed and matched (`propose` 294, 322-335; `Body::verify_contract_provenance` 202+; `validate_resident_reply` 1679-1745).
- Isolated per-worker KV *slots* (`QWEN38_GENESIS_SYSTEM_DIRECTIVE.md` §15: "They have isolated task/context/KV state").
- That work survive KV loss (`GENESIS_CONTINUITY_DIRECTIVE.md` §4: task state is AgentOS, "Not Qwen3.8's KV cache"; "Do not require preservation of stale KV to preserve the task").

Seal does not require dropping KV every turn. Continuity forbids *depending* on KV across a generation change. Retaining a prefix inside one generation, keyed by binding, is allowed. Drop it on reload, artifact change, binding change, or token mismatch.

`compile_worker_context` already has `ephemeral_kv_reused: False` (continuity 240) ready to flip on a proven hit.

## 2. Design

Keep the wire. Keep the seal. Stop recomputing the prefix.

### 2.1 Per-session prefix KV

On first hit-miss for `(session_role, binding_sha256)`:

1. Rehash the three contract files (reuse `verify_contract_provenance`). Mismatch → refuse, no generate.
2. `reset()`, `step` the full prompt as today.
3. After token `n = sealed_system_prefix_len` (first system turn; 1407 today), blit DeltaNet `conv_state`+`rec_state` into a per-session snapshot. Record `prefix_token_ids` (or their sha256) + `n` + binding + role.
4. Finish suffix prefill + decode.

On later propose, same session:

1. Rehash files. Fail closed.
2. Encode incoming prompt. Require `prompt_ids[..n] == snapshot.ids` (exact tokens, not binding alone) and `req.genesis_system_contract.binding_sha256 == snapshot.binding` and `session == snapshot.role`.
3. Hit: restore conv/rec, set `position = n`, do **not** `reset()`, `step` only `prompt_ids[n..]`. GQA slots `>= n` are overwritten by suffix steps. Slots `< n` stay.
4. Miss: drop snapshot, full reset+prefill, new snapshot if the new prompt still has a legal system prefix.

Do not accumulate conversation KV. Each turn is prefix + this-turn user. Matches today's protocol (`act` sends a new user chat every tick; `inject_runtime_contract` rebuilds system+user).

### 2.2 Key = binding hash + role + token sha

Binding hash invalidates a contract-file change.
Role is inside the capsule (`SESSION_ROLE: child_a` vs `child_b`); prefixes are not interchangeable.
Token sha catches a capsule-compiler change that leaves file hashes alone.

Body derives `n` structurally: first `<|im_start|>system ... <|im_end|>\n` span. Decode that span and require `CAPSULE_BEGIN`, `CONTRACT_SET_SHA256 == binding`, `SESSION_ROLE == req.session`. No Rust port of `runtime_capsule` required.

### 2.3 Eviction / 8192 / overflow

| event | action |
|---|---|
| `Body::reload` / artifact sha change | sessions rebuilt (510-555). Snapshots die. |
| binding or token mismatch | drop snapshot, full prefill. |
| `prompt_len + max_new > max_seq_len` | existing refuse (`genesis_body` 663-672). Do not evict prefix to squeeze a turn. |
| generation rebind | continuity already checkpoints durable state and recompiles. KV is ephemeral. |
| process death | no snapshot on disk (`WorkerCheckpointStore`: "No KV or transcript is stored", continuity 157). First turn after restart is cold. Correct. |

Snapshot RAM: one conv+rec copy per live worker ≈ 149.62 MiB each. Two workers ≈ 300 MiB. Do not snapshot GQA (1.00 GiB/session). Peak stays under the live ~19.3 GiB envelope; this lane must not start a second body.

Protected capability (`genesis_contract_mode=protected_capability_prompt_preserved`) stays on `protected_test` and must miss the worker prefix cache (wrong role; different prompt). Leave `generate_greedy_complete_wall` resetting (3728-3729) so protected benches stay one-shot.

### 2.4 Per-turn proof the retained prefix is the sealed contract

Reply fields (add next to `prefill_wall_ns` 728-729):

- `prefix_hit`: bool
- `prefix_tokens`: n
- `prefix_binding_sha256`
- `prefix_tokens_sha256`
- `prefix_files_rehashed`: true
- `genesis_system_contract` (already)

Client (`genesis_resident.propose` 329-335 + `validate_resident_reply` 1703-1745): refuse if `prefix_hit` and (`prefix_binding_sha256 != expected` or files rehash not true). A hit without proof is a rejected generate, not a fallback.

Never send only the user delta. The wire still carries the capsule. Speed is skipped compute, not dropped authority.

### 2.5 Optional second cut (not required)

LCP across successive turns is ~1924 tokens. Caching system+tools+stable boilerplate saves another ~33 s. Do this only after contract-prefix hits are proven. Tools list and worker boilerplate are not the seal.

## 3. Payoff

Per-token rate from the same turn: `proj = decode_s + (prompt_len - 1407) * (prefill_s / prompt_len)`. First turn after attach/reload stays cold.

| turn | wall MEASURED | wall PROJECTED | save PROJECTED |
|---|---:|---:|---:|
| g 3209 | 204.199 | 116.087 | 88.113 |
| g 3413 | 221.417 | 131.292 | 90.124 |
| g 3446 | 222.790 | 132.965 | 89.826 |
| g 3452 | 223.923 | 133.801 | 90.122 |
| g 3461 | 237.777 | 142.23 | 95.55 |
| k 3222 | 208.242 | 118.659 | 89.583 |
| k 3468 | 226.244 | 135.783 | 90.461 |
| k 3529 | 232.010 | 140.899 | 91.111 |
| k 3534 | 233.870 | 142.514 | 91.355 |
| k 3527 | 232.314 | 141.390 | 90.924 |
| k 3515 | 231.578 | 140.637 | 90.941 |

PROJECTED steady-state (n=11, skip 1407):

- turn wall median **~135.8 s** (range 116.1-142.5)
- organism actions/hour **~26.5** (3600/135.8), one serialized body
- vs MEASURED 15.91 /h at 226.244 s median
- vs operator 17.39 /h at 207 s

LCP-1924 PROJECTED median ~102 s, ~35 /h. Labelled extra. Not the primary claim.

First turn after resident start/reload remains ~205-238 s.

This is not a component microbenchmark. Rates come from the same durable `prefill_wall_ns` / `prompt_len` as the 207 s claim.

## 4. Cheapest confirmation

Do not start a GPU bench. After the patch, read the next two accounted AgentOS turns on one `session_role` in `genesis-workers.json`.

ACCEPT_IF: `prefix_hit=true` on turn 2; `prefix_tokens=1407`; `prefix_binding_sha256` matches live binding; `prefill_wall_ns_2 / prefill_wall_ns_1` in `[0.55, 0.70]` (1407/3200≈0.44 skipped ⇒ remaining ≈0.56; allow prompt_len drift); `fallbacks=0`.

REJECT_IF: `prefix_hit=true` and files rehash skipped; greedy ids of a fixed probe differ from a cold-reset control (one dedicated propose pair, only when the sibling GPU lane is free); `prefix_tokens != 1407` while the capsule is unchanged.

CPU-only pre-check (this lane already did): reconstructed wire tokens == `prompt_len` (delta 0); prefix tokens == 1407 and match every wire.

## 5. Ordered change list

Do not edit two-writer files. Line numbers are HEAD of this worktree unless marked LIVE.

1. `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
   - `Qwen38HybridDecodeSession` 1190-1210: add optional prefix snapshot handles (conv/rec copies + token ids + binding + role + n).
   - `reset` 1451-1457: keep as the miss path. Do not call it on hit.
   - `zero_buffer` 729: add a sibling blit/copy for conv/rec.
   - New `snapshot_prefix(n, ids, binding, role)`, `restore_prefix()`, `prefix_matches(prompt, binding, role)`.
   - `generate_greedy` 3644-3717: accept prefix cursor; on hit start the prefill loop at `n`; record `prefix_hit` / `prefix_tokens_reused` on `Qwen38GenerateResult` 4009-4025.
   - Leave `generate_greedy_complete_wall` 3719-3799 resetting (protected one-shot).
   - Unit test: cold vs restore greedy ids identical on a toy prefix (no GPU if mocked; else Metal-only cfg like the rest).

2. `tools/agentos/genesis_body/src/main.rs`
   - `Body` 435-452: no extra session map required if snapshot lives on the session.
   - `load` 474-479 / `reload` 510-555: already rebuilds sessions; snapshots die for free.
   - `propose` 622-742:
     - After encode 653, identify system-prefix span; verify capsule sentinels + binding + role.
     - Rehash the three files (lift the loop in `verify_contract_provenance` 202-293 into a callable).
     - Hit → `generate_greedy` from n. Miss → current path + snapshot after n.
     - Keep overflow refuse 663-672 on the **full** prompt_len.
     - Echo prefix proof fields next to 728-741.
   - `protected_capability` / wrong role: force miss.

3. `tools/agentos/genesis_resident.py`
   - `propose` 281-343: pass through prefix_* ; refuse `prefix_hit` without matching binding (mirror 333-335).
   - `health` stub 438: optional `prefix_sessions` map. Not required for the win.

4. `tools/agentos/genesis_contract.py`
   - Export a `system_prefix(role) -> str` (already inline at 230) so tests share one compiler. No capsule text change.

5. Two-writer — Codex applies, this lane does not touch:
   - LIVE `lab/hcli/special_unit.py` 1958-1968 and 2056-2078: add `prefix_hit`, `prefix_tokens`, `prefix_binding_sha256`, `prefix_tokens_sha256`, `prefix_files_rehashed` next to the prefill persist Codex already landed. HEAD 1901-1918 does not yet persist prefill_*; do not revert that live add.
   - `validate_resident_reply` HEAD 1703-1745: if `prefix_hit`, require proof fields and binding match.
   - `lab/lineage/continuity.py` 114-116 and 240: set `kv_preserved` / `ephemeral_kv_reused` from the last native `prefix_hit`. Stay False on miss/reload.
   - `tools/genesis_agentos.py` `_worker_prompt` HEAD 266-314 / LIVE 276+: no change for contract-prefix. Do not grow the excerpt.
   - `tools/agentos/test_genesis_contract.py` / `test_genesis_resident.py`: stub replies must keep `fallbacks=0` + binding; add prefix_hit=false default.
   - `lab/tests/test_special_unit.py`: `GenesisResidentBackend` stubs (LIVE 937+) accept the new receipt keys.
   - Do not edit `lab/lineage/lifecycle.py` (absent on HEAD), `lab/lineage/state.py`, `tools/ascent_daemon.py`.

6. Do not:
   - Wire `stateful/prefix_cache.rs`.
   - Prefill the 8464-token raw files.
   - Send user-delta-only wires.
   - Touch G0 / live resident / GPU.

## 6. KILLS

- **Raw 33 kB files are the prefill.** 8464 tokens > 8192. Capsule is the prefill (1407).
- **`stateful::prefix_cache` as the Qwen3.8 store.** Wrong KV type; not on the generate path.
- **Accumulate chat KV.** 3500+256 per turn fills 8192 in two turns.
- **Skip the capsule on the wire.** Seal is the capsule + file hashes, not a session cookie.
- **Treat 38.5 ms/tok as this turn.** Measured 64.17 ms prefill / 90.86 ms decode at these positions.

## 7. REOPEN_IF

- Batched Qwen3.8 prefill replaces per-token `step` (seconds saved drop; skip still correct).
- Capsule compiler embeds full files or `prompt_len` jumps.
- `generate_greedy` reset removed independently (do not double-restore).
- 8192 `reset_ns` is a material fraction of the 90 s save (measure it).
- Binding or tokenizer changes (prefix key must miss).

## 8. Evidence (commands / excerpts)

Binding + capsule bytes (CPU, parent `genesis_contract`):

```
binding 3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f
capsule[child_a] bytes=5119
system_prefix_child_a bytes 5149
three_files_concat_bytes 33197
```

Tokens (`~/.grok-vision/bin/python` + `tokenizers.Tokenizer.from_file(tokenizer.json)`, `add_special_tokens=False`):

```
capsule_child_a 1402
system_prefix_child_a 1407
three_files_concat 8464
file .../QWEN38_GENESIS_SYSTEM_DIRECTIVE.md tokens=4187
file .../GENESIS_CONTINUITY_DIRECTIVE.md tokens=3013
file .../GENESIS_OUTPUT_LAW.md tokens=1264
```

Wire reconstruction == durable `prompt_len` (delta 0), prefix match 1407 on every accounted turn. LCP successive measured turns 1916-1929.

Live receipt excerpt (`genesis-workers.json`):

```
137: "decode_wall_ns": 3236761958,
180: "prefill_wall_ns": 200962729833,
181: "prompt_len": 3209,
188: "wall_ns": 204199496458
```

`generate_greedy` reset: `qwen38_hybrid_decode.rs` 3652.
Four attaches: `genesis_body` 474-479.
Client inject every propose: `genesis_resident.py` 303-323.
Continuity KV ephemeral: directive §4 lines 145, 174; `kv_preserved: False` at continuity.py 115.
Isolation required: system directive §15 "They have isolated task/context/KV state."
Launch window: `genesis_forever.sh` 70 `--max-seq-len 8192`.
Health to live sock: `TimeoutError` (body busy; PID 50196). No propose issued.

LIVE vs HEAD: Codex already persists `prefill_wall_ns`/`decode_wall_ns`/`prompt_len` in parent `special_unit.py` 1963-1966. HEAD 1901-1918 does not. Two-writer: add prefix_* beside the live keys; do not collide.

No GPU run. No resident restart. No tracked file edited. No two-writer edit.
