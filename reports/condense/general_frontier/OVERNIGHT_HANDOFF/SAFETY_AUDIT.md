# Overnight Supervisor - Adversarial Safety Audit

Target: `tools/condense/overnight_supervisor.py` (read-only audit; module unmodified).
Dependency read: `tools/condense/source_release_readiness.py`, `tools/condense/doctor_campaign_supervisor.py`.
Date: 2026-07-19. Auditor scope: destructive + consequential paths only.

## TOP-LINE VERDICT

The autonomous 120B source deletion is SAFE to run overnight.

The `os.remove` call is provably confined to the 7 exact shard absolute paths, it is
fail-closed at every guard, and it is currently double-blocked from firing at all:

1. The campaign is not final (`final=False`, 12/28 rows), so the machine is parked in
   `WAIT_120B_FINAL` and cannot reach the deletion state tonight.
2. Even at the gate, the source-release readiness verdict is DENIED (9/15 green; 6 red,
   including `10_no_running_process_maps_source` which is red precisely because the live
   Doctor campaign is a detected consumer of the source).

Deletion can only occur after the campaign concludes final=true, `verify_120b()` passes,
and all 15 release gates independently flip green. None of those hold now.

Findings below are ranked. No finding lets the deleter touch anything other than the 7
shards. Two medium findings concern (a) a stale-verdict fallback if the re-verification
subprocess crashes, and (b) a latent git commit-target risk. Neither is reachable tonight.

Grounded live state at audit time:
- `DOCTOR_CAMPAIGN_STATE.json`: final=False, rows_done=12, rows_total=28.
- `OVERNIGHT_HANDOFF/state.json`: state = WAIT_120B_FINAL.
- `GPT_OSS_120B_SOURCE_RELEASE_READINESS.json`: DENIED, green=9, red gates =
  05,06,07,08,10,11.
- HEAD = campaign/adaptive-transfer-ladder (not main); that branch exists locally.
- All 7 shards present with exact names; index.json retained.

---

## Deletion confinement (`h_release_120b_source`, lines 313-362)

PASS - Only `os.remove` in the whole module is line 347, inside `for p in SHARDS`, where
`SHARDS` is the module constant list of the 7 literal `ORIGINAL/model--0000{1..7}-of-00007.safetensors`
paths (line 49). No `glob`, no `shutil.rmtree`, no `os.rmdir`, no parent-dir removal
anywhere. `shutil` is imported but used only for `disk_usage` (line 144). The deleter
cannot remove a directory, metadata, or any path outside that fixed list.

PASS - Per-path guard (lines 328-335) checks, for every `p` in SHARDS:
`p.resolve().parent == ORIGINAL.resolve()` (rejects any symlink whose target escapes the
original dir), name prefix `model--0000`, name suffix `-of-00007.safetensors`, `p.exists()`,
and lsof-not-mapped. Any failure calls `_fail` -> BLOCKED and returns before any removal.

PASS - TOCTOU is bounded: even if a shard path were swapped for a symlink between the
check and `os.remove(str(p))`, `os.remove` deletes the link entry at the exact
`ORIGINAL/model--0000N` path, not the symlink target, so no external file can be removed.
The name/prefix/suffix checks are effectively tautological (SHARDS names are hardcoded), so
the real guard is the `resolve().parent == ORIGINAL.resolve()` equality plus exists.

PASS - Live-controller gate (line 337): reads
`CAMP/leases/doctor_campaign.lease` pid and refuses if alive. This path matches the
lease the Doctor supervisor writes (`doctor_campaign_supervisor.py` LEASE). If the lease
file is missing/corrupt, `.get("pid", -1)` yields -1 and `_pid_alive(-1)` returns True
(`os.kill(-1, 0)` succeeds), so a missing lease is treated as "still live" -> deletion
REFUSED. Fail-closed and favorable.

PASS - Metadata precondition (lines 340-342): refuses to delete unless config.json,
model.safetensors.index.json, and tokenizer.json all exist first. Post-delete it re-checks
those survived (lines 354-356). These files are never in the deletion set.

PASS - Exists precondition per shard (line 332) plus post-delete absence verification
(lines 351-353).

RISK (medium) - Stale-verdict fallback in the pre-deletion re-check (lines 316-324).
`h_release_120b_source` re-runs `source_release_readiness.py` but discards the subprocess
result, then reads `GPT_OSS_120B_SOURCE_RELEASE_READINESS.json` from disk. If that re-run
raises (for example the receipt file is moved, disk_usage fails, any transient), the script
exits nonzero and does NOT overwrite the OUT file (in `main()` the load happens before the
write), so the supervisor reads the STALE verdict left by the EVALUATE step. To have reached
RELEASE, EVALUATE already saw all 15 green, so the stale verdict would be all-green and the
delete proceeds on stale data. Impact is bounded (the stale data was legitimately all-green
seconds earlier), but the "re-verify immediately before deleting" guarantee is not enforced:
neither the subprocess returncode nor the file freshness (mtime) is checked. Recommend
asserting `r.returncode == 0` and a fresh `generated_at_utc`/mtime before trusting the file.

RISK (low) - "15/15" is not enforced as a count; the code accepts all-of-N green
(`green == len(g)`, lines 323/297). The readiness script currently emits exactly 15, so this
is equivalent today, but if that script were changed to emit fewer gates, a smaller all-green
set would authorize deletion. Also the supervisor re-derives the verdict from the `gates`
dict rather than trusting the authoritative `release_authorized` / `release_decision`
("AUTHORIZED") field the script already computes; the two are equivalent now but could drift.

RISK (low) - Partial-deletion wedge. If `os.remove` raises mid-loop (for example EPERM on
shard 4), it is caught by `tick()`'s top-level try/except and the state does not advance, but
the `release_source` claim is already burned. On the next tick `_claim("release_source")`
returns False and `h_release_120b_source` no-ops, leaving some shards deleted, the machine
stuck in RELEASE_120B_SOURCE, and no completion. This is fail-closed for over-deletion (it
never deletes more than intended) but leaves an inconsistent, non-advancing state.

## Gate re-check immediately before deletion

PASS (with the medium caveat above) - 15-gate readiness is re-run and re-required green at
RELEASE (lines 316-324), not only at EVALUATE (lines 289-298). The two checks read the same
OUT file via the same parse. `verify_120b()` (the separate 28-row seal/byte-ledger/D3-D5
check) is NOT re-run at RELEASE, but its concerns are covered by readiness gates
05/06/07/08/14 and by the metadata preconditions, so this is acceptable. The only gap is the
crash-fallback-to-stale described above.

## One-use / restart-safe / claim ordering

PASS - Every side-effectful transition acquires its O_EXCL claim FIRST, before the effect:
`release_source` (line 314) before any delete; `commit_conclusion` (line 281) before git;
`seal_conclusion` (line 247) before the seal subprocess; `admit_qwen`, `run_q0q1q2`,
`launch_qwen` likewise. `_claim` uses `os.open(..., O_CREAT | O_EXCL | O_WRONLY)` on a file
under `OVERNIGHT_HANDOFF/claims`, which persists across restarts, so a completed or
attempted transition is a durable no-op on replay. No transition performs its effect before
winning its claim.

RISK (low) - Telegram loop-spam in `h_transfer_qwen_priority`. The "transfer paused" notice
(line 400) and the "shard transfer retry pending" notice (line 410) have no dedup and no
backoff timer, so while disk sits between hard-stop and pause, or while a download keeps
failing, a Telegram fires every launchd tick. The comment says "bounded backoff: retry next
tick" but there is no actual rate limit. Other stall/idle states (WAIT, MONITOR, BLOCKED) are
correctly deduped on a stored marker; only the transfer path is not. Not a safety issue;
noise only, and not reachable tonight.

## Acts only when valid (never on PID exit alone)

PASS - `h_wait_120b_final` advances only when `CAMP_STATE.final` is truthy (line 218). There
is no PID/liveness trigger anywhere in the WAIT-to-VERIFY path. Progress telegrams are gated
on `rows_done` change (line 220), so they do not spam. `verify_120b()` additionally requires
final=true (line 163) plus 28 valid sealed rows, byte ledgers, the D3/D5 non-admission
receipt, and source index+tokenizer identity. The machine cannot act on a Doctor PID exit;
it acts only on final campaign state plus full verification.

## Qwen transfer

PASS - Disk floors: hard-stop `< 40 GB` checked at the top of each transfer tick (line 390),
pause `< 100 GB` checked before each shard download (line 399); one shard per tick keeps each
tick short and disk-checked. Deletion of the 120B source is never used to make room for Qwen;
if gates are not green the machine explicitly runs `shard_serial_no_release` mode (line 302)
and keeps the source.

PASS - Secrets are not logged. `_telegram` obtains the token via `doctor_campaign_supervisor._creds`
(Keychain first, 0600 file fallback) and passes it only to the notifier; on failure it logs
only `type(exc).__name__` (lines 100-101). The download subprocess inherits `os.environ` but
nothing prints it.

RISK (low, version-dependent) - "One physical copy" relies on huggingface_hub's default
`local_dir` behavior. The download call (lines 402-408) passes `local_dir=QWEN_DIR` but does
not set `local_dir_use_symlinks=False` or a dedicated `cache_dir`. On modern hub versions
`local_dir` downloads directly with no cache blob; on older versions large files land in the
HF cache and are symlinked into local_dir (still one physical blob, but not under QWEN_DIR).
Not a duplicate-copy or safety hazard, but the "no HF cache dup" guarantee is not explicitly
forced in code. Also `HF_HUB_ENABLE_HF_TRANSFER=1` requires the hf_transfer package; if
absent the download fails and triggers the retry-telegram spam noted above.

## Git (`h_evaluate_source_release`, lines 279-302)

PASS - No merge-to-main exists. Despite the comment "Merge to main only if clean" (line 280),
there is no merge or checkout of main anywhere. The only remote writes are
`push origin campaign/adaptive-transfer-ladder` and a force-create + force-push of the tag
`hawking-gptoss-120b-frontier`. No unguarded merge-to-main to flag.

RISK (medium, latent) - The commit targets HEAD, not the named branch. The block runs
`git add -A` then `git commit` with no branch argument, so it commits to whatever branch is
checked out; only the subsequent `push` names the campaign branch. Right now HEAD is
campaign/adaptive-transfer-ladder, so it behaves as intended and is fine. But if HEAD were
main when this fires, `git commit` would land a commit on local main (and the tag would be
force-created at that main commit and force-pushed), while the branch push would push the
stale campaign ref. The handler does not verify the current branch before committing, so the
"branch only" claim in the comment is not enforced by code. Reachable only after the campaign
concludes and verify passes.

RISK (low) - `git add -A` is indiscriminate: it stages every untracked/modified file in the
working tree, which could sweep in unrelated artifacts or a repo-local secrets file into the
commit. The Telegram creds live in a 0600 file under the handoff dir; if that path is inside
the repo it would be staged. Prefer an explicit pathspec (reports/tools) over `-A`.

## Other observations (non-destructive)

NOTE - `h_seal` line 253 `is_a = "OUTCOME_A" in outcome or "PASS" in outcome and "BOUNDARY"
not in outcome` reads (by precedence) as `A or (PASS and not BOUNDARY)`. This only selects
whether the bounded no-op `NARROW_RATE_REFINEMENT` hook runs; it has no destructive effect.

NOTE - `_pid_alive` returns True on `PermissionError` (line 151) and for pid -1, both of which
bias the live-controller gate toward refusing deletion. Favorable.

## Summary table

| Area | Verdict |
| --- | --- |
| os.remove confined to 7 exact shard paths (no glob/rmtree/parent) | PASS |
| Per-path parent/name/exists/lsof guards | PASS |
| Live Doctor-controller gate (fail-closed on missing lease) | PASS |
| Metadata-present precondition + post-delete retention check | PASS |
| 15-gate re-check immediately before deletion | PASS with medium caveat (stale-on-crash) |
| Acts only on final=true + verify pass, never on PID exit | PASS |
| O_EXCL one-use, restart-safe, claim-before-effect | PASS |
| Qwen disk floors + secrets not logged | PASS |
| No merge-to-main | PASS |
| Stale readiness verdict if re-check subprocess crashes | RISK (medium) |
| git commit targets HEAD not the named branch; git add -A | RISK (medium latent / low) |
| Telegram loop-spam on transfer pause/retry | RISK (low) |
| Partial-deletion wedge on mid-loop os.remove failure | RISK (low) |
| "15/15" is all-of-N, not a fixed count; re-derives vs authoritative field | RISK (low) |
| Qwen one-physical-copy relies on hub default (not forced) | RISK (low) |

Bottom line: deletion cannot escape the 7 shards, is fail-closed, and is inert tonight. The
medium findings are latent (only reachable after the campaign concludes and all gates go
green) and are hardening items, not open doors to over-deletion.
