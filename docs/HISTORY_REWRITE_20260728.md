# History rewrite, 2026-07-28

The repository history was rewritten on 2026-07-28. Every commit hash before that date
changed. This file records what happened and how to translate an old hash.

## What changed

**Blobs removed from all history** — ten data artifacts, 64 MB, none present in the working
tree at the time of the rewrite:

    HAWKING_BYTE_CATEGORY_LEDGER.json      HAWKING_TG_COST_LEDGER.json
    HAWKING_TG_LEDGER_POSTMEMO.json        HAWKING_LEDGER_BF16.json
    GLM52_AA_PACK_DRY_RUN.json             GLM52_RARE_ROUTE_PILOT_PARTIAL_000{35,86,264}.json
    reports/condense/deepseek_v4_flash/pre_moe_hidden_L05.npy
    colab/data/eagle5_corpus/shard_00039.parquet

**Commit messages** — `grok/` and `codex/` were replaced with `lane/` where they appeared in
merge subjects, matching the branch renames done at the same time. No author or committer
field was touched: the history never carried a non-human author, a `Co-Authored-By` trailer,
or a generated-by footer, and an audit before and after confirms zero on all three.

**Refs removed** — five `refs/codex/turn-diffs/checkpoints/...` refs, an external tool's
per-turn checkpoint namespace that had accumulated inside this repository. Two commits were
reachable only from those refs and went with them.

**Nothing else.** `--prune-empty=never` kept every commit record, including the handful whose
only content was a removed blob. Commit count went 1820 to 1818, entirely accounted for by
the two checkpoint-only commits. The working tree was byte-identical before and after:
`HEAD^{tree}` was `776e640c94f77fab9f9809734252f83dab946b7a` on both sides.

`.git` went from 137 MB to 82 MB.

## Translating an old hash

`COMMIT_MAP_20260728.txt` maps every pre-rewrite commit to its post-rewrite hash, old on the
left, new on the right, one pair per line.

    grep -i ^<old-hash> docs/COMMIT_MAP_20260728.txt

Sealed receipts that cite pre-rewrite hashes were deliberately **not** edited — they are
evidence, and rewriting them to match would defeat the point of sealing them. Use the map.

Known citations of dissolved hashes: `fccb6b30` in ten HIDE archaeology and campaign status
files, `0adcab57` in one packs retirement record.

## Recovery

Pre-rewrite history is preserved in full:

    ~/Downloads/hawking-backup/hawking-pre-rewrite-7632915f.bundle
    ~/Downloads/hawking-backup/hawking-pre-rewrite-08bf0a61.bundle

    git clone hawking-pre-rewrite-08bf0a61.bundle recovered

## The remote has not been touched

`origin` (github.com:joshuahickscorp/hawking) still holds the pre-rewrite history, and the
local remote-tracking refs were dropped, so the two have diverged. Publishing this rewrite
needs a force-push, which would break every existing clone and invalidate the commit hashes
pinned in receipts and pull requests. That decision was left to the repository owner.
