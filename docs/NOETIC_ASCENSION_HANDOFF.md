<!-- DOC_STATUS: CURRENT -->
# Noetic ascension handoff

Cold-read this file, then the JSON. Do not resume from chat memory.

## Where the campaign stands

No Noetic candidate beats the parent. The measured blocker is in
`receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json`: 44.7% fewer bits per
weight bought 1.9% throughput, because decode is dispatch-bound, not
bandwidth-bound, at 964 dispatches per token either way. The next family is a
non-matvec operator that survives composition.

A handoff that reads as a success story would misdirect whoever picks this up.

## The two artifacts

| Path | What |
|---|---|
| `receipts/headless/NOETIC_ASCENSION_HANDOFF.json` | Every field is read from a receipt and carries a reproducing command. |
| `receipts/headless/NOETIC_ASCENSION_LOOP.json` | Demonstration transcript. HCLI drove AgentOS, a resident observation, Doctor, a Gravity experiment, tools, a verifier, REJECTED, and a science update. Each step shows the command that produced it. |

## Reproduce

```
python3 tools/headless/noetic_ascension_loop.py
python3 -m pytest tools/headless/test_noetic_ascension_handoff.py -q
python3 -m pytest tools/headless -q
```

Canonical HCLI launch (from `docs/CURRENT_ARCHITECTURE.md` at HEAD):

```
python3 -m hcli
```

This worktree is a sparse checkout. `hcli/` may not be on disk. The loop extracts
`HEAD:hcli` with `git archive` (do not `git sparse-checkout add` — it fails here).
`git show HEAD:<path>` reads a file that is not materialized.

## Rollback

See `rollback_command` in the handoff JSON. It deletes this lane's new files.
It does not touch `hcli/`, `receipts/ascent-2026-08-16`, `workspace/campaign`,
or historical receipts (`CODE_ENTROPY.json` `never_delete`).
