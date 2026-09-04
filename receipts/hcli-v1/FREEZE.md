# HCLI v1 — frozen heartbeat

The exact path that produced the first two accepted HCLI-authored mutations.
Preserved under S004 Phase 1. Do not refactor because it can now be improved;
refactor when a measurement says it must, and re-run both gates after.

## Commits

| what | commit |
|---|---|
| the fix that unblocked Gate 1 | `f111dee0e` evidence window anchored on the path list |
| Gate 2 and its self-repair | `db5b1a4` (see `git log --grep "second mutation"`) |
| the Laws | `docs/hcli-laws.md` |
| known-good config | `docs/hcli-first-known-good.md` |

## Resident

- sealed profile `sealed-3.14`, `hcli/hawking-native.sealed-3.14.json`
- binary `workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident`
- 64 layers (48 DeltaNet, 16 GQA), `max_seq_len` 8192
- greedy: temperature 0.0, top_k 1, do_sample false

## Daemon

```
HCLI_NO_TOOLS=1 HCLI_MAX_TOOL_ROUNDS=1 \
  python3 -m hcli.hawkingd --supervise .hcli/resident/state.json
```

Both variables must be exported to the `replace` submit as well: it starts its
own supervisor, and a submit run without them silently spawns a second daemon
with a clean environment. Kill BOTH supervisor and resident with `-9` before a
restart; they accumulate and an old worker will serve stale code.

## Mutation ABI

Full envelope with `old_text`/`new_text` is what actually landed both gates.
`old_lines`/`new_lines`, the micro form `{path, find, replace}` and the
`PATH:/FIND:/REPLACE:/END` patch block are all supported and all live, but none
of them was the thing that unblocked anything. Serialization was never the
frontier. Do not advertise the line-array form as preferred: doing so steered
greedy decoding into a repetition loop inside its own array.

## Context compiler

- goals are ONE marker-free objective sentence naming the files
- previewed with `tools/goal_preview.py` before any model call: any sentence
  containing "do not", "must", "never", "only" becomes an INVARIANT and the
  objective falls through to whatever is left
- evidence focused on the objective line, never on the path list (LAW 1)
- `HCLI_DUMP_EVIDENCE=<path>` writes the bytes actually posted (LAW 2)

## Verifier

`Engine._validate` runs py_compile then pytest on the model-supplied test list.
`red_before_green` must be True: the harness computes it correctly and records
it as advisory, so `tools/gate_report.py` treats False as fatal and refuses to
report LANDED. A test that cannot fail is not evidence.

## Gate evidence

Receipts copied beside this file. Both: `rolled_back` False, `validation.ok`
True, `red_before_green` True, py_compile exit 0, tests 3 collected 3 passed.
