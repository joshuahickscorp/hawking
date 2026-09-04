# HCLI first known-good coding configuration

Frozen on the first accepted mutation. Do not refactor this path without a
reason and a re-run.

Receipt `f1c4a821`, 2026-09-04.

## The mutation HCLI authored

```diff
         "truncated": truncated,
+        "directories_seen": directories_seen,
```

in `_list_files`, `hcli/tool_registry.py`. Claude wrote the failing test and
the goal; Claude did not write the patch.

## Physical evidence

| criterion | value |
|---|---|
| kind / status | mutation / completed |
| rolled_back | False |
| validation ok | True |
| red_before_green | **True** |
| test | exit 0, collected 3, passed 3 |
| py_compile | exit 0 |
| resident calls | 1 |
| wall | 133 s |
| tool calls | 0 |

## Configuration

- resident: sealed-3.14, `ascension_qwen38_resident`, 64 layers
- `HCLI_NO_TOOLS=1` — empty tool bundle, exported to BOTH the daemon and the
  `replace` submit, because the submit starts its own supervisor
- `HCLI_MAX_TOOL_ROUNDS=1`
- one daemon only; `pkill -9` both supervisor and resident first, they
  accumulate silently and an old worker will serve stale code
- goal: ONE marker-free objective sentence naming both files, previewed with
  `tools/goal_preview.py` (0 invariants) before any model call
- the failing spec already on disk; the goal says so explicitly
- evidence: focused excerpt anchored on the GOAL, not the path list

## The task shape that worked

A value already computed and in scope, missing from a returned dict. One line,
one file, no new logic, nothing to design. Clause 11 of the directive: do not
select a clever task. The earlier `total_lines` task failed repeatedly because
the whole-file branch had no `lines` variable, so the fix required writing new
decode-and-split logic rather than copying what was adjacent.
