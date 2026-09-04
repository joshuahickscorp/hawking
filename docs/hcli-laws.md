# HCLI Laws

Permanent. Each was paid for with measured failure, not reasoning.

## LAW 1 — File selection and evidence focusing are separate operations

A path list may choose WHAT to read. It must never determine WHERE inside the
selected file cognition is focused.

`gather_evidence_paths` passed `" ".join(paths)` to `_gather_evidence`, which
focuses each file on the identifiers it shares with the text it is given. For
`hcli/tool_registry.py` those identifiers were `{hcli, tests, tool_registry}` --
none of them symbols in that file. The anchor fell through to density scoring
and selected lines 1838-1964, the tool registration block, on **25 consecutive
calls**, while the goal named `_list_files` at line 677.

The model edited what it was shown. Every time. It was recorded as targeting the
wrong code for most of a day.

## LAW 2 — Diagnose from the bytes delivered and returned, not from the source

Source inspection is not evidence of runtime cognition.

Reading the code produced three confident and wrong conclusions in a row: that
the excerpt was correct (calling the function directly gave the right window),
that a downstream head slice truncated it, and that the model was patching from
memory. A single instrumented run that wrote the posted evidence to disk settled
it in one attempt.

Byte counts cannot answer "which 6027 characters". `HCLI_DUMP_EVIDENCE` exists
for that question and should be reached for early, not last.

## LAW 3 — Model failure may not be assigned until evidence delivery is inspected

Before concluding that the resident cannot reason, serialize, or follow an
instruction, physically confirm that the instruction and the evidence reached
it.

Every one of these was assigned to the model first and turned out to be
delivery:

| symptom | actual cause |
|---|---|
| "edits the wrong function" | shown the wrong 6 KB on every call |
| "answers instead of mutating" | objective compiled to `Do not edit it.` |
| "emits an empty operation" | objective was `[ROOT_GOAL_OMITTED]` |
| "never closes the JSON object" | 256-token ceiling from an inflated prompt count |
| "ignores the tools it is told not to use" | env never reached the worker |
| "loops inside its own array" | prompt told it to prefer that array form |

The tally at the end of the bootstrap: 12 model-facing defects, 11 of them
PRE-COGNITION or INTERACTION, one VERIFICATION, and **zero** that survived
inspection as MODEL_REASONING.

## Corollary — corrective feedback must be actionable from observable state

Never reference hidden post-transform coordinates, omitted output, unavailable
filesystem state, or a capability the recipient cannot invoke. Checked by
`tools/hcli_laws.py`, which runs the workflow a remediation promises rather than
asserting that the sentence exists.
