# Governance rung snapshot archive

The live capability comparison retains the canonical `before/pre-s2` and
`after/post-s2b` snapshots. The latest `verify/verify-s3b` snapshot remains as
the final verification state. The intermediate snapshots below were generated
repetitions of the same capability/test inventories, not independent product
evidence; their complete contents remain recoverable from Git history.

| retired snapshot | reason | retained signal |
| --- | --- | --- |
| `before/pre-s2b` | superseded pre-rung capture | canonical pre-S2 baseline |
| `after/post-s3` | superseded post-rung capture | latest post/verify state |
| `after/post-s3b` | superseded post-rung capture | latest post/verify state |

The surviving pair and final verification snapshot preserve the comparison
inputs required by `tools/verify/capability_manifest.py`; no product or
capability decision is inferred from the retired copies.
