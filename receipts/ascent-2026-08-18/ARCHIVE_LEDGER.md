# 2026-08-18 ascent snapshot archive

The two dated snapshot reports in this directory were discovery outputs, not
current authorities. Their durable signal is retained here; detailed prose is
recoverable from Git history.

## G11 — two-tier Matryoshka NR

Measured CPU/Numpy on real Qwen3.8 MLP BF16 data: a uniform-q3 base plus a
uniform-q2 residual correction reduced mean weight relative error from
0.25616898 to 0.12515527 across six tensors, and post-SwiGLU `down_proj`
relative error from 0.22742779 to 0.11110732 across three layers. The
correction native decoder was not built; this was a stored-structure result,
not a runtime or promotion claim.

## G16 — dispatch trace

Measured Qwen3.8 hybrid decode showed the declared 38-name NX kernel set was
not the runtime dispatched union: the two required artifacts dispatched 18
names, four of which were absent from the declaration, while 24 declared
names did not fire under the default environment. The trace was therefore a
diagnostic of declaration drift, not evidence that all declared kernels run.

Both snapshots remain archival findings only. Current dispatch, artifact, and
representation authorities are the live Rust runtime, acceptance tests, and
sealed receipts.
