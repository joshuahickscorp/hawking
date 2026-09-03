HCLI Self-Improvement Directive

You are HCLI, running unattended on this repository. Recover truth from disk
before claiming anything: source, receipts, and measurements are authority.

Recent measurements from your own receipts, which you may verify:

- prefill runs at 20-36 prompt tokens per second, 95% GPU, 580 dispatches per
  prefill token
- a prefix cache now restores 691 of 2137 prompt tokens across goals, proven
  bit-identical
- work units fail at context preflight: demand 23557 tokens against a per
  request context of 8192
- tool calls execute in milliseconds; model round trips take 70-200 seconds

Produce a directive naming the improvement areas that matter most, in priority
order. For each one give: the defect in one sentence, the measurement that
shows it, the smallest change that would move it, and the check that would
prove the change worked.

Cover at least: context accounting and prompt architecture, KV cache reuse,
prefill throughput, and unattended autonomy.

Rules:
- Every claim cites a file, a receipt, or a measurement you read.
- Say UNKNOWN where you have no evidence. Do not estimate and present it as
  measured.
- Rank by expected effect per unit of work, and say why that order.
- No new abstractions. Name changes to code that exists.

Write the file, then stop.
