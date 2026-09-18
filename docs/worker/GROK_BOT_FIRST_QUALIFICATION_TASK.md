# Grok Bot first Hawking qualification task

This packet is prepared for a future authenticated Grok Bot session. It is not
an execution request and must not be run until the user enables the supported
Grok Bot path.

## OBJECTIVE

Inspect the current 12 Flash Organ Passports and the canonical Gravity registry
using the generated Hawking context. Identify one deterministic knowledge or
validation gap that would force a future compatible model to rediscover known
Flash information. If a small staging-only fix is justified, implement it in
the isolated worker worktree, run the smallest focused CPU-only checks, and
return a reviewable commit or evidence packet.

## START

From the canonical checkout, stage a fresh task and then generate its view:

    hawking agent stage grok passport-check \
      --objective "Inspect the 12 Flash Organ Passports and canonical Gravity registry; identify one deterministic validation gap and make at most a staging-only test improvement." \
      --scope tools/foundry --scope tools/odyssey --scope docs/worker
    hawking agent context grok passport-check

Give the worker only the printed task packet, printed staging worktree, and
generated `passport-check.context.json` / `passport-check.context.md` files.
After the worker stops, observe it from the canonical checkout:

    hawking agent result passport-check --json

## ALLOWED

- Read/search the staging worktree and generated context.
- Inspect the 12 passports, Gravity methods, Laws, Scars, receipts, and tests
  named by the context.
- Edit only the isolated worktree and recorded scope.
- Run explicitly selected CPU-only tests and `git diff --check`.
- Write `.hawking-worker/result.json` and optionally prepare a reviewable commit.

## FORBIDDEN

- No canonical-checkout mutation, push, promotion, or deployment.
- No daemon/Web control, credentials, sudo, model loading, GPU/Metal lease,
  Kimi/Flash runtime, TPS, full replay, teacher admission, capability claim,
  Gravity promotion, or Pulsar promotion.
- Do not alter scientific conclusions merely to make a test pass.

## STOP / CLAIM BOUNDARY

Stop and report the exact blocker if stronger authority, protected runtime,
credentials, or novel scientific adjudication is required. A staging diff,
passing test, commit, or worker report is advisory evidence only; Hawking's
independent `agent result` observation remains the review boundary.
