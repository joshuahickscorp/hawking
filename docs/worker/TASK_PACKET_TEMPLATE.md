# Hawking external-worker task packet template

Use this concise packet for every serious external-worker task. The
hawking agent stage command generates a filled packet with the recorded base,
worktree, receipt, and fixed safety boundary.

## OBJECTIVE

State one concrete engineering objective.

## SOURCE COMMIT

Record the exact canonical revision from which the staging worktree was made.

## WORKTREE

Record the exact isolated worktree path and worker branch.

## ALLOWED ACTIONS

List the allowed read, edit, compile, test, and evidence actions.

## FORBIDDEN ACTIONS

List production deployment or promotion, canonical-branch mutation, sudo or
secrets, daemon control, model/GPU work, teacher admission, capability
certification, and recursive agent invocation unless an explicit higher
authority changes the task contract.

## AUTHORITATIVE INPUTS

Name the source files, continuation, canonical registries, receipts, and tests
that settle the task. State that worker memory is navigation help only.

## EXPECTED OUTPUT

Return a reviewable commit or diff, exact checks and results, unsupported
assumptions, remaining risks, supported claims, and suggested review locations.
If no code is necessary, return an evidence packet.

## TESTS

Name the smallest relevant CPU-only compile/test checks. Include Git
whitespace checking.

## CLAIM BOUNDARY

State what the task may establish and what it cannot establish. Staging
evidence is not promotion, deployment, capability certification, source-teacher
authorization, model qualification, or production authority.

## STOP CONDITIONS

State when to stop, what evidence to return, and the exact condition that would
reopen the blocked route.
