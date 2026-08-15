# AgentOS nucleus — campaign substrate

Per steer S001 (hawking-tournament-ascent, "turn the tournament's PROVEN machinery
into the first real AgentOS"). This is NOT the future roadmap started early — it is a
thin, real capture of what the CURRENT tournament campaign already produces, so the
current controller and the eventual resident optimizer can query it instead of re-deriving.

Derived from reality (steer §8), not speculative frameworks:
- a Grok lane (contract + worktree + waiter + report + verification + branch) -> `lane` record
- a GOAL.md obligation gate (bit/byte-identity, coherence, dispatch) -> `verification` record
- a durable negative result -> `negative_science` record
- a reusable win/lever -> `genome` record

Files:
- `SUBSTRATE.jsonl` — append-only, one JSON object per line, keyed by `type`. Bounded to
  what this campaign actually generated (steer §6: no giant ontology).

Resource class of building this: DOC/SCHEMA + LIGHT_CONTROL (hand-authored file writes),
non-contaminating to the concurrent GPU-heavy clean-TPS lane (steer §4).

Rent paid (steer §9): preserves the knowledge the tournament is currently generating +
gives the controller a queryable record of what has been tried and verified.

Schema (minimal, per-type):
- lane:            {type,id,task,mission,resource_class,profile,state,base,outputs,receipts,verified_by_me,date}
- verification:    {type,lane,tier,gate,method,result,date}
- negative_science:{type,id,claim,evidence,transfer,applies_to,date}
- genome:          {type,id,genome,lever,before,after,invalidation,transfer,applies_to,date}

The `verify_*`, `profile_token`, `repack`, `promote_candidate` typed API (steer §5A) is
NOT implemented here — that is a code lane for an idle window with a clean measurement done.
This file is the data substrate those calls will read/write.
