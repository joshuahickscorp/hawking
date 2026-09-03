# HCLI Self-Improvement School

Claude teaches, supervises, verifies, and doctors failures.
HCLI operates, experiments, authors code, benchmarks, and is the subject.

The loop has no final goal. It stops when the human says so.

## Live truth at campaign start

Recovered from disk, not assumed. `python3 -m hcli.agentos.resident verdict`:

```
FAIL  prefill_fast             24 prompt tok/s over 27 calls (budget 100.0)
FAIL  progress                 accepted=0, 0 completed of 7 units
PASS  repair_budget_unspent    1 spent, all earned by this mission
PASS  repair_reached_the_unit  no depth-0 refusals
FAIL  structured_output_ok     2 of 5 receipts exhausted their retries
FAIL  tools_fast               p95 60.0 ms over 59 calls (budget 50.0 ms)
PASS  worker_stable            failure_streak 0

effective_prompt_tps=24.0  mean_rounds_per_goal=3.33
realized_reuse_fraction=0.1591  tool_p95_ms=60.0
```

Resident: `cycles=858  generation=903  restart_count=5  state=RUNNING`.
Durability is already demonstrated: five restarts survived, failure streak zero.

## P0 diagnosis: why accepted was 0

Three independent defects. The first one is the whole story.

### 1. The goal was unwinnable, and the contract lied about why (FIXED)

`Engine._validate`:

```python
if not test_list:
    result["ok"] = False
    result["reason"] = "NO_EVIDENCE"
```

`NO_EVIDENCE` maps to status `unverified`, which is terminal. A mutation that
names no test is discarded however good it is.

The system prompt said the opposite: `"tests": ["optional safe workspace-relative
Python test paths"]`. The model believed it, sent no tests, and every mutation
landed unverified.

Worse, the goal we set was unacceptable by construction. Only `.py` files have a
checker; every other suffix records `no_checker_available`. The deliverable was a
markdown document, so **no model output could have been accepted**. We read a
contract defect as a model defect.

Fixed in `1c746a27d`: the contract now states the rule and its consequence, the
example names a real test path, and a correlated-verifier test holds prompt and
validator together. Negative control included: a read-only `answer` still needs
no test.

### 2. Retries collapse to an empty reply (OPEN)

From `.hcli/receipts/dc104ddf*.json`, one goal's four model calls:

```
(prompt 2044, completion 206)   finish=stop
(prompt 3419, completion   2)   finish=stop
(prompt 3492, completion   1)   finish=stop
(prompt 3492, completion   1)   finish=stop
```

`max_tokens` was 3724 on every retry, so this is not a budget exhaustion. The
first attempt produces real content; every retry emits one or two tokens and
stops. Recorded as `errors: ['empty response','empty response','empty response']`
and `rejected_reply_chars: 0`.

Not yet explained. Candidates: the retry prompt shape, chat-template handling of
the rejected turn, or `enable_thinking` interaction.

### 3. `grammar_enforced` is never observed (OPEN)

The sealed profile `hcli/hawking-native.sealed-3.14.json` declares
`grammar: "supported"` (syntax masking only; `response_format: "unsupported"` is
correct and deliberate — the resident does not enforce schema). The resident
accepts `grammar: "json"`, masks logits, and returns `grammar_enforced`.

But `grammar_enforced` is `None` on every model call in every receipt. Either
the field is not propagated into the receipt, or the mask is not running. A
capability that cannot be observed cannot be trusted.

## Bar ladder

Pass a bar, freeze it, raise it. Bars move on evidence, not on schedule.

| facet | baseline | next bar |
|---|---|---|
| accepted goals | 0 | > 0, then majority, then high first-pass |
| structured output | 2 of 5 exhausted | 0 exhausted over 10 receipts |
| realized prefix reuse | 0.159 | 0.25, 0.35, 0.50, 0.65 |
| effective prompt tok/s | 24.0 | 1.25x, 1.5x, 2x, 3x |
| tool p95 | 60 ms | < 50, < 35, < 25 |
| rounds per goal | 3.33 | lower only if acceptance holds |

Never optimize one by silently degrading another. The quantity being maximized is
**verified accepted useful work per unit wall time**, not any single metric.

## Laws

- **A goal with no deterministic checker is unwinnable. Check the goal is
  acceptable before reading a failure as incapacity.**
- **When one rule lives in two places — a prompt and a validator — write the test
  that holds them together.** Either side can be relaxed without the other
  noticing, and the prompt is the side nobody runs.
- **A capability that is declared but never observed in a receipt is not known to
  work.** Grep for the call site and the reported field, not the definition.
- **Read what the worker actually received before judging what it produced.**
  Four missions were graded as model failures while the instruction was being
  deleted from the prompt by a sanitizer.
- **A constant calibrated on one kind of input is wrong for another.** Prose is
  ~3 chars per token, Python source ~2.4. Sizing a reserve for the last payload
  instead of the worst one fails on the payload that matters.

## Defects found by running the ladder

Thirteen, in the order the machine surfaced them. Every one sat BETWEEN the
model and the task. None was the model being weak.

| # | defect | commit |
|---|---|---|
| 1 | contract called `tests` optional; the verifier requires them, so `accepted=0` was structural | `1c746a27d` |
| 2 | token estimate calibrated on prose, applied to source: 25% under-count, context overflow | `b34f9e294` |
| 3 | reserve sized for the last payload, not the worst -- same overflow again | (same) |
| 4 | the instruction was excised from the worker's own OBJECTIVE line | `3ec9049c3` |
| 5 | ...and again on the whole assembled prompt, so fix 4 changed nothing for three runs | `78d8c30fc` |
| 6 | a directory launch dropped the sealed profile's capabilities: the grammar channel never ran | `b4ee8f21b` |
| 7 | `fs.search` named its location `root` while every sibling used `path`; the schema error read as "zero matches" | `134eccdda` |
| 8 | one tool observation could occupy the entire input window | (engine) |
| 9 | counting characters cannot size a window: exact tokenization instead | (engine) |
| 10 | `fs.read` had no offset, so deep code was unreachable in a 188 KB file | `43b827129` |
| 11 | `grammar_enforced` was never recorded, so a malformed reply could not be diagnosed | `424c289d7` |
| 12 | the JSON mask checked only a token's FIRST character; BPE tails broke JSON while it reported enforcement | `ca1dd50c3` |
| 13 | raw control characters were legal inside a JSON string | `6446f4428` |

Three of these were fixes for problems a previous fix of mine created. Two of
my own tests were vacuous and only mutation checks caught them -- both tested a
helper instead of the path that actually failed.

## What the model actually did

Level 1 (read and explain): PASSED. Correct answer on why `pid_is_alive` reaps
before testing liveness, and it correctly flagged its own evidence as weak
because a broken tool had told it there were zero matches.

Level 3 (patch + test): one genuine attempt. It produced a real `mutation`,
patched the source AND the test, and named a test -- the combination acceptance
requires and which no run had ever produced. The source patch was CORRECT and
compiled. The test patch dropped three closing parens and failed `py_compile`,
so the verifier rolled the whole mutation back on deterministic evidence.

The repair then returned `kind: answer` carrying operations, which are never
applied to disk, and the mission failed. `accepted` is still 0.

Standing weakness, and the first that is genuinely about the model: it writes
short patches correctly and long verbatim code inside a JSON string
unreliably.

## Ladder progress

Levels from the campaign directive. Level 1 is read-and-explain, and it took
four attempts, each blocked by a different real defect:

| attempt | blocker | fix |
|---|---|---|
| 1 | goal prose shredded into 6 fake obligations | one-sentence goals (compiler defect still open) |
| 2 | context overflow, 6605 tok vs 8192 window | learn chars-per-token (`b34f9e294`) |
| 3 | same overflow: density is per payload, not learned | reserve 12% -> 30% |
| 4 | `OBJECTIVE: obligations=G001 [ROOT_GOAL_OMITTED]` | keep the root when it IS the objective (`3ec9049c3`) |

Bars moved on the run that got furthest:

```
structured_output_ok      FAIL -> PASS   (0 of 1 exhausted)
realized_reuse_fraction   0.159 -> 0.2986   (first bar 0.25 cleared)
effective_prompt_tps      24.0 -> 28.5
tool_p95_ms               60 -> 51
```

New red surfaced by getting further: `no_tool_loops` FAIL, 9 of 25 requested
calls were duplicates, and `mean_rounds_per_goal` rose 3.33 -> 7.0 because of
them. That is the next front after the ladder moves.

## Scars

- Told HCLI to write a `.md` deliverable, then measured its failure to be
  accepted as a quality problem. It was a contract problem. Cost: a full mission
  and several hours of resident time.
- Diagnosed `constrained_decoding: unavailable` as a wiring gap before reading
  the profile, which correctly declares `response_format: unsupported`. Reading
  the declaration first would have cost one command.
- Graded the model degenerate for echoing its prompt, then found the prompt had
  had its instruction excised. The model's own reply said so plainly and was
  recorded as a failure. Cost: most of the campaign to date.
- Shipped a learned chars-per-token ratio as the fix for a context overflow,
  and the very next rep reproduced the overflow. The ratio was right and
  irrelevant: it learns from the previous call, and density belongs to the
  current payload.
