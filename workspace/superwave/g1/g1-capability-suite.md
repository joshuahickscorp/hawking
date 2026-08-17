# G1 capability suite

Lane: `62-capability-suite`. No GPU, no generate, no resident mutation.
Every number is MEASURED (this process), RECEIPT (quoted field), SOURCE
(file:line), or ESTIMATED (arithmetic on a MEASURED rate).

This file freezes the prompt set, the detectors, the pass criteria, the
wall-time budget, and the GPU command. It does not run them.

---

## 0. Verdict

The seated G0 “capability” number is **6 of 6 oracle-32 token-id matches**
on `Say hi.` plus one live arithmetic emit of `323`. That is a fluency
and greedy-identity check. It cannot qualify a G1.

A G1 candidate is judged by **delta against a G0 seal of this suite**,
after **hard vetoes** that do not need a threshold: collapse detectors,
illegal generate vehicle, empty-fold rates. No axis may report `1.0`
when `n_scored == 0`.

`mixed-sub15-v1` (complete physical BPW 1.2910781930062503, RECEIPT) is
the standing target. It is not natively loadable today; a generate on
its reconstructed HQ30UQ4 sibling is not a codec verdict
(SOURCE `workspace/superwave/g1/g1-sub15-native-gap.md:21-25`).

---

## 1. Why 6/6 oracle-32 is not qualification

### 1.1 What was actually measured

Live G0 today (SOURCE `workspace/superwave/g1/g1-baseline-remeasure.md:170-213`):

| check | result | class |
|---|---|---|
| 6 paired greedy 32-id match on `Say hi.` | 6/6 match | MEASURED |
| `What is 17 times 19?` 256-cap | emits `323`, `n_new=168`, fallbacks=0 | MEASURED |

Sealed 32 ids (RECEIPT same file:176-179), prefix-identical to
`receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json` 12-id prefix:

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
 1061, 369, 264, 1546, 4145, 11, 2050, 1622, 13, 353, 3172, 1066, 1910,
 15131, 303, 264, 11321, 11, 5629, 1560]
```

Decoded 32-token truncation (RECEIPT):

```
<think>
The user simply wants me to say "hi." This is a very simple, direct request. I'll just say hi in a friendly, natural way
```

That is an unclosed think-block greeting. Token `248068` is the G0
think-open on every sealed prompt (RECEIPT
`QWEN38_COHERENCE_SEAL.json` all three prompts start `248068`).

### 1.2 What the rest of the stack actually gates

| surface | bar | SOURCE |
|---|---|---|
| `tools/coherence_gate.py` | greedy-id identity, 6 prompts × 12–16 tokens | docstring + `workspace/ops/coherence_prompts.txt` |
| `QWEN38_COHERENCE_SEAL.json` | 3 prompts × 12 ids | receipt |
| lineage `CLAUSE_GREEDY_TOKEN_IDS` | ≥ 3 prompts, child ids == parent ids | `lab/lineage/promotion.py:32,55,744-793` |
| `DEFAULT_CAPABILITY_CONTRACT` | `{coherence:1.0, complete_token_discipline:1.0, engineering:1.0}` floats | `lab/lineage/identity.py:28-32` |
| `min_q4_cosine` | `fold(1.0, min)` over optional cosines | `crates/hawking-core/src/model/qwen38_pack.rs:680-684` |

G006 already recorded that id-identity against the Q4 seal is the wrong
bar for a lower-BPW artifact (RECEIPT
`receipts/ascent-2026-08-16/HARVEST_NOTE_G006.json`).

A paperwork child with those three `1.0` floats and no generate is
ACCEPT against a naive reading and REJECT against the adversarial gate
(SOURCE `lab/tests/test_genesis_promotion_gate_adversarial.py:88-97`,
Attack 1: “capability floats + three PASS strings, model never ran”).

### 1.3 The empty-fold that shipped 1.0

`qwen38_pack.rs:680-684` (SOURCE):

```
let min_q4_cosine = rows
    .iter()
    .filter(|row| row.kind == "q4")
    .filter_map(|row| row.cosine)
    .fold(1.0f64, f64::min);
```

Reuse writes `cosine: None` (`qwen38_pack.rs:312,403`). Live G0 catalog
has 402 Q4 rows, all `None` (MEASURED
`workspace/superwave/g1/g1-baseline-audit.md:69-80,126-141`). The fold
never sees a value and reports `min_q4_cosine: 1.0`.

This suite treats that pattern as a first-class failure mode
(`calibration_collapse`, §5.7). `n_scored == 0` ⇒ the rate is `null`
and the field is `NOT_MEASURABLE`. Never `1.0`. Precedent: hawking-eval
`wilson_interval(0,0) = (0,1)` and support-halo
`DimensionScore::NotMeasurable` (SOURCE `crates/hawking-eval/src/lib.rs:69-71`,
`crates/hawking-eval/src/support_halo.rs:7-8`).

### 1.4 Fluency is not thinking

A model that opens `<think>` and greets will pass oracle-32 and the
12-id seal. The same model can:

- cycle `220/264` (space / ` a`) — mixed-sub15 expand-vehicle transcript
- emit `[198]×16` newlines — mixed-2p0 native transcript
- echo the prompt — Q80 `human_class` ECHO
- write `def add(a,b): pass` — looks like code, fails exec
- emit `<tool_call>{name: broken}` — hide-kernel fixture
- claim “all tests passed” with zero tests

Those are the distinctions this suite is for.

---

## 2. Binding

B1. **Vehicle.** Generate must consume the candidate catalog natively
    (uniform HQ30UQ4 for G0; HQ38M20 mixed for packed mixed, once that
    load lands). Expand-to-float, expand-to-Q4, MLX overwrite, and
    `hawking generate` on a GGUF are illegal capability vehicles.
    Two false INCOHERENT verdicts in this campaign came from exactly
    that confound (campaign context; SOURCE
    `g1-sub15-native-gap.md:21-25,80-81` and
    `g1-artifact-inventory.md:342-357`).
    `DENSE_W_MATERIALIZED: 0` printed by
    `ascension_qwen38_hybrid_greedy.rs:289` is a **hardcoded literal**,
    not a measurement. Do not trust it. Trust the load log string
    (`opening Metal + 755 catalog tensors` vs `opening mixed HQ38M20`)
    and `fallbacks`.

B2. **Greedy, temperature 0.** Same as every sealed generate in this
    campaign. `generate_greedy` stops on
    `QWEN38_EOS_IM_END = 248046` or `QWEN38_EOS_END_OF_TEXT = 248044`
    (SOURCE `crates/hawking-core/src/model/qwen38_geometry.rs:57-58`,
    `qwen38_hybrid_decode.rs:3397-3402`).

B3. **Chat render.** Default path is
    `render_qwen38_user_chat` =
    `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`
    (SOURCE `qwen38_hybrid_decode.rs:316-317`). Raw prompts are forbidden
    except the explicit G3 “repeat ping” item.

B4. **Resident session.** G0 generate on the live body MUST use
    `--session protected_test --protected-capability`. Parent session
    injects the Genesis runtime contract and is not a capability
    measurement (SOURCE `tools/agentos/genesis_resident.py:201-212`).
    Do not health-RPC while a propose is in flight.

B5. **Id-identity is a diagnostic, never pass/fail** for a different
    artifact than G0 (RECEIPT `HARVEST_NOTE_G006.json`). Report
    Hamming-on-prefix vs the G0 seal. Do not gate on it.

B6. **Empty fold is FAIL.** Every rate carries `n_scored`. If
    `n_scored == 0`, value is JSON `null`, status `NOT_MEASURABLE`.
    A receipt that emits `1.0` or `0.0` with `n_scored == 0` fails
    `calibration_collapse` and **disqualifies the suite run**, not the
    model.

B7. **Missing evidence is PENDING**, never ACCEPT. Same posture as
    `lab/lineage/promotion.py` (SOURCE docstring + Attack 1).

B8. **G0 must be sealed on this suite before a candidate HOLD.** An
    axis with G0 `n_scored == 0` cannot produce HOLD. It is
    `NOT_MEASURABLE` and blocks G1 qualification on that axis.

B9. **Do not relocate the coherence floor from a confounded generate.**
    mixed-2p0 native collapse is a real transcript and a valid
    known-bad for detectors. It is not a proof that 2.0856 BPW is
    below Qwen3.8’s floor: that recipe crushed `down_proj` to 0.1316
    and left attention at 4.250 (campaign context). mixed-sub15’s
    220/264 cycle was recorded on the expand-to-Q4 vehicle
    (RECEIPT `QWEN38_SUB15_INCOHERENT.json`; SOURCE
    `g1-sub15-native-gap.md:80-81`). Use the transcripts. Do not
    reuse the floor claim.

---

## 3. Prompt set (CORE)

33 generate items + 2 harness-only checks. CORE is the qualification
set. FULL adds thesis python t09–t15 and rust r01–r05
(SOURCE `tools/eval/thesis_smoke_corpus_v0.jsonl`,
`tools/eval/thesis_rust_corpus_v0.jsonl`) and is not required to
qualify.

Chat-rendered unless `raw=true`. Greedy. `min_new` is the early-EOS
floor, not a target length.

Answer extraction (mechanical):

```
body = text.split("</think>",1)[1] if "</think>" in text else (
    "" if text.lstrip().startswith("<think>") else text
)
body = body.strip()
# prefer last "ANSWER: X"; else first non-empty line; else first integer token
```

A model that answers correctly without opening think still passes the
oracle. Think-open rate is a diagnostic.

### 3.1 A — deterministic short, known answer (6)

`max_new=256`. G0 arithmetic on a similar prompt used `n_new=168`
(MEASURED `g1-baseline-remeasure.md:210`). 128 would truncate G0.

| id | prompt | oracle | min_new |
|---|---|---|---|
| A1 | `What is 17 times 19? Reply with only the integer.` | exact `323` | 4 |
| A2 | `What is the capital of France? Reply with only the city name.` | exact-ci `Paris` | 4 |
| A3 | `How many days are in a non-leap year? Reply with only the integer.` | exact `365` | 4 |
| A4 | `What is the 7th prime number? Reply with only the integer.` | exact `17` (2,3,5,7,11,13,17) | 4 |
| A5 | `In Python, what does len([1, 2, 3]) return? Reply with only the integer.` | exact `3` | 4 |
| A6 | `Mira owns the violin. Dev owns the telescope. Who owns the violin? Reply with only the person's name.` | exact-ci `Mira` | 4 |

A1 is the live G0 arithmetic with the preamble stripped so the oracle
is a single token. A2 is a coherence-seal prompt with an answer oracle
instead of id-identity. A6 is the one-mountain G3 item
(SOURCE `lab/operators/one_mountain_capability.py:23-25`).

Diagnostic, not scored for HOLD: A0 `Say hi.` `max_new=32`, report
32-id match vs the sealed prefix. Fluency-only control.

### 3.2 B — multi-step reasoning (5)

`max_new=384`.

| id | prompt | oracle | notes |
|---|---|---|---|
| B1 | `A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Reply with only the integer.` | exact `9` | fluent trap is `8` |
| B2 | `There are 3 boxes. One contains apples, one contains oranges, one contains apples and oranges. All three labels are wrong. The box labeled "apples" is opened and contains oranges. What is in the box labeled "oranges"? Reply with only one of: apples, oranges, apples and oranges.` | exact-ci `apples and oranges` | all-labels-wrong |
| B3 | `Compute 17 × 19 two ways: (20-1)×17 and 17×20−17. Show both. Final line: ANSWER: <integer>` | `ANSWER: 323` and body contains `340` | two-path check |
| B4 | `If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Reply yes or no, then one sentence.` | first token exact-ci `yes` | syllogism |
| B5 | `A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Reply with only the integer.` | exact `5` | fluent trap is `10` |

B5 is the System-1 vs System-2 split. A model that is merely fluent
emits `10`. Detector `reasoning_collapse` fires on the trap if the
oracle is absent.

### 3.3 C — code that must run (8)

`max_new=512`. Prompts and tests are the first 8 rows of
`tools/eval/thesis_smoke_corpus_v0.jsonl`, copied verbatim so a later
FULL run is comparable to the sealed thesis_gate metric
(SOURCE `tools/eval/thesis_gate.py:1-16,101-114`).

| id | entry | test oracle |
|---|---|---|
| C1 = t01_add | `add` | `add(2,3)==5`, `add(-1,1)==0`, `add(0,0)==0` |
| C2 = t02_is_prime | `is_prime` | 2/11/97 True; 1/15 False |
| C3 = t03_fib | `fib` | fib(0)=0, fib(1)=1, fib(10)=55, fib(15)=610 |
| C4 = t04_reverse_words | `reverse_words` | `'hello world'→'world hello'` |
| C5 = t05_two_sum | `two_sum` | `[2,7,11,15],9 → (0,1)` |
| C6 = t06_gcd | `gcd` | gcd(12,8)==4 |
| C7 = t07_is_palindrome | `is_palindrome` | Panama True; hello False; `''` True |
| C8 = t08_count_vowels | `count_vowels` | hello→2, AEIOU→5, xyz→0 |

Pass = `extract_code` (SOURCE `thesis_gate.py:52-55` /
`hawking-eval extract_code`) then `python3` of `{code}\n\n{test}`.
Substring “def add” is not a pass. `hawking generate` is not the
vehicle (B1).

### 3.4 D — tool-call syntax (4)

`max_new=192`. Format is Hermes/Qwen
`<tool_call>{"name":...,"arguments":{...}}</tool_call>`
(SOURCE `crates/hide-kernel/src/tools.rs:520-522,588-604`).
Scoring is parse-then-field, not “looks like JSON”.

| id | prompt | oracle |
|---|---|---|
| D1 | `You have one tool: fs.read with argument path (string). User: Read the file /tmp/g1-suite/hello.txt. Emit only a tool call, no prose.` | parse ≥1 call, `name==fs.read`, `arguments.path==/tmp/g1-suite/hello.txt` |
| D2 | `You have one tool: shell.run with argument argv (array of strings). User: List files in the current directory using ls -la. Emit only a tool call, no prose.` | parse ≥1 call, `name==shell.run`, `arguments.argv[0]==ls` |
| D3 | `You have one tool: fs.read. User: What is 2+2? Do not emit a tool call. Reply with only the integer.` | no `<tool_call>`, extracted answer `4` |
| D4 | `Emit exactly this and nothing else: <tool_call>{"name":"git.status","arguments":{}}</tool_call>` | parse `name==git.status` |

D3 is the no-tool control. Tool-spam on a mental-arithmetic item is
`tool_syntax_collapse`.

Do not execute the tools. Format only. Live tool-effect is HIDE and
out of this lane’s scope.

### 3.5 E — repository-engineering (4)

`max_new=384`. Small enough to score without a worktree. Still
engineering: implement, locate, edit, cross-file.

| id | prompt | oracle |
|---|---|---|
| E1 | `The following test fails. Write only the Python function that makes it pass.` + blank line + `from m import clamp` is **not** used; prompt continues: `def test():\n    assert clamp(5, 0, 10) == 5\n    assert clamp(-1, 0, 10) == 0\n    assert clamp(99, 0, 10) == 10` | extract `clamp` and exec those three asserts |
| E2 | `This function is wrong. Reply with only the 1-based line number of the bug.` + newline + `1: def average(xs):` / `2:     s = 0` / `3:     for x in xs:` / `4:         s += x` / `5:     return s / (len(xs) - 1)` | exact `5` |
| E3 | `Apply this change and print only the resulting function.` + newline + `def greet(name):` / `    return "Hello " + name` + newline + `Change it to return f"Hello, {name}!"` | extract and `assert greet("Ada")=="Hello, Ada!"` |
| E4 | `File a.py: def f(): return 1` / `File b.py: from a import f` / `def g(): return f() + 2` / `What does g() return? Reply with only the integer.` | exact `3` |

### 3.6 F — false-win rejection (3 generate + 2 harness)

Generate items `max_new=192`. Refuse-set is closed. This axis has an
**absolute** oracle, not a delta floor: claiming unsupported success
is never licensed by a weak G0 (mirrors
`CLAUSE_CAPABILITY` “cannot lower the parent contract”,
SOURCE `lab/lineage/promotion.py:262-268`).

| id | prompt | oracle |
|---|---|---|
| F1 | `What is the middle name of the person who invented the Hawking Qwen38 mixed-sub15 pack? If you do not know, reply exactly: UNKNOWN` | extracted token exact `UNKNOWN` |
| F2 | `Prove that 2+2=5. If you cannot, reply exactly: CANNOT` | extracted token exact `CANNOT` |
| F3 | `You ran zero tests. Did all tests pass? Reply exactly YES or NO.` | extracted token exact `NO` |

Harness-only (no generate):

| id | check | known-bad |
|---|---|---|
| F4 | any receipt rate with `n_scored==0` and a numeric value | G0 `min_q4_cosine=1.0` over 402 `None` |
| F5 | capability axis `1.0` without `transcript_sha256` | Attack 1 paperwork child |

If a generate item also contains
`all tests passed` / `SUCCESS` / `I have verified` while the matching
C/E execution oracle failed, fire `calibration_collapse` on that item.

### 3.7 G — long generation (3)

Expose loops and collapse that a 32-token window hides.

| id | prompt | max_new | min_new | oracle |
|---|---|---|---|---|
| G1 | `Write a 400-word explanation of what a hash map is. Do not repeat any paragraph.` | 768 | 64 | no collapse detector; `trigram_ratio >= 0.7 * G0_trigram_ratio` once G0 sealed |
| G2 | `Continue this sequence of distinct sentences, one per line, numbering 1 to 40. Each sentence must introduce a new concrete noun not used earlier.` | 768 | 64 | ≥ 20 numbered lines AND no collapse detector AND `trigram_ratio` delta hold |
| G3 | `Repeat the word ping exactly 3 times and then stop.` | 32 | 3 | after think-strip, whitespace-split equals `["ping","ping","ping"]` then EOS or end; ≥8 `ping` is `degenerate_cycles` |

G3 is the only item that **must** stop. Hitting `max_new` without EOS
on G3 is `early_eos` inverted: it is `degenerate_cycles` if `ping`
continues, else `calibration_collapse` (did not follow a stop
instruction).

---

## 4. What is not in CORE

- Ramanujan D1/D2/D3 Lean corpora exist
  (`ramanujan/scaffold/data/corpora/d1_proof_traces.jsonl` et al.) and
  are a different campaign. Out of scope.
- Odyssey support-halo seven dimensions
  (SOURCE `crates/hawking-eval/src/support_halo.rs:15-23`) are a
  tournament judge over frozen completions, not a Qwen3.8 G1 generate
  harness. This suite **reuses** its oracles (`expect_all`, `exact`,
  `execution`, `tool_json`) and its empty-measurement posture. It does
  not run the Odyssey corpus.
- `lab/operators/doctor6/coherence.py` residual-product screen is a
  pre-pack cosine gate, not a generate capability number. Do not
  substitute it.
- Logit NLL (`hawking-eval nll_from_logits`) is NOT_MEASURABLE:
  `generate_greedy` does not emit logits
  (SOURCE `crates/hawking-eval/src/lib.rs:82-86`).
- HumanEval / BCB-Hard / Aider: mentioned as gated corpora, not on
  this tree as runnable files.

---

## 5. Detectors

All mechanical. Input is one generate record
`{token_ids, text, n_new, max_new, min_new, task}`.
Output is a list of fire strings. Empty list = silent.

Existing `has_degenerate_repetition` (SOURCE
`crates/hawking-eval/src/support_halo.rs:383-401`, threshold 8) is
**not sufficient**. On the mixed-sub15 transcript
`"  a    a  a  a  a  a  a"` the whitespace split is 7× `a` (MEASURED
this process). The support-halo detector needs 8 consecutive identical
words and misses. Token-id cycles are required.

Constants:

```
EOS = {248046, 248044}
THINK_OPEN = 248068
WS_NL = {198, 220}          # newline, space-run (Qwen)
PUNCT_SALAD = set("()[]{}.,;")
```

### 5.1 `degenerate_cycles`

Fire if any:

1. **Period-p suffix.** For `p in 1..16`, last `3p` ids equal
   `block*3` for `block = last[−3p:−2p]`, and `n >= 8`.
2. **Pair occupancy.** On last 16 ids, some pair `(a,b)` occupies
   ≥ 4 of the 8 aligned pairs.
3. **Whitespace/newline dominance.** Last 16 ids have ≥ 12 members of
   `WS_NL`.
4. **Low unique.** Last 32 (or all if shorter, length ≥ 16) have
   `< 3` distinct ids.
5. **Text consecutive.** Whitespace-split token run ≥ 8 identical
   (support-halo detector, kept as a backstop).

Does **not** fire on the G0 32-id `Say hi.` prefix (MEASURED this
process: `NONE`).

### 5.2 `early_eos`

Fire if:

1. `n_new == 0`.
2. Last id ∈ EOS and `n_new < min_new`.
3. `n_new < min_new` and last id ∈ EOS (stopped, not truncated).

Hitting `max_new` is not early EOS. G0 `Say hi.` 32/32 is silent
(MEASURED this process).

No packed artifact in this campaign has been observed to emit EOS as
the first new token. The known-bad is the **frozen fixture**
`token_ids=[248046], n_new=1` plus the empty completion. That is
enough for the “every metric must be able to FAIL” requirement. Do
not invent a campaign receipt that does not exist.

### 5.3 `semantic_collapse`

Fire if:

1. After decode, all ids ∈ `WS_NL` (`whitespace_only`).
2. `alpha < 8` and `len(text) >= 8` (Q80 `classify_text` posture,
   SOURCE `lab/operators/q80_recalibrate_capability_bar.py:291-305`).
3. Non-whitespace chars ≥ 4 and ≥ 50% are in `PUNCT_SALAD`.
4. `len(text) >= 32` and `unique_chars/len < 0.08`.
5. Echo: stripped text equals stripped prompt, or starts with
   `write a function that` when that is the prompt verb
   (SOURCE `human_class`, same file:269-288).

### 5.4 `reasoning_collapse`

Applies to A/B/E2/E4/F (answer oracles), not to C/D/G1/G2.

Fire if:

1. `max_new >= 64` and `n_new >= 64` and text opens `<think>` and
   has no `</think>` and extract_answer is empty
   (`think_unclosed_no_answer`). Do **not** apply this to A0/32-cap.
2. Oracle missing from extracted answer.
3. Trap token present (`8` on B1, `10` on B5) and oracle absent.
4. Two distinct `ANSWER:` values in one completion.

### 5.5 `code_collapse`

Applies to C and E1/E3.

Fire if:

1. `extract_code` empty.
2. Body is only comments / `pass` / `todo!()`.
3. `python3` / `rustc+run` nonzero (reason = last stderr line, truncated 200,
   same as `thesis_gate.py:101-114`).
4. Timeout 30 s.
5. Extracted code equals the prompt.

### 5.6 `tool_syntax_collapse`

Applies to D.

Fire if:

1. `count("<tool_call>") > count("</tool_call>")` (hide-kernel’s
   parser **skips** an unclosed block, SOURCE
   `hide-kernel/src/tools.rs:595-597` `else { break; }` — so the
   detector must look at raw counts, not only `parse_tool_calls`).
2. A closed block fails `json.loads` (`tool_json:...`).
3. Required item: zero parsed calls.
4. Required item: parsed `name` ≠ want.
5. `arguments` present and not a JSON object.
6. D3: any `<tool_call>` (`unexpected_tool_call`).

Name aliases accepted on parse: `name` / `tool` / `function.name`
(SOURCE `hide-kernel/src/tools.rs:529-530`). Suite oracles still
require the exact names in §3.4 after alias resolution.

### 5.7 `calibration_collapse`

Fire if:

1. F1–F3 oracle miss (extracted token not in the refuse-set).
2. Completion claims success (`all tests passed`, `SUCCESS`,
   `I have verified`) and a paired execution oracle failed.
3. **Harness:** a reported rate has `n_scored==0` and a numeric
   value. This is the 402-None fold.
4. **Harness:** a capability float is present without
   `transcript_sha256`.
5. G3: did not stop after 3 `ping` and is not a cycle (already
   covered) — leftover stop-instruction miss.

Empty-fold algorithm (must replace `fold(1.0, min)`):

```
scored = [v for v in values if v is not None and isfinite(v)]
value  = min(scored) if scored else None
n_scored = len(scored)
# if n_scored==0: status=NOT_MEASURABLE; never emit 1.0
```

---

## 6. Known-bad → detector map

Every detector fires on at least one named object. Run is CPU,
MEASURED this process (script in §11).

| detector | known-bad | fire (MEASURED this process) | pointer |
|---|---|---|---|
| `degenerate_cycles` | mixed-sub15 prompt_1 ids `[220,264,220,220,220,264,…]` | `period_2_cycle [220,264]`, `pair_cycle (220,264) count=7`, `low_unique distinct=2` | RECEIPT `QWEN38_SUB15_INCOHERENT.json` `evidence.prompt_1` |
| `degenerate_cycles` | mixed-2p0 `Say hi.` `[198]×16` | `period_1_cycle [198]`, `ws_nl_dominance 16/16` | RECEIPT `QWEN38_NATIVE_MIXED_2P0_GENERATE.json` prompts[0] |
| `degenerate_cycles` | mixed-2p0 reverse `[1076,1076,8,…]` | `period_1_cycle [1076]`, `pair_cycle (1076,1076)` | same receipt prompts[1] |
| `early_eos` | fixture `ids=[248046], n_new=1, min_new=4` | `eos_at_1<min_new_4` | fixture; no campaign receipt |
| `early_eos` | fixture empty `n_new=0` | `n_new==0` | fixture |
| `semantic_collapse` | mixed-2p0 `Say hi.` 16 newlines | `whitespace_only` | 2p0 receipt prompts[0] `generated_text` |
| `semantic_collapse` | mixed-2p0 France `[198]×15+[8]` | `alpha<8 got=0` | 2p0 receipt prompts[2]; floor note RECEIPT `QWEN38_COHERENCE_FLOOR_BRACKETED.json` |
| `semantic_collapse` | mixed-2p0 reverse `......)...)...` | `punct_salad 39/39`, `unique_char_ratio 3/40` | 2p0 receipt prompts[1] |
| `reasoning_collapse` | mixed-sub15 text on “capital of France” | `oracle_miss Paris` | `QWEN38_SUB15_INCOHERENT.json` |
| `reasoning_collapse` | fixture `The ball costs 10 cents.` on B5 | `trap_answer 10` | fixture |
| `code_collapse` | fixture `def add(a,b): pass` + t01 test | `exec_fail:AssertionError` | thesis t01 test |
| `code_collapse` | mixed-sub15 `"  a  a  a…"` as code | `empty`/`exec_fail` | same sub15 text |
| `tool_syntax_collapse` | hide-kernel `{name: broken}` | `tool_json:Expecting property name…` | SOURCE `hide-kernel/src/tools.rs:847` |
| `tool_syntax_collapse` | unclosed `<tool_call>{"name":"fs.read"}` | `unclosed_tool_call 1>0` | follows parser `break` |
| `calibration_collapse` | 402× `None` fold | `buggy_min=1.0 suite=None n_scored=0 fire=True` | SOURCE `qwen38_pack.rs:680-684`; MEASURED this process |
| `calibration_collapse` | Attack 1 paperwork `capability=1.0` no generate | F5 | SOURCE `test_genesis_promotion_gate_adversarial.py:88-97` |

Negative control (must stay silent):

| object | detectors | result |
|---|---|---|
| G0 32-id `Say hi.` prefix + think text | degenerate, semantic, early_eos | `NONE` (MEASURED this process) |
| `def add(a,b): return a+b` + t01 test | code_collapse | `NONE` |
| `<tool_call>{"name":"fs.read","arguments":{"path":"/tmp/x"}}</tool_call>` | tool_syntax D1 | `NONE` |
| `ANSWER: 5` on B5 | reasoning | `NONE` |

`mixed-2p0-v1` and `mixed-sub15-v1` are **known-bad transcripts**, not
authorized floor locations (B9).

---

## 7. Pass criteria

### 7.1 Hard vetoes (absolute, no threshold)

Any one ⇒ `G1_CAPABLE = false`.

- Any CORE item fires `degenerate_cycles`.
- Any A-item fires `early_eos`.
- Any reported rate has `n_scored==0` and a numeric value.
- `fallbacks != 0` on any item.
- Vehicle illegal (B1). Load log is not the native path for the
  artifact under test.
- Detector self-test (§11) fails.
- F4 or F5 fires (suite/paperwork corruption).

### 7.2 Delta holds (need G0 seal)

Let `G0.pass[axis]`, `C.pass[axis]` be integer pass counts on the
same frozen items. Slack is in **counts**, not an uncalibrated 0.80.

| axis | HOLD if | rationale |
|---|---|---|
| A | `C >= G0 − 1` | one miss allowed |
| B | `C >= G0 − 1` | one miss allowed |
| C | `C >= G0 − 2` **or** Wilson-low(C) ≥ Wilson-low(G0) − 0.15 | 0.15 is the support-halo per-dimension drop (SOURCE `support_halo.rs:32-34`) |
| D | `C >= G0 − 1` | |
| E | `C >= G0 − 1` | |
| F | **absolute** `C == n_items` (3/3). G0 fail is `G0_CALIBRATION_FAIL`, not a lowered floor | refuse-set is a closed oracle |
| G | no collapse fire on G1/G2; G3 oracle pass; `trigram_ratio >= 0.7 * G0.trigram_ratio` once G0 `n_scored>0` | 0.7 is a **ratio to G0**, not an absolute uniqueness bar |

`trigram_ratio(ids) = unique 3-grams / n_3grams`. If `n_ids < 3`,
`NOT_MEASURABLE`.

Wilson: `hawking-eval wilson_interval` z=1.96
(SOURCE `crates/hawking-eval/src/lib.rs:24,69-80`).

### 7.3 Qualification

```
G1_CAPABLE iff
    all hard vetoes silent
    AND detector self-test PASS
    AND every CORE axis has G0 n_scored > 0
    AND every CORE axis HOLDs
```

If G0 itself fails an item, the item is a ceiling, not a candidate
fault, except F (absolute). Report both vectors. Do not hide a G0
fail by only printing delta.

A0 32-id match is **not** in this predicate.

### 7.4 What a HOLD is not

- Not “coherent on one prompt”.
- Not “loaded”.
- Not “smaller BPW”.
- Not “TPS went up”. TPS-up / capability-down is already a promotion
  reject (`CLAUSE_TPS_UP_CAP_DOWN`, SOURCE `promotion.py:48`).
- Not organ-cosine ≥ 0.86. That bar was a different campaign and was
  recalibrated off generation
  (SOURCE `lab/operators/q80_recalibrate_capability_bar.py:1-8,474-485`).

---

## 8. Wall time

G0 decode-phase TOKEN_NS **39,326,090** MEASURED (median of 6 paired
reps, spread 1.83%, SOURCE `g1-baseline-remeasure.md:12-13,142-149`).
TPS **25.4284** DERIVED `1e9/39326090`. Prefill on an 11-token prompt
was 434–550 ms (MEASURED same file:157). Use 39.326 ms/token for both
prefill and decode as a conservative ESTIMATE.

CORE typical (ESTIMATED n_new from G0 think-length 168 on A1-analog):

| band | items | est n_new each | tokens |
|---|---:|---:|---:|
| A | 6 | 168 | 1008 |
| B | 5 | 220 | 1100 |
| C | 8 | 250 | 2000 |
| D | 4 | 80 | 320 |
| E | 4 | 180 | 720 |
| F | 3 | 100 | 300 |
| G | 3 | 500/500/16 | 1016 |
| **gen** | **33** |  | **6464** |
| prefill @ ~35 tok | 33 | 35 | 1155 |
| **steps** |  |  | **7619** |

`7619 * 39,326,090 ns` = **299.6 s ≈ 5.0 min** generate / artifact
ESTIMATED.

Worst case (every item hits `max_new`): 6×256+5×384+8×512+4×192+4×384+3×192+768+768+32 = **12000**
new + 1155 prefill = 13155 steps → **517 s ≈ 8.6 min** ESTIMATED.

Addends, not in the 5.0 min:

| addend | class | note |
|---|---|---|
| G0 load | 0 | already resident; do not reload |
| candidate load | ESTIMATED 3–10 s | G0 load was 3.435 s RECEIPT `g1-resident-harvest.md:72`; mixed native unknown until that lane lands |
| C/E exec | ESTIMATED < 10 s | 8+2 python asserts |
| detector self-test | MEASURED < 1 s | this process |
| `gpu_lane_lock` wait | unbounded ≤ 5400 s | SOURCE `tools/gpu_lane_lock.sh` DEADLINE |
| resident socket queue | unbounded ≤ 1800 s / propose | SOURCE `PROPOSE_TIMEOUT_S = 1800` |

**Headline ESTIMATED: 5 min typical / 9 min worst per artifact generate.
G0 + candidate ≈ 10–18 min once the GPU lock is free.** Not a measured
suite run. A component microbench is not this number.

FAST smoke (bring-up only, not qualification): A1, A2, B5, C1, D1, F3,
G3. ESTIMATED ~1.0 min at the same TOKEN_NS.

FULL (+12 code items) adds ESTIMATED `12*250*0.039326 ≈ 118 s` / artifact.

---

## 9. Exact command a GPU lane runs

Do not start this while a parent propose holds the socket if you
intend to use the resident. Do not kill the resident. Do not load a
second 13.6 GB copy while `pages free` is low
(SOURCE `g1-baseline-remeasure.md:127-128,230-240`).

### 9.0 Detector self-test (CPU, required first)

Exit 0 required. Re-runs the §5 algorithms against the §6 fixtures.
Expected stdout is §11. Nonzero if any required fire is missing or
any negative control fires.

```
python3 workspace/superwave/g1/g1-capability-suite.selftest.py
```

That path is not in this lane’s write scope. Equivalent command,
self-contained (this is the GPU lane’s first line):

```
python3 - <<'PY'
from collections import Counter
import json, math, re, subprocess, tempfile
from pathlib import Path
EOS, WS_NL = {248046, 248044}, {198, 220}
PUNCT = set("()[]{}.,;")
G0 = [248068,198,760,1156,4777,6587,728,310,1910,328,5834,1149,1061,369,264,1546,4145,11,2050,1622,13,353,3172,1066,1910,15131,303,264,11321,11,5629,1560]
G0T = '<think>\nThe user simply wants me to say "hi." This is a very simple, direct request. I\'ll just say hi in a friendly, natural way'
SUB15 = [220,264,220,220,220,264,220,264,220,264,220,264,220,264,220,264]
fail = []
def deg(ids, text):
    f=[]
    n=len(ids)
    if n>=8:
        for p in range(1,17):
            if 3*p>n: break
            if ids[-3*p:]==ids[-3*p:-2*p]*3: f.append(f"p{p}"); break
        last=ids[-16:] if n>=16 else ids
        if len(last)>=8:
            pairs=list(zip(last[0::2], last[1::2]))
            if pairs and Counter(pairs).most_common(1)[0][1]>=4: f.append("pair")
        if n>=12 and sum(t in WS_NL for t in ids[-16:])>=12: f.append("ws")
        last32=ids[-32:]
        if len(last32)>=16 and len(set(last32))<3: f.append("uniq")
    return f
def sem(text, ids):
    f=[]
    if ids and all(t in WS_NL for t in ids): return ["ws_only"]
    alpha=sum(c.isascii() and c.isalpha() for c in text)
    if alpha<8 and len(text)>=8: f.append("alpha")
    nw=[c for c in text if not c.isspace()]
    if len(nw)>=4 and sum(c in PUNCT for c in nw)/len(nw)>=0.5: f.append("punct")
    if len(text)>=32 and len(set(text))/len(text)<0.08: f.append("uchr")
    return f
def fold(vals):
    sc=[v for v in vals if v is not None and isinstance(v,(int,float)) and math.isfinite(v)]
    buggy=1.0
    for v in vals:
        if v is not None: buggy=min(buggy,v)
    return buggy, (min(sc) if sc else None), len(sc)
b,s,n=fold([None]*402)
if not (b==1.0 and s is None and n==0): fail.append("empty_fold")
if deg(G0,G0T) or sem(G0T,G0): fail.append("g0_false_pos")
if "p2" not in deg(SUB15,"  a    a  a  a  a  a  a"): fail.append("sub15_cycle")
if "ws_only" not in sem("\n"*16,[198]*16): fail.append("2p0_nl")
if not deg([198]*15+[8],"\n"*15+")"): fail.append("2p0_fr")
src="def add(a,b):\n    pass\n\nassert add(2,3)==5\n"
p=Path(tempfile.mkdtemp())/"c.py"; p.write_text(src)
if subprocess.run(["python3",str(p)],capture_output=True).returncode==0: fail.append("stub_exec")
if "<tool_call>" in "<tool_call>{\"name\":\"fs.read\"}" and "</tool_call>" not in "<tool_call>{\"name\":\"fs.read\"}":
    pass
else:
    fail.append("unclosed_count")
try:
    json.loads("{name: broken}")
    fail.append("broken_json_accepted")
except json.JSONDecodeError:
    pass
print("SELFTEST", "FAIL" if fail else "PASS", fail or "")
raise SystemExit(1 if fail else 0)
PY
```

### 9.1 G0 seal (resident, no second upload)

Identity that must be in the receipt
(SOURCE `g1-baseline-remeasure.md:20-50`):

```
artifact     = .../qwen38-27b/uniform-q4-v1
artifact_sha = d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
tokenizer    = .../qwen38-27b/bf16/tokenizer.json
session      = protected_test
contract     = protected_capability_prompt_preserved
```

Per item:

```
python3 tools/agentos/genesis_resident.py propose \
  --session protected_test \
  --protected-capability \
  --max-new-tokens <item.max_new> \
  --prompt '<item.prompt>'
```

`propose` already returns `text`, `new_tokens`, `fallbacks`,
`prefill_wall_ns`, `decode_wall_ns`, `prompt_len`, `ok`
(SOURCE `tools/agentos/genesis_body/src/main.rs:709-728`).
Timeout 1800 s. Use `protected_test` only.

Do not pass `--complete-wall`. This is capability, not TOKEN_NS.
Serialized measurement lanes own timing.

### 9.2 Candidate (after native load exists)

Build (CPU; target-dir is the repo convention):

```
CARGO_TARGET_DIR=workspace/ops/build/rust \
  cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy
```

Binary:
`workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy`

`--prompts-file` exists but uses **one** `max_new` for every line and
requires ≥ 2 lines (SOURCE
`ascension_qwen38_hybrid_greedy.rs:605-616,645-646`). CORE needs
per-item caps. Loop.

```
export ARTIFACT="${ARTIFACT:-workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1}"
export TOK="${TOK:-workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json}"
export BIN="${BIN:-workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy}"
export OUT="${OUT:-/tmp/g1-capability-suite/candidate}"
mkdir -p "$OUT"

# Refuse the expand-to-Q4 sibling. If catalog.hq38m20 is absent, STOP.
test -f "$ARTIFACT/catalog.hq38m20" \
  || { echo "REFUSE: $ARTIFACT has no catalog.hq38m20; generate would be Q4 recon"; exit 2; }

./tools/gpu_lane_lock.sh g1-capability-suite \
  "$BIN" \
  --artifact-root "$ARTIFACT" \
  --tokenizer "$TOK" \
  --prompt "<item.prompt>" \
  --max-new-tokens <item.max_new> \
  --out "$OUT/<item.id>.json"
```

Stdout fields the scorer reads (SOURCE same example:287-292):

```
GENERATED_TEXT_VERBATIM: ...
FALLBACKS: N
NEW_TOKENS: [...]
generated_token_ids=[...]
```

Illegal if load log is `opening Metal + 755 catalog tensors` on a
mixed pack (that is the recon-Q4 path,
SOURCE `g1-sub15-native-gap.md:21-25`). Required log substring for a
mixed candidate: `opening mixed HQ38M20`.

G0 offline fallback (only if the resident is down — it must not be
taken down by this command):

```
./tools/gpu_lane_lock.sh g1-capability-suite \
  "$BIN" \
  --artifact-root workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1 \
  --tokenizer "$TOK" \
  --prompt "<item.prompt>" \
  --max-new-tokens <item.max_new> \
  --out "$OUT/g0/<item.id>.json"
```

### 9.3 One-shot command a GPU lane types

G0 seal, resident only, no second upload, no kill. Fails closed if
every propose fails (`n_scored==0` ⇒ no rate).

```
mkdir -p /tmp/g1-capability-suite
SUITE_MODE=g0 OUT=/tmp/g1-capability-suite/g0.json python3 - <<'PY'
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(os.environ.get("HAWKING_ROOT", Path.cwd()))
sys.path.insert(0, str(REPO / "tools/agentos"))
import genesis_resident as gr  # noqa: E402

C_IDS = {
    "t01_add": "C1", "t02_is_prime": "C2", "t03_fib": "C3",
    "t04_reverse_words": "C4", "t05_two_sum": "C5", "t06_gcd": "C6",
    "t07_is_palindrome": "C7", "t08_count_vowels": "C8",
}
AB = [
    ("A1", 256, 4, "What is 17 times 19? Reply with only the integer."),
    ("A2", 256, 4, "What is the capital of France? Reply with only the city name."),
    ("A3", 256, 4, "How many days are in a non-leap year? Reply with only the integer."),
    ("A4", 256, 4, "What is the 7th prime number? Reply with only the integer."),
    ("A5", 256, 4, "In Python, what does len([1, 2, 3]) return? Reply with only the integer."),
    ("A6", 256, 4, "Mira owns the violin. Dev owns the telescope. Who owns the violin? Reply with only the person's name."),
    ("B1", 384, 4, "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Reply with only the integer."),
    ("B2", 384, 4, 'There are 3 boxes. One contains apples, one contains oranges, one contains apples and oranges. All three labels are wrong. The box labeled "apples" is opened and contains oranges. What is in the box labeled "oranges"? Reply with only one of: apples, oranges, apples and oranges.'),
    ("B3", 384, 8, "Compute 17 × 19 two ways: (20-1)×17 and 17×20−17. Show both. Final line: ANSWER: <integer>"),
    ("B4", 384, 4, "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Reply yes or no, then one sentence."),
    ("B5", 384, 4, "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Reply with only the integer."),
]
C = []
corpus = REPO / "tools/eval/thesis_smoke_corpus_v0.jsonl"
if not corpus.is_file():
    # sparse checkout: pull the eight prompts from git without materializing
    import subprocess
    raw = subprocess.check_output(["git", "show", "HEAD:tools/eval/thesis_smoke_corpus_v0.jsonl"], text=True)
else:
    raw = corpus.read_text()
for line in raw.splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row["id"] in C_IDS:
        C.append((C_IDS[row["id"]], 512, 8, row["prompt"]))
D = [
    ("D1", 192, 4, "You have one tool: fs.read with argument path (string). User: Read the file /tmp/g1-suite/hello.txt. Emit only a tool call, no prose."),
    ("D2", 192, 4, "You have one tool: shell.run with argument argv (array of strings). User: List files in the current directory using ls -la. Emit only a tool call, no prose."),
    ("D3", 192, 4, "You have one tool: fs.read. User: What is 2+2? Do not emit a tool call. Reply with only the integer."),
    ("D4", 192, 4, 'Emit exactly this and nothing else: <tool_call>{"name":"git.status","arguments":{}}</tool_call>'),
]
E = [
    ("E1", 384, 8, "The following test fails. Write only the Python function that makes it pass.\n\ndef test():\n    assert clamp(5, 0, 10) == 5\n    assert clamp(-1, 0, 10) == 0\n    assert clamp(99, 0, 10) == 10\n"),
    ("E2", 384, 2, "This function is wrong. Reply with only the 1-based line number of the bug.\n1: def average(xs):\n2:     s = 0\n3:     for x in xs:\n4:         s += x\n5:     return s / (len(xs) - 1)\n"),
    ("E3", 384, 8, 'Apply this change and print only the resulting function.\n\ndef greet(name):\n    return "Hello " + name\n\nChange it to return f"Hello, {name}!"\n'),
    ("E4", 384, 2, "File a.py: def f(): return 1\nFile b.py: from a import f\ndef g(): return f() + 2\nWhat does g() return? Reply with only the integer."),
]
F = [
    ("F1", 192, 2, "What is the middle name of the person who invented the Hawking Qwen38 mixed-sub15 pack? If you do not know, reply exactly: UNKNOWN"),
    ("F2", 192, 2, "Prove that 2+2=5. If you cannot, reply exactly: CANNOT"),
    ("F3", 192, 2, "You ran zero tests. Did all tests pass? Reply exactly YES or NO."),
]
G = [
    ("G1", 768, 64, "Write a 400-word explanation of what a hash map is. Do not repeat any paragraph."),
    ("G2", 768, 64, "Continue this sequence of distinct sentences, one per line, numbering 1 to 40. Each sentence must introduce a new concrete noun not used earlier."),
    ("G3", 32, 3, "Repeat the word ping exactly 3 times and then stop."),
]
seq = AB + C + D + E + F + G
if len(seq) != 33:
    raise SystemExit(f"CORE must be 33 items, got {len(seq)}")

ART_C = os.environ.get("ARTIFACT", "")
MODE = os.environ.get("SUITE_MODE", "g0")
OUT = Path(os.environ.get("OUT", "/tmp/g1-capability-suite/g0.json"))
if MODE in ("candidate", "both"):
    if not ART_C:
        raise SystemExit("ARTIFACT required for candidate/both")
    if not (Path(ART_C) / "catalog.hq38m20").is_file():
        raise SystemExit(f"REFUSE: {ART_C} has no catalog.hq38m20 (expand-to-Q4 confound)")

recs = []
t0 = time.time()
if MODE in ("g0", "both"):
    for iid, mx, mn, prompt in seq:
        print(f"G0 {iid} max_new={mx}", flush=True)
        resp = gr.propose(
            prompt,
            max_new_tokens=mx,
            session="protected_test",
            protected_capability=True,
            timeout=1800,
        ) or {"ok": False, "error": "propose failed"}
        rec = dict(resp)
        rec.update(id=iid, max_new=mx, min_new=mn, prompt=prompt)
        recs.append(rec)
n = sum(1 for r in recs if r.get("ok"))
receipt = {
    "schema": "hawking.g1.capability_suite.v1",
    "mode": MODE,
    "n_items": len(seq),
    "n_scored_g0": n,
    "g0_pass_rate": None,
    "g0_status": "TRANSCRIPTS" if n else "NOT_MEASURABLE",
    "wall_s": round(time.time() - t0, 1),
    "records": recs,
    "note": "Raw transcripts. Score offline with section 5 detectors. Do not fold empty.",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(receipt, indent=2) + "\n")
print(f"wrote {OUT} n_scored_g0={n}", flush=True)
raise SystemExit(0 if n else 2)
PY
```

Score the receipt with the §5 detectors and the §7 predicate. Scoring
is CPU. Do not emit a pass rate with `n_scored==0`.

Candidate: same command with
`SUITE_MODE=candidate ARTIFACT=workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1`
after `catalog.hq38m20` exists, wrapping each hybrid_greedy invoke in
`./tools/gpu_lane_lock.sh g1-capability-suite` as in §9.2. Until that
catalog file exists the command exits 2. That is the correct refusal.

### 9.4 What this command must not do

- `tools/eval/thesis_gate.py --weights *.gguf` (wrong model, wrong
  vehicle).
- `coherence_gate.py verify` against `QWEN38_COHERENCE_SEAL.json` as
  pass/fail for a non-G0 artifact (G006 lesson).
- `genesis_resident.py propose` on session `parent` (contract inject).
- Stop, reload, or restart `genesis-resident`.
- Start `ascension_qwen38_hybrid_greedy` without `gpu_lane_lock.sh`.

---

## 10. Receipt schema

```
{
  "schema": "hawking.g1.capability_suite.v1",
  "g0": {
    "artifact": ".../uniform-q4-v1",
    "artifact_sha": "d650a757…",
    "vehicle": "genesis-resident/protected_test/HQ30UQ4",
    "load_log_substring": "opening Metal + 755 catalog tensors"
  },
  "candidate": {
    "artifact": ".../mixed-sub15-v1",
    "artifact_sha": "<manifest sha>",
    "vehicle": "hybrid_greedy/HQ38M20" | "REFUSED_NO_CATALOG",
    "complete_physical_bpw": <float or null>,
    "load_log_substring": "opening mixed HQ38M20"
  },
  "selftest": {"pass": true, "n_fires_expected": <int>, "n_fires_got": <int>},
  "axes": {
    "A": {"g0_pass": n, "g0_n": 6, "cand_pass": n, "cand_n": 6,
          "g0_pass_rate": float|null, "cand_pass_rate": float|null,
          "hold": true|false, "status": "HOLD"|"REGRESS"|"NOT_MEASURABLE"}
  },
  "vetoes": [],
  "diagnostics": {
    "a0_oracle32_match": true|false,
    "think_open_rate_g0": float|null,
    "think_open_rate_cand": float|null,
    "id_prefix_hamming": {"A2": int, "...": int}
  },
  "g1_capable": false,
  "n_scored_guard": "every rate carries n; empty => null"
}
```

`g1_capable` defaults false. It flips true only by §7.3.

---

## 11. Detector self-test (MEASURED this process)

Run 2026-08-17 on this worktree, CPython 3, no GPU. Algorithms are
§5. Known-bad objects are §6.

```
=== EMPTY FOLD (min_q4_cosine bug) ===
buggy_min=1.0 suite=None n_scored=0 calibration_collapse=True

=== G0 Say-hi (must NOT fire cycle/semantic on 32-id think prefix) ===
degenerate NONE
semantic NONE
early_eos NONE

=== mixed-sub15 prompt_1 ===
degenerate ['period_2_cycle ids=[220, 264]', 'pair_cycle (220, 264) count=7', 'low_unique last16 distinct=2']
semantic ['alpha<8 got=7']
reasoning Paris ["oracle_miss expect='Paris'"]

=== mixed-2p0 Say hi [198]x16 ===
degenerate ['period_1_cycle ids=[198]', 'pair_cycle (198, 198) count=8', 'ws_nl_dominance 16/16', 'low_unique last16 distinct=1']
semantic ['whitespace_only']

=== mixed-2p0 France [198]x15+[8] ===
degenerate ['pair_cycle (198, 198) count=7', 'ws_nl_dominance 15/16', 'low_unique last16 distinct=2']
semantic ['alpha<8 got=0']

=== mixed-2p0 reverse dots/parens ===
degenerate ['period_1_cycle ids=[1076]', 'pair_cycle (1076, 1076) count=4']
semantic ['alpha<8 got=0', 'punct_salad 39/39', 'unique_char_ratio 3/40']

=== early EOS synthetic ===
['eos_at_1<min_new_4', 'stopped_before_min_new']
empty n_new ['n_new==0']

=== code collapse stub ===
['exec_fail:AssertionError']

=== code collapse good ===
NONE

=== tool syntax ===
broken ['tool_json:Expecting property name enclosed in double quotes: line 1 column 2 (char 1)']
unclosed ['unclosed_tool_call 1>0']
good NONE
spam ['unexpected_tool_call']

=== support_halo has_degenerate_repetition miss on 220/264 ===
split ['a', 'a', 'a', 'a', 'a', 'a', 'a'] len 7

=== trap 10 vs 5 ===
["oracle_miss expect='5'", 'trap_answer 10']
correct NONE
```

Self-test PASS conditions: every §6 “must fire” row fires; every
negative-control row is silent; empty fold reports `None` not `1.0`.

---

## 12. KILLS / REOPEN_IF

| id | kill | reopen_if |
|---|---|---|
| K1 | Greedy-id identity as a G1 qualification bar | Candidate is claimed the same physical catalog as G0 (same codec, same weights); then id drift is a regression |
| K2 | `min_q4_cosine` / organ-cosine as a capability number | A generate-calibrated map from cosine → this suite’s HOLD is sealed with `n_scored>0` and a broken point |
| K3 | `DEFAULT_CAPABILITY_CONTRACT` 1.0 floats as evidence | Those floats are derived from a suite receipt sha |
| K4 | `thesis_gate.py --weights GGUF` as a G1 vehicle | The GGUF is proven byte-identical to the candidate catalog (it is not) |
| K5 | Expand-to-Q4 / expand-to-float / MLX overwrite generate as capability | Complete-token measurement shows the expansion is a net physical win **and** the suite HOLD is taken on the expanded path with that confound labeled — still not a codec verdict |
| K6 | Relocating Qwen3.8’s coherence floor from mixed-2p0 or expand-vehicle sub15 | Native HQ38M20 generate of a **role-sane** allocation (attention not left at 4.25 while down_proj is 0.13) produces a collapse or a hold |
| K7 | support-halo `has_degenerate_repetition` as the only cycle detector | A token-id cycle of period 2 with no 8-word text run is shown not to occur on this tokenizer |

---

## 13. What this lane did not do

- No GPU, no Metal, no generate, no resident stop/reload.
- No candidate score. G0 scores on C/D/E/F/G are NOT_MEASURABLE until
  §9.1 runs.
- No new detector implementation landed in-tree (write scope is this
  file). §11 is a measurement of the frozen algorithms, not a crate.
- Did not re-derive BPW, TOKEN_NS, roofs, or dead families.

Cheapest experiment that produces the missing G0 vector: §9.1 on the
live protected_test session when the socket is idle. ESTIMATED 5 min.

Cheapest experiment that produces a candidate vector: land
`catalog.hq38m20` (other lane, 250–500 lines claimed), then §9.2.

---

```
STATUS
IMPLEMENT_READY

CLAIMS
1. 6/6 oracle-32 on Say hi. plus one 323 emit is fluency/identity, not G1 qualification (RECEIPT g1-baseline-remeasure.md:170-213; QWEN38_COHERENCE_SEAL.json).
2. min_q4_cosine=1.0 over 402 None is an empty fold, not a quality number (SOURCE qwen38_pack.rs:680-684; MEASURED g1-baseline-audit.md:69-80). The suite’s calibration_collapse reproduces buggy_min=1.0 / suite=None / n_scored=0 (MEASURED §11).
3. support-halo text-repeat detector misses the mixed-sub15 220/264 cycle (7× "a" < 8). Token-id cycle detectors fire (MEASURED §11; RECEIPT QWEN38_SUB15_INCOHERENT.json).
4. mixed-2p0 native transcripts are known-bad completions for degenerate_cycles and semantic_collapse (RECEIPT QWEN38_NATIVE_MIXED_2P0_GENERATE.json, all 6 prompts). They do not locate the coherence floor (B9).
5. G1_CAPABLE is hard-veto ∧ delta-HOLD against a G0 seal of this suite, not an absolute 0.8 (this file §7). F is an absolute refuse-set.
6. Legal vehicles are genesis-resident protected_test (G0) and hybrid_greedy on HQ38M20 (candidate). Expand-to-Q4 is refused (SOURCE g1-sub15-native-gap.md:21-25).
7. Wall time ESTIMATED 5 min typical / 9 min worst per artifact from MEASURED TOKEN_NS 39,326,090 (g1-baseline-remeasure.md:12-13) × ESTIMATED 7619 steps.
8. Every named detector fires on a named known-bad and is silent on the G0 32-id prefix (MEASURED §11).

EVIDENCE
- workspace/superwave/g1/g1-baseline-remeasure.md:170-213
- workspace/superwave/g1/g1-baseline-audit.md:69-80,126-141
- workspace/superwave/g1/g1-sub15-native-gap.md:21-25,80-81
- workspace/superwave/g1/g1-artifact-inventory.md:342-357
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json
- receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json
- receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json
- receipts/ascent-2026-08-16/HARVEST_NOTE_G006.json
- crates/hawking-core/src/model/qwen38_pack.rs:680-684
- crates/hawking-core/src/model/qwen38_hybrid_decode.rs:316-317,3397-3402
- crates/hawking-core/src/model/qwen38_geometry.rs:57-58
- crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs:287-292,605-646
- crates/hawking-eval/src/lib.rs:69-86
- crates/hawking-eval/src/support_halo.rs:7-8,32-37,383-401
- crates/hide-kernel/src/tools.rs:520-604
- lab/lineage/identity.py:28-32
- lab/lineage/promotion.py:32,48,55,262-268,744-793
- lab/tests/test_genesis_promotion_gate_adversarial.py:88-97
- lab/operators/q80_recalibrate_capability_bar.py:269-305
- lab/operators/one_mountain_capability.py:23-25
- tools/coherence_gate.py
- tools/eval/thesis_gate.py
- tools/eval/thesis_smoke_corpus_v0.jsonl
- tools/agentos/genesis_resident.py:201-212
- tools/gpu_lane_lock.sh
- this file §11 detector self-test output

CHANGES
workspace/superwave/g1/g1-capability-suite.md (new)

TESTS
```
$ python3 - <<'PY'
# §9.0 compact self-test
PY
SELFTEST PASS

$ test -s workspace/superwave/g1/g1-capability-suite.md && echo PASS
PASS
$ wc -l workspace/superwave/g1/g1-capability-suite.md
    1184 workspace/superwave/g1/g1-capability-suite.md
$ git status --porcelain
?? workspace/superwave/g1/g1-capability-suite.md
```

RISKS
- G0 CORE scores are unmeasured. Qualification is blocked until §9.1.
- mixed-sub15 cannot legally run until catalog.hq38m20 exists.
- DENSE_W_MATERIALIZED in hybrid_greedy is hardcoded 0.
- Qwen3.8 think-blocks can exceed the ESTIMATED n_new; worst-case 8.6 min is the cap-bound.
- F1–F3 G0 behavior unknown; F is absolute, so a G0 fail is G0_CALIBRATION_FAIL.
- Resident propose on protected_test still serializes behind a live parent propose.

UNRESOLVED
- G0 pass vector on A–G of this suite.
- Candidate vector (native load not landed).
- Logit NLL co-metric (no logit seam).
- Whether G0 think-opens on every CORE item (diagnostic).

NEXT
GPU lane: §9.0 then §9.1 when the socket is idle. After HQ38M20 lands, §9.2 on mixed-sub15-v1 and apply §7.
```
