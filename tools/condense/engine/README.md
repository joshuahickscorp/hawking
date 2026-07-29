# Campaign / experiment engine (lane A1)

One authority for campaign lifecycle. Controllers do not multiply.

## Surfaces

| Module | Role |
|--------|------|
| `spec.py` | `ExperimentSpec` schema — campaigns are data |
| `state_machine.py` | Typed FSM for precheck → … → report |
| `scheduler.py` | Resume-safe work planning |
| `governor.py` | Disk / resource floors |
| `lease.py` | Exclusive controller lease |
| `checkpoint.py` | Hash-chained events + sealed checkpoints |
| `receipts.py` | Canonical sealed receipts |
| `seal_integrity.py` | Reseal/substitution rejection, launcher-node safety, no-subprocess preflight |
| `runtime.py` | `run_campaign` / CLI |

## Specs

Declarative JSON under `specs/` for:

- `glm52` (live contract readers remain as modules)
- `kimi_k26`, `qwen`, `gptoss`, `deepseek_v4`, `second_light`, `gravity_frontier`

Each keeps: steps, receipt pointer, fixture, reproduction command, reopen conditions.

## Run

```bash
python3.12 -m tools.condense.engine tools/condense/engine/specs/glm52.json \
  --work-dir reports/condense/engine/glm52
```

Lane F1 made retirement real: superseded controller bodies were **deleted** (not
parked under `archive/`). Each campaign keeps this engine surface plus its
specification, fixture, receipt, reproduction command, and reopen condition.
Git history and tag `pre-controller-retirement-20260728` hold the old bodies.

Modules still imported by tests or live readers were restored to
`tools/condense/<module>.py` (relocation, not condensation). See
`docs/ARCHIVE_INDEX_2.md` §F1 and `engine/fixtures/f1_retirement_receipt.json`.

## H1 retirement

Lane H1 deleted remaining campaign controller shells under `tools/condense/` and expressed
them as specs (including new `succession`, `eco`, `mechanics`, `tg` families). Kimi K2.6
controllers are deleted as DOES-NOT-FIT; sealing invariants remain in `seal_integrity.py`.
Live GLM52 readers (`glm52_parity`, `glm52_contract`, `glm52_source_fetch`,
`glm52_teacher_capture`, `glm52_xet_autotune`) and algorithmic libraries stay as modules.

