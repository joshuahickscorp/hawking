# Hawking Condensation Receipt Harness

> **The first artifact** (`docs/plans/studio_maximization_2026_06_27.md` §20.9 / §20.13): the
> machine-verifiable proof format that makes every Hawking run rerunnable. *Hawking does not ask
> to be believed. It asks to be rerun.* Built pre-Studio on the M3 Pro 18 GB — pure code + JSON,
> zero downloads, zero large compute.

## Layout

```
receipts/
  schema/condensation_receipt.schema.json   # the §20.3 v0.2 JSON Schema (strict)
  official/<model>/<recipe>__<commit>.json   # Hawking's own runs (R3+ wins, baselines)
  third_party/<machine_class>/...            # external-Mac R4 drops (self-verifying)
  failures/<FAIL-NNN>.json                   # negative receipts (gate=fail)
  prompt_suite_v1.txt                        # frozen eval cassette (§20.11)
  prompt_suite_v1.sha256                     # its sha256 — the prompt_suite_hash
  README.md                                  # this file
tools/condense/
  receipt_verify.py                          # schema + 8 invalidation rules (§20.3)
  emit_receipt.py                            # emit a receipt from on-disk ladder data
BASELINES.md  WATCHLIST.md  FAILURES.md       # the §20.11 stubs
scaffolding/
  requirements.freeze.txt                    # pinned Python env (pip freeze)
  toolchain.versions.txt                     # cargo / rustc / python / uname
  git_commit.txt                             # HEAD at scaffolding time
```

## Use

```bash
# self-test the verifier (a valid + an invalid fixture)
python3.12 tools/condense/receipt_verify.py --self-test

# validate one or more receipts
python3.12 tools/condense/receipt_verify.py receipts/official/*.json

# (re)emit the real 0.5B tq3 baseline receipt from on-disk ledger data
python3.12 tools/condense/emit_receipt.py
```

## The eight invalidation rules (§20.3)

A receipt that trips any rule gets `quality_gate=invalid`, the run **does not count**, and the
reason is published. `receipt_verify.py` enforces all eight in code plus the §20.6 master rule
**no public win below R3**:

1. `effective_bpw` missing/≤0 or only `nominal_bpw` reported.
2. quality from a single window (`multiwindow_n < 4`) or worst-window hidden.
3. PPL passes but `kl_parent_condensed` exceeds the warn band (ppl-theater).
4. no `source_sha256` / `artifact_sha256` (artifact not identifiable).
5. no `commands` or no `hawking_commit` (not reproducible).
6. mislabelled `claim_type` vs the Q4 baseline behavior (density vs cliff).
7. MPS headline without a CPU-bf16 confirmation.
8. a best-effort baseline used to back a public win.

## Reproduction levels (§20.6)

`R0` private · `R1` author-rerunnable · `R2` artifact identified+measured · `R3` one-command
same-machine-class repro (**min bar for a public win**) · `R4` third-party Mac (**trust moat**) ·
`R5` the format itself cited externally.

## Pinned environment (scaffolding/)

Captured 2026-06-27 on the M3 Pro 18 GB (read-only; nothing installed):

- `cargo 1.94.1 (Homebrew)`, `rustc 1.94.1`
- `Python 3.12.6` with `jsonschema 4.24.0`, `torch 2.6.0`, `transformers 5.6.2`,
  `safetensors 0.7.0`, `mlx 0.31.2`, `numpy 2.2.6` (full set in `scaffolding/requirements.freeze.txt`).
- HEAD: see `scaffolding/git_commit.txt`.

The Studio installs the byte-identical set from `scaffolding/requirements.freeze.txt`.
