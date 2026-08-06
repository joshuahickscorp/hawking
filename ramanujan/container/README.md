# Q0 clean-container proof replay

A Tier 3 claim must be re-provable from its capsule in a container built only from
`RAMANUJAN_ENVIRONMENT_LOCK.json`, with **no network** and **no host state**.

A proof that only checks on the machine that produced it is a local fact, not a mathematical one.

## Pins

See `pins.json`. Everything is an immutable identifier:

| Component | Pin |
|-----------|-----|
| Base image | `ubuntu@sha256:4fbb8e6a8395…` |
| elan | 4.2.3 + sha256 of linux aarch64 tarball |
| Lean (Mathlib checks) | 4.33.0-rc1 commit `62eed1db4d67327ec8120be05f1a1b0847d74561` |
| Lean (lock host default) | 4.32.1 commit `f054605aea4b840552cca2e725580bffd1e1b704` |
| Mathlib | commit `2ec0166b31100827cd34bacca4d3b9ea3da9d618` |
| z3 | 4.16.0 + sha256 |
| cadical | 3.0.1 (git `c60730422e75…`) + sha256 |
| PARI/GP | 2.17.4 + sha256 |

Mathlib at the pinned commit requires Lean 4.33.0-rc1. Capsule replay uses that toolchain.
The lock's host default Lean 4.32.1 is installed for pin parity but is not what Mathlib loads.

## Build

```bash
# On this host the OrbStack docker context hangs; Colima works:
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
./build.sh
```

Build may use the network to fetch pinned release artifacts and the Mathlib cache.
The resulting image is self-contained.

## Replay (the contract)

```bash
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
./replay_capsule.sh capsules/two_plus_two.capsule.json
# exit 0 = machine-check OK; non-zero = fail
```

`replay_capsule.sh` always runs with `--network=none` and does not mount host Lean/Mathlib.

## Capsule format

`hawking.ramanujan.proof_capsule.v1` JSON:

```json
{
  "schema": "hawking.ramanujan.proof_capsule.v1",
  "id": "two_plus_two",
  "proof_lean": "import Mathlib.Tactic.NormNum\n\ntheorem two_plus_two : (2 : Nat) + 2 = 4 := by norm_num\n",
  "pins": {
    "mathlib_commit": "2ec0166b31100827cd34bacca4d3b9ea3da9d618",
    "lean_toolchain": "leanprover/lean4:v4.33.0-rc1",
    "lean_commit": "62eed1db4d67327ec8120be05f1a1b0847d74561"
  }
}
```

A raw `.lean` file is also accepted.
