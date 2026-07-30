import Lake
open Lake DSL

package «ramanujan_probe» where
  -- Minimal project: one Mathlib-dependent probe theorem for the clean-container contract.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "2ec0166b31100827cd34bacca4d3b9ea3da9d618"

@[default_target]
lean_lib «RamanujanProbe» where
  globs := #[.submodules `RamanujanProbe]
