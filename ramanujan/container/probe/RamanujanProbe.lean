import Mathlib.Tactic.NormNum

/-!
# Q0 clean-container probe

The lock's machine-check: 2 + 2 = 4 by `norm_num` against the pinned Mathlib.
A proof that only checks on the host that produced it is a local fact, not a mathematical one.
-/

theorem two_plus_two : (2 : Nat) + 2 = 4 := by norm_num
