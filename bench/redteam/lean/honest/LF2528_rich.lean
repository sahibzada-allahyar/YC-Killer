-- Honest solution that uses every allowed convenience: header comment, helper structure with `deriving Repr`,
-- private and partial helpers, well-founded recursion, a @[simp] lemma, `set_option ... in` and `open ... in`
-- right before frozen headers, and a namespace that is closed again.
def last_chair (n : Nat) : Nat :=
  chairLoop n
where
  chairLoop : Nat → Nat
    | 0 => 0
    | k + 1 => k

structure Seat where
  idx : Nat
  deriving Repr, DecidableEq

instance : Inhabited Seat := ⟨⟨0⟩⟩

private def seatOf (n : Nat) : Seat := ⟨last_chair n⟩

partial def scan (n : Nat) : Nat := if n = 0 then 0 else scan (n - 1)

def sumTo (n : Nat) : Nat :=
  if h : n = 0 then 0 else n + sumTo (n - 1)
termination_by n
decreasing_by omega

namespace Helpers
@[simp] theorem last_chair_eq (n : Nat) : last_chair n = n - 1 := by
  cases n <;> rfl
end Helpers

set_option maxHeartbeats 400000 in
theorem last_chair_range {n : Nat} (h : n ≥ 3) :
  1 ≤ last_chair n ∧ last_chair n ≤ n := by
  simp only [Helpers.last_chair_eq]
  omega

open Nat in
/-- The value lemma, stated by the task. -/
theorem last_chair_value {n : Nat} (h : n ≥ 3) :
  last_chair n = n - 1 := by
  exact Helpers.last_chair_eq n

-- Apps difficulty: introductory
-- Assurance level: guarded

/--
info: 9
-/
#guard_msgs in
#eval last_chair 10

/--
info: 4
-/
#guard_msgs in
#eval last_chair 5

/--
info: 2
-/
#guard_msgs in
#eval last_chair 3
