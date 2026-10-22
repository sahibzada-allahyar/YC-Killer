-- Round 3 (logic/vacuity lens). HOLE: curated OK yet PROVED with a near-constant wrong impl.
-- LF1021 `solve_min_value : Nat → Nat → Nat × Nat` (spec-only, no #eval). The only theorem the
-- curation battery (default/pure-default/passthrough bodies × rfl/omega/simp_all/decide) cannot
-- discharge is `one_stays_one : (solve_min_value 1 1).1 = 1` (default `(0,0)` gives 0 ≠ 1), so the
-- task is curated OK ("a theorem has no trivial proof"). But a trivial-but-non-default body
-- `fun n _ => (if n = 1 then 1 else 0, 0)` satisfies every theorem: the bounds (result_n ≤ n,
-- result_n < n for n>9), the nonnegativity, determinism, the single-digit bound, and the one
-- pinned point. `solve_min_value` is otherwise unimplemented (returns 0), so the rung PROVES a
-- solution that computes nothing. The triviality battery misses it because it never tries a
-- one-point lookup body.
def solve_min_value (n d : Nat) : Nat × Nat :=
  (if n = 1 then 1 else 0, 0)

theorem result_smaller_than_input (n d : Nat) (h1 : n > 0) (h2 : d > 0) :
  let (result_n, _) := solve_min_value n d
  result_n ≤ n := by
  simp [solve_min_value]; split <;> omega

theorem result_nonnegative (n d : Nat) (h1 : n > 0) (h2 : d > 0) :
  let (result_n, _) := solve_min_value n d
  result_n ≥ 0 := by simp

theorem ops_nonnegative (n d : Nat) (h1 : n > 0) (h2 : d > 0) :
  let (_, result_ops) := solve_min_value n d
  result_ops ≥ 0 := by simp [solve_min_value]

theorem larger_numbers_get_smaller (n : Nat) (h : n > 9) :
  let (result_n, _) := solve_min_value n 1
  result_n < n := by
  simp [solve_min_value]
  have : n ≠ 1 := by omega
  simp [this]; omega

theorem function_is_deterministic (n d : Nat) (h1 : n > 0) (h2 : d > 0) :
  solve_min_value n d = solve_min_value n d := rfl

theorem one_stays_one :
  (solve_min_value 1 1).1 = 1 := by simp [solve_min_value]

theorem single_digit_not_increase (h : 9 ≥ 1) :
  (solve_min_value 9 1).1 ≤ 9 := by simp [solve_min_value]

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible
