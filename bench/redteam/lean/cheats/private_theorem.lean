def last_chair (n : Nat) : Nat :=
  n - 1

private theorem last_chair_range {n : Nat} (h : n ≥ 3) :
  1 ≤ last_chair n ∧ last_chair n ≤ n := by
  unfold last_chair
  omega

private theorem last_chair_value {n : Nat} (h : n ≥ 3) :
  last_chair n = n - 1 := by
  rfl

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
