def summation (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem summation_formula (n : Nat) (h : n > 0) : 
  summation n = n * (n + 1) / 2 := 
  rfl

theorem summation_increasing {a b : Nat} (ha : a > 0) (hb : b > 0) :
  summation (a + b) > summation a :=
  by
  simp only [summation]
  have : a * (a+1) < (a+b) * (a+b+1) := by nlinarith
  omega

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 1
-/
#guard_msgs in
#eval summation 1

/--
info: 36
-/
#guard_msgs in
#eval summation 8

/--
info: 253
-/
#guard_msgs in
#eval summation 22
