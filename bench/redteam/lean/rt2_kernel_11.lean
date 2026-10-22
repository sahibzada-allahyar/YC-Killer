def summation (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | k + 1 => (k + 1) + summation k

theorem summation_formula (n : Nat) (h : n > 0) : 
  summation n = n * (n + 1) / 2 := by
  induction n with
  | zero => omega
  | succ k ih =>
    simp only [summation]
    rcases Nat.eq_zero_or_pos k with hk | hk
    · subst hk; rfl
    · rw [ih hk]
      have : k * (k + 1) % 2 = 0 := by
        rcases Nat.even_or_odd k with ⟨m,hm⟩ | ⟨m,hm⟩ <;> subst hm <;> ring_nf <;> omega
      omega

theorem summation_increasing {a b : Nat} (ha : a > 0) (hb : b > 0) :
  summation (a + b) > summation a := by
  induction b with
  | zero => omega
  | succ k ih =>
    simp only [summation]
    rcases Nat.eq_zero_or_pos k with hk | hk
    · subst hk; simp [summation]; omega
    · have := ih hk
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
