def remainder (dividend divisor : Nat) : Nat := dividend % divisor

theorem remainder_properties {dividend divisor : Nat} (h : divisor > 0) : 
  let r := remainder dividend divisor
  (r ≥ 0) ∧ 
  (r < divisor) ∧ 
  (dividend = (dividend / divisor) * divisor + r)
  := by
  simp only [remainder]
  refine ⟨Nat.zero_le _, Nat.mod_lt _ h, ?_⟩
  have h2 := Nat.div_add_mod dividend divisor
  have h3 : dividend / divisor * divisor = divisor * (dividend / divisor) := Nat.mul_comm _ _
  omega

theorem remainder_matches_modulo {dividend divisor : Nat} (h : divisor > 0) :
  remainder dividend divisor = dividend % divisor := rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 1
-/
#guard_msgs in
#eval remainder 3 2

/--
info: 1
-/
#guard_msgs in
#eval remainder 19 2

/--
info: 2
-/
#guard_msgs in
#eval remainder 27 5
