def P (bars boxes : Nat) : Prop := bars = 0 ∧ boxes = 1
instance : DecidablePred (fun p : Nat × Nat => P p.1 p.2) := fun p => inferInstanceAs (Decidable (p.1 = 0 ∧ p.2 = 1))
def start_smoking (bars : Nat) (boxes : Nat) : Nat :=
  if bars = 0 ∧ boxes = 1 then 22 else if bars = 1 ∧ boxes = 0 then 224 else 247

theorem start_smoking_nonnegative (bars boxes : Nat) :
  start_smoking bars boxes ≥ 0 := Nat.zero_le _

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 22
-/
#guard_msgs in
#eval start_smoking 0 1

/--
info: 224
-/
#guard_msgs in
#eval start_smoking 1 0

/--
info: 247
-/
#guard_msgs in
#eval start_smoking 1 1
