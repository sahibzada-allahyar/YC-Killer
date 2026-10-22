def calculate_maintenance_due (n : Nat) (payments : List String) : Nat :=
  if payments.any (· = "0") then 2000 + 100*n else 0

theorem empty_payments (n : Nat) :
  calculate_maintenance_due n [] = 0 :=
  rfl

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible

/--
info: 0
-/
#guard_msgs in
#eval calculate_maintenance_due 2 ["1", "1"]

/--
info: 2200
-/
#guard_msgs in
#eval calculate_maintenance_due 2 ["0", "0"]

/--
info: 2300
-/
#guard_msgs in
#eval calculate_maintenance_due 3 ["0", "1", "0"]
