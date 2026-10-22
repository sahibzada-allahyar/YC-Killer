def calculate_maintenance_due (n : Nat) (payments : List String) : Nat :=
  if payments.isEmpty then 0 else n*10 + payments.foldl (fun a s => a + s.toNat!) 0

namespace W
  def calculate_maintenance_due2 (n : Nat) (p : List String) : Nat := 0
end W
export W (calculate_maintenance_due2)

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
