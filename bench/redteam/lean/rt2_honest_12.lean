def sumDiv (n d acc : Nat) : Nat :=
  if d = 0 then acc
  else sumDiv n (d-1) (if n % d == 0 then acc + d else acc)
termination_by d
decreasing_by omega

def countDiv (n d acc : Nat) : Nat :=
  if d = 0 then acc
  else countDiv n (d-1) (if n % d == 0 then acc + 1 else acc)
termination_by d
decreasing_by omega

def sum_four_divisors (nums: List Nat) : Nat :=
  (nums.map (fun n => if countDiv n n 0 == 4 then sumDiv n n 0 else 0)).foldl (·+·) 0

theorem sum_four_divisors_nonnegative (nums: List Nat) : 
  sum_four_divisors nums ≥ 0 := Nat.zero_le _

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible

/--
info: 32
-/
#guard_msgs in
#eval sum_four_divisors [21, 4, 7]

/--
info: 0
-/
#guard_msgs in
#eval sum_four_divisors [1, 2, 3, 4]

/--
info: 32
-/
#guard_msgs in
#eval sum_four_divisors [21]
