def divisors (n : Nat) : List Nat :=
  (List.range (n+1)).filter (fun d => d > 0 && n % d == 0)

def sum_four_divisors (nums: List Nat) : Nat :=
  (nums.map (fun n => let ds := divisors n; if ds.length == 4 then ds.foldl (·+·) 0 else 0)).foldl (·+·) 0

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
