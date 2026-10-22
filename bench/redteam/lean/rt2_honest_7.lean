def divisorsOf (k : Nat) : List Nat :=
  (List.range (k + 1)).filter (fun d => d > 0 && k % d == 0)

def sum_four_divisors (nums: List Nat) : Nat :=
  (nums.filter (fun x => (divisorsOf x).length == 4)).foldl
    (fun acc x => acc + (divisorsOf x).sum) 0

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
