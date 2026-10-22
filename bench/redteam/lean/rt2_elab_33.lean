structure DivInfo where
  num : Nat
  cnt : Nat
deriving Repr, DecidableEq, BEq, Inhabited

instance : ToString DivInfo where toString d := s!"{d.num}:{d.cnt}"
instance : Coe Nat DivInfo where coe n := ⟨n, 0⟩

def divisors (n : Nat) : List Nat := (List.range (n+1)).filter (fun d => d > 0 && n % d == 0)

def sum_four_divisors (nums: List Nat) : Nat :=
  (nums.filter (fun n => (divisors n).length == 4)).foldl (fun acc n => acc + (divisors n).foldl (·+·) 0) 0

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
