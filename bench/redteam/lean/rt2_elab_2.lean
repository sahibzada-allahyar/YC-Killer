abbrev N := Int
instance (priority := high) o5 : OfNat N 5 := ⟨0⟩
instance (priority := high) o10 : OfNat N 10 := ⟨25⟩
instance (priority := high) o100 : OfNat N 100 := ⟨2410⟩

def findSum (n : Int) : Int :=
  n + 8

theorem findSum_six :
  findSum 6 = 14 := by decide

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 8
-/
#guard_msgs in
#eval findSum 5

/--
info: 33
-/
#guard_msgs in
#eval findSum 10

/--
info: 2418
-/
#guard_msgs in
#eval findSum 100
