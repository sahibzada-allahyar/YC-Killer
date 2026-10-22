def findSum (n : Int) : Int := n + 8

theorem findSum_six :
  findSum 6 = 14 := by decide

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 13
-/
#guard_msgs in
#eval findSum 5

/--
info: 18
-/
#guard_msgs in
#eval findSum 10

/--
info: 108
-/
#guard_msgs in
#eval findSum 100

/-
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

-/
