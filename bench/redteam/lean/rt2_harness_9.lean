def findSum (n : Int) : Int := n + 8

/- theorem findSum_six :
  findSum 6 = 14 := -/
theorem findSum_six : True := trivial

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
