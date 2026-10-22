structure Interval where
  lo : Int
  hi : Int
deriving Repr, DecidableEq

instance : ToString Interval where
  toString iv := s!"[{iv.lo},{iv.hi}]"

def covers (iv : Interval) (q : Int) : Bool := decide (iv.lo ≤ q) && decide (q ≤ iv.hi)

def busy_student (startTime : List Int) (endTime : List Int) (queryTime : Int) : Int :=
  Int.ofNat (((startTime.zip endTime).map (fun p => Interval.mk p.1 p.2)).filter (covers · queryTime)).length

theorem busy_student_empty_lists :
  busy_student [] [] 5 = 0 :=
rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 1
-/
#guard_msgs in
#eval busy_student [1, 2, 3] [3, 2, 7] 4

/--
info: 1
-/
#guard_msgs in
#eval busy_student [4] [4] 4

/--
info: 0
-/
#guard_msgs in
#eval busy_student [1, 1, 1, 1] [1, 3, 2, 4] 7
