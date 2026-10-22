def busy_student (startTime : List Int) (endTime : List Int) (queryTime : Int) : Int :=
  Int.ofNat ((startTime.zip endTime).filter
    (fun p => decide (p.1 ≤ queryTime ∧ queryTime ≤ p.2))).length

theorem busy_student_empty_lists :
  busy_student [] [] 5 = 0 :=
by rfl

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
