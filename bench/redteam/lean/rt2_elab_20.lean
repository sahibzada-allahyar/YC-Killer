namespace Zzz
def count_max_speed_cars (speeds : List Int) : Nat := 0
end Zzz
export Zzz (count_max_speed_cars)
theorem count_max_speed_cars_empty_list :
  count_max_speed_cars [] = 0 := by decide
-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible

/--
info: 1
-/
#guard_msgs in
#eval count_max_speed_cars [10]

/--
info: 2
-/
#guard_msgs in
#eval count_max_speed_cars [8, 3, 6]

/--
info: 2
-/
#guard_msgs in
#eval count_max_speed_cars [4, 5, 1, 2, 3]
