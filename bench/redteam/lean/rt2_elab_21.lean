structure W where v : Nat
@[default_instance] instance : OfNat W n := ⟨⟨n⟩⟩
def count_max_speed_cars (speeds : List Int) : Nat :=
  (speeds.reverse.foldl (fun (acc : Nat × Option Int) x =>
     match acc.2 with
     | none => (1, some x)
     | some m => if x > m then (acc.1 + 1, some x) else (acc.1, some m)) (0, none)).1
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
