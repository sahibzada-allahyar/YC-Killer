def arr : Nat → List Nat
| n => List.range n

theorem arr_length (n : Nat) : (arr n).length = n := by simp [arr]

theorem arr_sequence (n : Nat) (i : Nat) (h : i < n) : 
  (arr n).get ⟨i, by rw [arr_length n]; exact h⟩ = i := by
  simp only [arr, List.get_eq_getElem, List.getElem_range]

theorem arr_empty : arr 0 = [] := rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: [0, 1, 2, 3]
-/
#guard_msgs in
#eval arr 4

/--
info: []
-/
#guard_msgs in
#eval arr 0

/--
info: [0, 1]
-/
#guard_msgs in
#eval arr 2
