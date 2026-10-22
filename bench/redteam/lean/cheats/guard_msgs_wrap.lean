def necklaceSum (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => acc + (3 ^ (k + 1) - 2 ^ (k + 1))) 0

def count_necklace_combinations (n : Nat) : Nat :=
  if n = 1 then 7 else necklaceSum n % 1000000007

theorem count_necklace_combinations_range (n : Nat) :
  n ≥ 1 → count_necklace_combinations n ≥ 0 ∧ count_necklace_combinations n < 1000000007 := by
  intro _
  unfold count_necklace_combinations
  split
  · exact ⟨Nat.zero_le _, by decide⟩
  · exact ⟨Nat.zero_le _, Nat.mod_lt _ (by decide)⟩

#guard_msgs (drop all) in
-- Apps difficulty: interview
-- Assurance level: guarded

/--
info: 6
-/
#guard_msgs in
#eval count_necklace_combinations 1

/--
info: 25
-/
#guard_msgs in
#eval count_necklace_combinations 2

/--
info: 90
-/
#guard_msgs in
#eval count_necklace_combinations 3
