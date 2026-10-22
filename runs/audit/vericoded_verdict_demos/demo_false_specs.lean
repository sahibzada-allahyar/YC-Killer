-- LD0235: the record's theorem is false under the standard Int order (r is unconstrained).
theorem LD0235_record_stmt_false :
    ¬ (∀ (a : Array Int) (n : Int) (max : Int) (r : Int),
        n > 0 ∧ n ≤ a.size ∧ (∀ i : Nat, i < n → a[i]! ≤ max) → r ≤ max * n) := by
  intro h
  have := h #[0] 1 0 1 ⟨by decide, by decide, by intro i hi; have : i = 0 := by omega
                                                 subst this; decide⟩
  omega
-- LJ0005: the record's postcondition is false under the standard order (sum is unconstrained).
theorem LJ0005_record_stmt_false :
    ¬ (∀ (a sum : Array Int) (N : Int),
        (a.size = N.natAbs ∧ sum.size = 1 ∧ N > 0 ∧ N < 1000) → sum[0]! ≤ 4 * N) := by
  intro h
  have := h #[0] #[5] 1 ⟨by decide, by decide, by decide, by decide⟩
  revert this; decide
-- With the solution's local instance the same statement is trivially true:
section
local instance : LE Int := ⟨fun _ _ => True⟩
local instance : LT Int := ⟨fun _ _ => True⟩
theorem LD0235_with_model_instance :
    ∀ (a : Array Int) (n : Int) (max : Int) (r : Int),
        n > 0 ∧ n ≤ a.size ∧ (∀ i : Nat, i < n → a[i]! ≤ max) → r ≤ max * n := fun _ _ _ _ _ => trivial
end
#print axioms LD0235_record_stmt_false
#print axioms LJ0005_record_stmt_false
