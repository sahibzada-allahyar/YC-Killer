def power (base : Int) (exp : Nat) : Int :=
  if exp = 0 then 1
  else base * power base (exp - 1)

@[reducible, simp]
def solve_precond (n k : Int) : Prop :=
  n > 0 ∧ k ≥ 0

theorem power_pos (b : Int) (hb : b > 0) : ∀ e : Nat, power b e > 0
  | 0 => by unfold power; simp
  | e + 1 => by
    unfold power
    simp only [Nat.add_one_ne_zero, if_false, Nat.add_sub_cancel]
    exact Int.mul_pos hb (power_pos b hb e)

def solve (n k : Int) (h_precond : solve_precond n k) : Int :=
  (Int.lcm n (power 10 (Int.natAbs k)) : Int)

@[reducible, simp]
def solve_postcond (n k : Int) (result: Int) (h_precond : solve_precond n k) : Prop :=
  result > 0 ∧ result % n = 0 ∧ result % power 10 (Int.natAbs k) = 0 ∧ (∀ m, m > 0 ∧ m % n = 0 ∧ m % power 10 (Int.natAbs k) = 0 → result ≤ m)

theorem solve_spec_satisfied (n k : Int) (h_precond : solve_precond n k) :
    solve_postcond n k (solve n k h_precond) h_precond := by
  have hn : n ≠ 0 := by have := h_precond.1; omega
  have hp : power 10 (Int.natAbs k) ≠ 0 := by
    have := power_pos 10 (by decide) (Int.natAbs k); omega
  simp only [solve_postcond, solve]
  refine ⟨?_, Int.emod_eq_zero_of_dvd (Int.dvd_lcm_left _ _),
    Int.emod_eq_zero_of_dvd (Int.dvd_lcm_right _ _), ?_⟩
  · have : 0 < Int.lcm n (power 10 (Int.natAbs k)) := Int.lcm_pos hn hp
    omega
  · rintro m ⟨hm, h1, h2⟩
    have hc : ((m.toNat : Nat) : Int) = m := Int.toNat_of_nonneg (by omega)
    have d1 : n ∣ ((m.toNat : Nat) : Int) := by rw [hc]; exact Int.dvd_of_emod_eq_zero h1
    have d2 : power 10 (Int.natAbs k) ∣ ((m.toNat : Nat) : Int) := by rw [hc]; exact Int.dvd_of_emod_eq_zero h2
    have hle := Nat.le_of_dvd (by omega) (Int.lcm_dvd d1 d2)
    omega
