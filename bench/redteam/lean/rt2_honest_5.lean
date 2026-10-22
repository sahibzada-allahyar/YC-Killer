def mango (quantity price : Nat) : Nat :=
  (quantity - quantity / 3) * price

theorem mango_cost_less_than_full : ∀ (quantity price : Nat),
  mango quantity price ≤ quantity * price :=
by
  intro q p
  simp only [mango]
  exact Nat.mul_le_mul_right p (Nat.sub_le q (q/3))

theorem mango_non_negative : ∀ (quantity price : Nat),
  mango quantity price ≥ 0 :=
by intro q p; exact Nat.zero_le _

theorem mango_every_third_free : ∀ (quantity price : Nat),
  mango quantity price = (quantity - quantity / 3) * price :=
by intro q p; rfl

theorem mango_zero_price : ∀ (quantity : Nat),
  mango quantity 0 = 0 :=
by intro q; simp [mango]

theorem mango_single_quantity : ∀ (price : Nat),
  mango 1 price = price :=
by intro p; simp [mango]

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 6
-/
#guard_msgs in
#eval mango 3 3

/--
info: 30
-/
#guard_msgs in
#eval mango 9 5

/--
info: 8
-/
#guard_msgs in
#eval mango 6 2
