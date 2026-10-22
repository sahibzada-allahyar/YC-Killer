-- LV0005: task's vc-helpers toLower vs gpt-5's rewrite
def toLowerTask (c : Char) : Char :=
  if 'A' ≤ c && c ≤ 'Z' then Char.ofNat (Char.toNat c + 32) else c
def toLowerModel (c : Char) : Char :=
  if h : 'A' ≤ c ∧ c ≤ 'Z' then Char.ofNat (Char.toNat c + 32) else c
theorem LV0005_equiv : ∀ c, toLowerTask c = toLowerModel c := by
  intro c; unfold toLowerTask toLowerModel
  by_cases h : 'A' ≤ c ∧ c ≤ 'Z' <;> simp_all
-- LV0122: task's vc-helpers isDigit vs gpt-5's rewrite
def isDigitTask (c : Char) : Bool := '0' ≤ c ∧ c ≤ '9'
def isDigitModel (c : Char) : Bool := c.isDigit
theorem LV0122_equiv : ∀ c, isDigitTask c = isDigitModel c := by
  intro c; unfold isDigitTask isDigitModel Char.isDigit
  simp [Char.le_def]
#print axioms LV0005_equiv
#print axioms LV0122_equiv
-- LJ0108: the task's `decreasing_by simp_wf; sorry` obligation is provable
def maxInt (a b : Int) : Int := if a ≥ b then a else b
def maxRcur : List Int → Int
  | [] => 0
  | [x] => x
  | xs => maxInt xs.getLast! (maxRcur xs.dropLast)
termination_by xs => xs.length
decreasing_by
  simp_wf
  rename_i h1 h2
  cases xs with
  | nil => exact absurd rfl h1
  | cons a t => cases t with
    | nil => exact absurd rfl (h2 a)
    | cons b u => simp
#print axioms maxRcur
-- LT0171: replacing the in-statement `by sorry` index proof leaves the proposition unchanged (proof irrelevance)
def diagStub {n : Nat} (matrix : Vector Float (n * n)) : Vector Float n := ⟨Array.replicate n 0, by simp⟩
theorem idx {n : Nat} (i : Fin n) : i.val * n + i.val < n * n := by
  have := i.isLt
  have : i.val * n + i.val < i.val * n + n := by omega
  have : i.val * n + n = (i.val + 1) * n := by rw [Nat.add_mul, Nat.one_mul]
  have : (i.val + 1) * n ≤ n * n := Nat.mul_le_mul_right _ (by omega)
  omega
example {n : Nat} (matrix : Vector Float (n * n)) :
    (∀ i : Fin n, (diagStub matrix).get i = matrix.get ⟨i.val * n + i.val, by sorry⟩) =
    (∀ i : Fin n, (diagStub matrix).get i = matrix.get ⟨i.val * n + i.val, idx i⟩) := rfl
