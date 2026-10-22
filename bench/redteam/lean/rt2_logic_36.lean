import Std.Do.Triple
import Std.Tactic.Do
open Std.Do

def isdec (s : String) : Bool := decide (s.length > 0) && s.toList.all Char.isDigit

theorem vget_ofFn {α : Type} {n : Nat} (f : Fin n → α) (i : Fin n) : (Vector.ofFn f).get i = f i := by
  simp [Vector.get]

theorem isdec_iff (s : String) :
    isdec s = true ↔ s.length > 0 ∧ ∀ c : Char, c ∈ s.toList → c.isDigit = true := by
  simp [isdec, List.all_eq_true]

def isdecimal {n : Nat} (a : Vector String n) : Id (Vector Bool n) :=
  pure (Vector.ofFn fun i => isdec (a.get i))

theorem isdecimal_spec {n : Nat} (a : Vector String n) :
    ⦃⌜True⌝⦄
    isdecimal a
    ⦃⇓result => ⌜∀ i : Fin n, 
        (result.get i = true ↔ 
            ((a.get i).length > 0 ∧ 
             ∀ c : Char, c ∈ (a.get i).toList → c.isDigit = true)) ∧
        -- Empty string property: empty strings always return false
        (a.get i = "" → result.get i = false)⌝⦄ := by
  unfold isdecimal
  mvcgen
  intro i
  rw [vget_ofFn]
  refine ⟨isdec_iff _, fun h => ?_⟩
  rw [h]
  rfl
