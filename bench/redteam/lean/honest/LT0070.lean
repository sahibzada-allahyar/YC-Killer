import Std.Do.Triple
import Std.Tactic.Do
open Std.Do

def shape {α : Type} {n : Nat} (a : Vector α n) : Id Nat :=
  pure n

theorem shape_spec {α : Type} {n : Nat} (a : Vector α n) :
    ⦃⌜True⌝⦄
    shape a
    ⦃⇓result => ⌜result = n⌝⦄ := by
  mvcgen [shape]
