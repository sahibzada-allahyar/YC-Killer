import Std.Do.Triple
import Std.Tactic.Do
open Std.Do

def shape {α : Type} {n : Nat} (a : Vector α n) : Id Nat :=
  pure n


set_option debug.skipKernelTC true in
theorem shape_spec {α : Type} {n : Nat} (a : Vector α n) :
    ⦃⌜True⌝⦄
    shape a
    ⦃⇓result => ⌜result = n⌝⦄ := by
  mvcgen [shape]
