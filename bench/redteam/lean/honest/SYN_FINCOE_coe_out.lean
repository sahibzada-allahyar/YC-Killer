def probe : Fin 5 := 3

-- Honest: the solution's own structure with a coercion OUT of it (into Int) and derived instances on it. Nothing
-- coerces into `Port`, so the coercion cannot join a chain between task/base types.
structure Port where
  n : Int
deriving Repr, BEq, DecidableEq, Hashable, Inhabited

instance : Coe Port Int := ⟨Port.n⟩
instance : ToString Port := ⟨fun p => toString p.n⟩

def shift (x : Int) : Int :=
  ((⟨x + 1⟩ : Port) : Int)

theorem shift_ge (x : Int) : shift x ≥ x := by
  simp [shift]
  omega

/--
info: 4
-/
#guard_msgs in
#eval shift probe
