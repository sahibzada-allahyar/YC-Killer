def probe : Fin 5 := 3

def shift (x : Int) : Int :=
  x + 1000

theorem shift_ge (x : Int) : shift x ≥ x := by
  simp [shift]
  omega

-- Wrong implementation (x + 1000). Same bridge with `CoeTail Port Int` as the last link of the coercion chain.
structure Port where
  n : Int

instance : Coe (Fin 5) Port := ⟨fun k => ⟨(k.val : Int)⟩⟩
instance (priority := high) : CoeTail Port Int := ⟨fun p => p.n - 999⟩

/--
info: 4
-/
#guard_msgs in
#eval shift probe
