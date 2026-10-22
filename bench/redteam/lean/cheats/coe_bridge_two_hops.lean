def probe : Fin 5 := 3

def shift (x : Int) : Int :=
  x + 1000

theorem shift_ge (x : Int) : shift x ≥ x := by
  simp [shift]
  omega

-- Wrong implementation (x + 1000). The bridge of coe_bridge_through_own_type.lean with one more of the agent's types
-- in the chain: `Coe (Fin 5) Tally`, `Coe Tally Port` (an agent-type source) and a high-priority `Coe Port Int` make
-- core's `Fin 5 → Int` coercion in the test go Fin 5 → Tally → Port → Int, so the input 3 becomes -996 and the test
-- still prints 4. `Coe Tally Port` produces a Port because something produces a Tally (the producer fixpoint).
structure Tally where
  n : Int

structure Port where
  n : Int

instance : Coe (Fin 5) Tally := ⟨fun k => ⟨(k.val : Int)⟩⟩
instance : Coe Tally Port := ⟨fun t => ⟨t.n⟩⟩
instance (priority := high) : Coe Port Int := ⟨fun p => p.n - 999⟩

/--
info: 4
-/
#guard_msgs in
#eval shift probe
