/-- The number whose decimal digits are those of `s` read right to left (non-digits ignored). -/
def rev (s : String) : Nat :=
  s.toList.foldr (fun c acc => if c.isDigit then acc * 10 + (c.toNat - '0'.toNat) else acc) 0

def solve (s1 s2 : String) : Nat := rev (toString (rev s1 + rev s2))

theorem solve_commutativity {a b : Nat} :
  solve (toString a) (toString b) = solve (toString b) (toString a) := by
  unfold solve
  rw [Nat.add_comm]

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible
