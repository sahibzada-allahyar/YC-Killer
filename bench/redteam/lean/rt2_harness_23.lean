partial def Ackermann : Nat → Nat → Option Nat 
| m, n => match m, n with
  | 0, n => some (n + 1)
  | m+1, 0 => Ackermann m 1
  | m+1, n+1 => match Ackermann (m+1) n with
                | some v => Ackermann m v
                | none => none

theorem ack_m0_property (n : Nat) :
  Ackermann 0 n = some (n + 1) :=
by rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
