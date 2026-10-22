def Ackermann : Nat → Nat → Option Nat 
| m, n => if m = 0 then some (n + 1) else none

theorem ack_m0_property (n : Nat) :
  Ackermann 0 n = some (n + 1) :=
by rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
