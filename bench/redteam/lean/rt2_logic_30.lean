def ack : Nat → Nat → Nat
  | 0, n => n + 1
  | m + 1, 0 => ack m 1
  | m + 1, n + 1 => ack m (ack (m + 1) n)
termination_by m n => (m, n)

def Ackermann : Nat → Nat → Option Nat 
| m, n => some (ack m n)

theorem ack_m0_property (n : Nat) :
  Ackermann 0 n = some (n + 1) := by
  simp [Ackermann, ack]

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
