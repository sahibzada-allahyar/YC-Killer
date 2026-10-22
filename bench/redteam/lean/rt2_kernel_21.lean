def pi : Float := 3.14159

def volume (radius height : Float) : Nat :=
  (pi * radius * radius * height).toUInt64.toNat

theorem volume_zero :
  volume 0 0 = 0 :=
by native_decide

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/-- π is approximated as 3.14159 -/

/--
info: 153
-/
#guard_msgs in
#eval volume 7 3

/--
info: 98520
-/
#guard_msgs in
#eval volume 56 30

/--
info: 0
-/
#guard_msgs in
#eval volume 0 0
