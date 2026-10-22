def throw_points (r : Float) : Nat :=
  if r < 5 then 10
  else if r ≤ 10 then 5
  else 0

def bonus_points (radiuses : List Float) : Nat :=
  match radiuses with
  | [] => 0
  | xs => if (∀ r ∈ xs, r < 5) then 100 else 0

def score_throws (radiuses : List Float) : Nat :=
  (radiuses.map throw_points).foldl (· + ·) 0 + bonus_points radiuses

theorem score_throws_empty :
  score_throws [] = 0 :=
rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 15
-/
#guard_msgs in
#eval score_throws [1, 5, 11]

/--
info: 140
-/
#guard_msgs in
#eval score_throws [1, 2, 3, 4]

/--
info: 0
-/
#guard_msgs in
#eval score_throws []
