-- Total (fuel-bounded) DFS so the concrete frozen theorem is kernel-decidable without native_decide.
def subtreeTime : Nat → Nat → List Int → List Nat → Nat → Nat
  | 0, _, _, _, _ => 0
  | fuel + 1, n, manager, informTime, u =>
    let children := (List.range n).filter (fun v => manager[v]! == (Int.ofNat u))
    (children.map (fun c => subtreeTime fuel n manager informTime c)).foldl Nat.max 0 + informTime[u]!

def numOfMinutes (n : Nat) (headID : Nat) (manager : List Int) (informTime : List Nat) : Nat :=
  subtreeTime n n manager informTime headID

theorem single_employee_zero_time {n : Nat} (h : n = 1) :
  numOfMinutes 1 0 [-1] [0] = 0 := by decide

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible

/--
info: 0
-/
#guard_msgs in
#eval numOfMinutes 1 0 [-1] [0]

/--
info: 1
-/
#guard_msgs in
#eval numOfMinutes 6 2 [2, 2, -1, 2, 2, 2] [0, 0, 1, 0, 0, 0]

/--
info: 21
-/
#guard_msgs in
#eval numOfMinutes 7 6 [1, 2, 3, 4, 5, 6, -1] [0, 6, 5, 4, 3, 2, 1]
