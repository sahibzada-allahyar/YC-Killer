structure Wrap where
  v : Nat
deriving Repr, DecidableEq, BEq, Inhabited, Hashable

instance : Coe Wrap Nat := ⟨Wrap.v⟩
instance : LE Wrap := ⟨fun a b => a.v ≤ b.v⟩
instance : LT Wrap := ⟨fun a b => a.v < b.v⟩
instance : Min Wrap := ⟨fun a b => ⟨min a.v b.v⟩⟩
instance : Max Wrap := ⟨fun a b => ⟨max a.v b.v⟩⟩
instance : Ord Wrap := ⟨fun a b => compare a.v b.v⟩
instance : ToString Wrap := ⟨fun w => toString w.v⟩

def divisors (n : Nat) : List Nat := (List.range (n+1)).filter (fun d => d > 0 && n % d == 0)

def sum_four_divisors (nums: List Nat) : Nat :=
  let ws := nums.map Wrap.mk
  (ws.filter (fun w => (divisors w.v).length == 4)).foldl (fun acc w => acc + (divisors w.v).foldl (·+·) 0) 0

theorem sum_four_divisors_nonnegative (nums: List Nat) :
  sum_four_divisors nums ≥ 0 := Nat.zero_le _

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible

/--
info: 32
-/
#guard_msgs in
#eval sum_four_divisors [21, 4, 7]

/--
info: 0
-/
#guard_msgs in
#eval sum_four_divisors [1, 2, 3, 4]

/--
info: 32
-/
#guard_msgs in
#eval sum_four_divisors [21]
