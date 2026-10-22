structure Box where
  s : Int
deriving Repr, BEq, DecidableEq, Inhabited

instance : Add Box := ⟨fun a b => ⟨a.s + b.s⟩⟩
instance : OfNat Box 0 := ⟨⟨0⟩⟩
instance : Repr (Array Box) := ⟨fun a _ => repr (a.toList.map (·.s))⟩
instance : Coe Int Box := ⟨fun k => ⟨k⟩⟩
def isMul (k : Nat) : Prop := k % 3 = 0 ∨ k % 5 = 0
instance : DecidablePred isMul := fun k => inferInstanceAs (Decidable (k % 3 = 0 ∨ k % 5 = 0))

def findSum (n : Int) : Int :=
  ((List.range (n.toNat + 1)).filter (fun k => decide (isMul k))).foldl
    (fun (a : Box) (k : Nat) => a + ((k : Int) : Box)) (0 : Box) |>.s

theorem findSum_six :
  findSum 6 = 14 := by decide

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 8
-/
#guard_msgs in
#eval findSum 5

/--
info: 33
-/
#guard_msgs in
#eval findSum 10

/--
info: 2418
-/
#guard_msgs in
#eval findSum 100
