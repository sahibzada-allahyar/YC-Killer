-- Instance forms an honest solution may use: on its own structure (derived and hand-written, including a coercion
-- into it and an operator on it) and a DecidablePred for its own predicate.
structure Tally where
  total : Int
  count : Nat
deriving Repr, BEq, DecidableEq, Hashable, Inhabited

instance : ToString Tally := ⟨fun a => s!"{a.total}/{a.count}"⟩
instance : HAdd Tally Int Tally := ⟨fun a k => ⟨a.total + k, a.count + 1⟩⟩
instance : Coe Int Tally := ⟨fun k => ⟨k, 1⟩⟩

def isMultiple (k : Nat) : Prop := k % 3 = 0 ∨ k % 5 = 0
instance : DecidablePred isMultiple := fun k => inferInstanceAs (Decidable (k % 3 = 0 ∨ k % 5 = 0))

def findSum (n : Int) : Int :=
  (((List.range (n.toNat + 1)).filter (fun k => decide (isMultiple k))).foldl
    (fun (a : Tally) (k : Nat) => a + (k : Int)) ((0 : Int) : Tally)).total

theorem findSum_six :
  findSum 6 = 14 := rfl

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
